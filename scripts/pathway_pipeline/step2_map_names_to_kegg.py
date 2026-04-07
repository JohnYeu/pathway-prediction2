"""Step 2: map normalized names to KEGG compound IDs.

This module takes the per-compound alias contexts prepared in step 1 and tries
to resolve them to KEGG compound IDs. The matching strategy itself lives in the
v2 helper code; this wrapper organizes the step into a table-aligned stage and
produces the step-2 audit files.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from process_chebi_to_pathways_v2 import (
    XrefInfo,
    build_candidate_mappings,
    load_kegg_compounds,
    mapping_confidence_label,
    mapping_method_label,
    reason_summary,
    select_candidates,
    serialize_support_context,
)

from pathway_pipeline.cli_utils import build_context, build_parser, format_counter, print_summary
from pathway_pipeline.context import PipelineContext
from pathway_pipeline.step1_alias_standardization import run as run_step1


def build_standard_name_indexes(kegg_compounds):
    """Materialize primary-name indexes from loaded KEGG rows.

    The KEGG loader already returns a full-name index and an alias-only index.
    For the refactored step flow we also build a dedicated standard-name index
    keyed only by primary_name variants. This keeps the "standard name first,
    alias second" rule explicit in the orchestration layer.
    """

    indexes = {
        "exact": defaultdict(set),
        "compact": defaultdict(set),
        "singular": defaultdict(set),
        "stereo_stripped": defaultdict(set),
    }

    # Each KEGG compound contributes its canonical label in several normalized
    # forms. Downstream matching can then try strict to relaxed variants.
    for kegg_id, kegg in kegg_compounds.items():
        indexes["exact"][kegg.primary_exact].add(kegg_id)
        indexes["compact"][kegg.primary_compact].add(kegg_id)
        indexes["singular"][kegg.primary_singular].add(kegg_id)
        indexes["stereo_stripped"][kegg.primary_stereo_stripped].add(kegg_id)
    return indexes


def run(context: PipelineContext) -> PipelineContext:
    """Build KEGG candidates, select final mappings, and write mapping tables.

    Outputs written here:
    - mapping summary table for every ChEBI compound
    - selected KEGG mapping table for trusted final mappings

    State captured for later steps:
    - full ranked candidate list per compound
    - final selected mappings per compound
    - compact support context retained only for selected compounds
    """

    # Load KEGG compound names and the supporting indexes used by the matching
    # logic. The v2 loader returns several specialized indexes; in this
    # refactored step we keep only the ones actually needed for the current
    # matching policy.
    (
        context.kegg_compounds,
        _all_indexes,
        _token_exact_indexes,
        _token_delete_indexes,
        _all_compact_delete_index,
        context.kegg_alias_indexes,
        context.kegg_primary_compact_delete_index,
    ) = load_kegg_compounds(context.paths.kegg_compound_list_path)

    # Build an explicit primary-name-only index so the matching order remains:
    # standard KEGG name -> KEGG alias -> guarded fuzzy correction.
    context.kegg_standard_indexes = build_standard_name_indexes(context.kegg_compounds)

    # These per-compound structures are both written to disk and carried forward
    # in memory for pathway expansion.
    selected_by_compound = {}
    ranked_candidates_by_compound = {}
    support_contexts = {}
    mapping_status = defaultdict(int)

    # Step 2 produces two different audit views:
    # 1. a summary row for every ChEBI compound
    # 2. one row per trusted selected KEGG mapping
    with context.paths.mapping_summary_path.open("w", newline="", encoding="utf-8") as mapping_handle, context.paths.mapping_selected_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as selected_handle:
        mapping_writer = csv.DictWriter(
            mapping_handle,
            fieldnames=[
                "compound_id",
                "chebi_accession",
                "chebi_name",
                "selection_status",
                "selected_kegg_compound_ids",
                "selected_kegg_names",
                "selected_scores",
                "top_candidate_kegg_compound_id",
                "top_candidate_name",
                "top_candidate_score",
                "candidate_count",
                "alias_count",
                "matched_pubchem_cids",
                "mapping_reason",
            ],
            delimiter="\t",
        )
        mapping_writer.writeheader()

        selected_writer = csv.DictWriter(
            selected_handle,
            fieldnames=[
                "compound_id",
                "chebi_accession",
                "chebi_name",
                "kegg_compound_id",
                "kegg_primary_name",
                "mapping_score",
                "mapping_confidence_level",
                "mapping_method",
                "best_alias",
                "best_alias_source",
                "best_variant",
                "evidence_count",
                "external_sources",
                "has_structure_evidence",
                "used_pubchem_synonym",
                "mapping_reason",
            ],
            delimiter="\t",
        )
        selected_writer.writeheader()

        # Process compounds one by one so the step output remains deterministic
        # and easy to diff across runs.
        for compound_id in sorted(context.compounds, key=int):
            compound = context.compounds[compound_id]
            compound_context = context.compound_contexts[compound_id]

            # Ask the shared v2 matching code to gather every plausible KEGG
            # candidate, accumulate evidence, and score the candidates.
            ranked = build_candidate_mappings(
                compound=compound,
                structure=context.structures.get(compound_id),
                xrefs=context.xrefs.get(compound_id, XrefInfo()),
                context=compound_context,
                plantcyc_records=context.plantcyc_records,
                lipidmaps_records=context.lipidmaps_records,
                kegg_compounds=context.kegg_compounds,
                kegg_standard_indexes=context.kegg_standard_indexes,
                kegg_alias_indexes=context.kegg_alias_indexes,
                kegg_primary_compact_delete_index=context.kegg_primary_compact_delete_index,
                name_formula_index=context.name_formula_index,
            )
            ranked_candidates_by_compound[compound_id] = ranked

            # Apply the conservative selection policy: direct curated mappings
            # are always kept; otherwise only clearly dominant high-scoring
            # candidates survive.
            selected = select_candidates(ranked)
            selected_by_compound[compound_id] = selected

            # Track whether the compound mapped cleanly, had unresolved
            # competition, or had no credible KEGG candidate at all.
            if selected:
                selection_status = "selected"
                support_contexts[compound_id] = serialize_support_context(compound_context)
            elif ranked:
                selection_status = "ambiguous"
            else:
                selection_status = "unmapped"
            mapping_status[selection_status] += 1

            # The summary table always records the top candidate, even if it was
            # ultimately rejected, so users can inspect near-misses manually.
            top = ranked[0] if ranked else None
            mapping_writer.writerow(
                {
                    "compound_id": compound_id,
                    "chebi_accession": compound.chebi_accession,
                    "chebi_name": compound.name,
                    "selection_status": selection_status,
                    "selected_kegg_compound_ids": ";".join(mapping.kegg_compound_id for mapping in selected),
                    "selected_kegg_names": ";".join(mapping.kegg_primary_name for mapping in selected),
                    "selected_scores": ";".join(f"{mapping.final_score:.3f}" for mapping in selected),
                    "top_candidate_kegg_compound_id": top.kegg_compound_id if top else "",
                    "top_candidate_name": top.kegg_primary_name if top else "",
                    "top_candidate_score": f"{top.final_score:.3f}" if top else "",
                    "candidate_count": len(ranked),
                    "alias_count": len(compound_context.all_aliases),
                    "matched_pubchem_cids": ";".join(sorted(compound_context.pubchem_cids)),
                    "mapping_reason": reason_summary(top) if top else "",
                }
            )

            # The selected table contains only the final trusted mappings and
            # exposes the strongest supporting evidence used to keep them.
            for mapping in selected:
                selected_writer.writerow(
                    {
                        "compound_id": compound_id,
                        "chebi_accession": compound.chebi_accession,
                        "chebi_name": compound.name,
                        "kegg_compound_id": mapping.kegg_compound_id,
                        "kegg_primary_name": mapping.kegg_primary_name,
                        "mapping_score": f"{mapping.final_score:.3f}",
                        "mapping_confidence_level": mapping_confidence_label(mapping.final_score),
                        "mapping_method": mapping_method_label(mapping),
                        "best_alias": mapping.best_alias,
                        "best_alias_source": mapping.best_source_type,
                        "best_variant": mapping.best_variant,
                        "evidence_count": mapping.evidence_count,
                        "external_sources": ";".join(sorted(mapping.external_sources)),
                        "has_structure_evidence": str(mapping.has_structure_evidence).lower(),
                        "used_pubchem_synonym": str(mapping.used_pubchem_synonym).lower(),
                        "mapping_reason": reason_summary(mapping),
                    }
                )

    # Persist both the full ranked candidate space and the final selections for
    # downstream pathway expansion, ranking, and reporting.
    context.ranked_candidates_by_compound = ranked_candidates_by_compound
    context.selected_by_compound = selected_by_compound
    context.support_contexts = support_contexts
    context.mapping_status.update(mapping_status)
    return context


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for standalone step-2 execution."""

    return build_parser(
        description="Run pathway pipeline steps 1-2: alias normalization plus KEGG compound mapping.",
        default_output_tag="step2_cli",
    ).parse_args()


def main() -> None:
    """Run steps 1-2 and write the step-2 outputs."""

    args = parse_args()
    context = build_context(workdir=args.workdir, output_tag=args.output_tag)
    run_step1(context)
    run(context)
    print_summary(
        "Step 2 completed.",
        [
            f"Alias table: {context.paths.alias_output_path}",
            f"Mapping summary: {context.paths.mapping_summary_path}",
            f"Selected mappings: {context.paths.mapping_selected_path}",
            f"Mapping status counts: {format_counter(context.mapping_status)}",
        ],
    )


if __name__ == "__main__":
    main()
