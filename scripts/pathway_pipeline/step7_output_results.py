"""Step 7: write ranked pathways and the summary JSON.

This final mandatory step materializes the in-memory ranking results produced in
step 6 and writes a compact JSON summary for run auditing.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from process_chebi_to_pathways_v2 import build_summary

from pathway_pipeline.cli_utils import build_context, build_parser, format_counter, print_summary
from pathway_pipeline.context import PipelineContext
from pathway_pipeline.step1_alias_standardization import run as run_step1
from pathway_pipeline.step2_map_names_to_kegg import run as run_step2
from pathway_pipeline.step3_link_compounds_to_pathways import run as run_step3
from pathway_pipeline.step4_map_to_ath import run as run_step4
from pathway_pipeline.step5_annotate_pathways import run as run_step5
from pathway_pipeline.step6_score_pathways import run as run_step6


def run(context: PipelineContext) -> PipelineContext:
    """Write the final ranked pathway table and batch summary."""

    # The ranked TSV is the primary user-facing batch output. It keeps the score
    # plus the feature-level explanation fields emitted by step 6.
    with context.paths.pathway_output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "compound_id",
                "chebi_accession",
                "chebi_name",
                "pathway_rank",
                "score",
                "confidence_level",
                "evidence_type",
                "mapping_confidence",
                "support_kegg_compound_ids",
                "support_kegg_names",
                "pathway_target_id",
                "pathway_target_type",
                "ath_pathway_id",
                "map_pathway_id",
                "pathway_name",
                "pathway_group",
                "pathway_category",
                "map_pathway_compound_count",
                "ath_gene_count",
                "plantcyc_support_source",
                "plantcyc_support_examples",
                "reactome_matches",
                "top_positive_features",
                "top_negative_features",
                "feature_contributions_json",
                "reason",
            ],
            delimiter="\t",
        )
        writer.writeheader()

        # Write rows compound by compound to keep output order deterministic and
        # aligned with the original input compound ordering.
        for compound_id in sorted(context.compounds, key=int):
            for row in context.ranked_pathway_rows.get(compound_id, []):
                writer.writerow(
                    {
                        "compound_id": row.compound_id,
                        "chebi_accession": row.chebi_accession,
                        "chebi_name": row.chebi_name,
                        "pathway_rank": row.pathway_rank,
                        "score": f"{row.score:.3f}",
                        "confidence_level": row.confidence_level,
                        "evidence_type": row.evidence_type,
                        "mapping_confidence": f"{row.mapping_confidence:.3f}",
                        "support_kegg_compound_ids": ";".join(row.support_kegg_compound_ids),
                        "support_kegg_names": ";".join(row.support_kegg_names),
                        "pathway_target_id": row.pathway_target_id,
                        "pathway_target_type": row.pathway_target_type,
                        "ath_pathway_id": row.ath_pathway_id,
                        "map_pathway_id": row.map_pathway_id,
                        "pathway_name": row.pathway_name,
                        "pathway_group": row.pathway_group,
                        "pathway_category": row.pathway_category,
                        "map_pathway_compound_count": row.map_pathway_compound_count,
                        "ath_gene_count": row.ath_gene_count,
                        "plantcyc_support_source": row.plantcyc_support_source,
                        "plantcyc_support_examples": row.plantcyc_support_examples,
                        "reactome_matches": row.reactome_matches,
                        "top_positive_features": row.top_positive_features,
                        "top_negative_features": row.top_negative_features,
                        "feature_contributions_json": row.feature_contributions_json,
                        "reason": row.reason,
                    }
                )

    # Reuse the proven v2 summary builder so the refactored pipeline reports the
    # same audit-level counts and database usage notes as the monolithic script.
    summary = build_summary(
        compounds=context.compounds,
        mapping_status=context.mapping_status,
        pathway_status=context.pathway_status,
        alias_rows=context.alias_rows,
        comments_profile=context.comments_profile,
        plantcyc_records=context.plantcyc_records,
        lipidmaps_records=context.lipidmaps_records,
        pubchem_stats=context.pubchem_stats,
    )

    # Append notes produced by step wrappers, including optional step-8/9 hook
    # messages when those placeholders are enabled.
    if context.step_notes:
        summary.setdefault("notes", []).extend(context.step_notes)

    # Persist the summary as JSON for quick inspection and regression checking.
    with context.paths.summary_output_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    return context


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for standalone step-7 execution."""

    return build_parser(
        description="Run pathway pipeline steps 1-7 and write the final ranked pathway TSV plus summary JSON.",
        default_output_tag="step7_cli",
    ).parse_args()


def main() -> None:
    """Run the full mandatory pipeline through step 7."""

    args = parse_args()
    context = build_context(workdir=args.workdir, output_tag=args.output_tag)
    run_step1(context)
    run_step2(context)
    run_step3(context)
    run_step4(context)
    run_step5(context)
    run_step6(context)
    run(context)
    print_summary(
        "Step 7 completed.",
        [
            f"Alias table: {context.paths.alias_output_path}",
            f"Mapping summary: {context.paths.mapping_summary_path}",
            f"Selected mappings: {context.paths.mapping_selected_path}",
            f"Ranked pathways: {context.paths.pathway_output_path}",
            f"Summary JSON: {context.paths.summary_output_path}",
            f"Pathway status counts: {format_counter(context.pathway_status)}",
        ],
    )


if __name__ == "__main__":
    main()
