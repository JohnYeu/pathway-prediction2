"""Step 1: alias normalization and alias-table materialization.

This module is responsible for all name-side preparation before KEGG mapping.
It does four things in order:

1. Load the core ChEBI compound table and profile comments.tsv.
2. Load alias / structure / cross-reference sources from ChEBI and external
   support databases.
3. Expand each ChEBI compound into a full per-compound alias context.
4. Write the fully standardized alias audit table used downstream.

The output of this step is intentionally verbose. Later steps only need part of
the information, but keeping the full alias table makes debugging and manual
review much easier.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from process_chebi_to_pathways_v2 import (
    XrefInfo,
    build_compound_context,
    build_name_formula_index,
    build_variants,
    load_base_aliases,
    load_comments_profile,
    load_compounds,
    load_formula_info,
    load_lipidmaps_records,
    load_plantcyc_compounds,
    load_pubchem_synonyms,
    load_structures,
    load_xrefs,
)

from pathway_pipeline.cli_utils import build_context, build_parser, print_summary
from pathway_pipeline.context import PipelineContext


def _inchi_key_prefix(value: str) -> str:
    """Return the connectivity block of an InChIKey when available."""

    text = (value or "").strip().upper()
    if not text:
        return ""
    return text.split("-", 1)[0]


def write_name_normalization_index(context: PipelineContext) -> int:
    """Write the canonical name + alias table consumed by query step 1."""

    row_count = 0
    with context.paths.name_normalization_index_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "compound_id",
                "chebi_accession",
                "canonical_name",
                "alias",
                "source_type",
                "is_primary_name",
                "exact_name",
                "compact_name",
                "singular_name",
                "stereo_stripped_name",
            ],
            delimiter="\t",
        )
        writer.writeheader()
        for compound_id in sorted(context.compounds, key=int):
            compound = context.compounds[compound_id]
            compound_context = context.compound_contexts[compound_id]
            for alias in sorted(compound_context.all_aliases, key=lambda item: (item.source_type, item.raw_name)):
                writer.writerow(
                    {
                        "compound_id": compound_id,
                        "chebi_accession": compound.chebi_accession,
                        "canonical_name": compound.name,
                        "alias": alias.raw_name,
                        "source_type": alias.source_type,
                        "is_primary_name": str(alias.source_type == "compound_name").lower(),
                        "exact_name": alias.exact,
                        "compact_name": alias.compact,
                        "singular_name": alias.singular,
                        "stereo_stripped_name": alias.stereo_stripped,
                    }
                )
                row_count += 1
    return row_count


def aggregate_name_formula_rows(context: PipelineContext) -> dict[tuple[str, str], dict[str, object]]:
    """Aggregate normalized names to structure signatures for weak-match validation."""

    rows: dict[tuple[str, str], dict[str, object]] = {}

    def add_name(name: str, formula_key_value: str, inchi_key_value: str, source: str) -> None:
        if not name:
            return
        variants = build_variants(name)
        key = (variants["exact"], variants["compact"])
        if not key[0] or not key[1]:
            return
        entry = rows.setdefault(
            key,
            {
                "exact_name": key[0],
                "compact_name": key[1],
                "formula_keys": set(),
                "inchi_key_prefixes": set(),
                "inchi_key_fulls": set(),
                "evidence_sources": set(),
            },
        )
        if formula_key_value:
            entry["formula_keys"].add(formula_key_value)
        normalized_inchi = (inchi_key_value or "").strip().upper()
        if normalized_inchi:
            entry["inchi_key_fulls"].add(normalized_inchi)
            prefix = _inchi_key_prefix(normalized_inchi)
            if prefix:
                entry["inchi_key_prefixes"].add(prefix)
        entry["evidence_sources"].add(source)

    for compound_id, aliases in context.base_aliases.items():
        structure = context.structures.get(compound_id)
        if not structure:
            continue
        for alias in aliases:
            add_name(alias.raw_name, structure.formula_key, structure.standard_inchi_key, "ChEBI")

    for record in context.plantcyc_records.values():
        add_name(record.common_name, record.formula_key, "", record.source_db)
        for synonym in sorted(record.synonyms):
            add_name(synonym, record.formula_key, "", record.source_db)

    for record in context.lipidmaps_records.values():
        add_name(record.common_name, record.formula_key, record.inchi_key, "LIPID MAPS")
        add_name(record.systematic_name, record.formula_key, record.inchi_key, "LIPID MAPS")
        for synonym in sorted(record.synonyms):
            add_name(synonym, record.formula_key, record.inchi_key, "LIPID MAPS")

    return rows


def write_name_to_formula_index(context: PipelineContext, rows: dict[tuple[str, str], dict[str, object]]) -> int:
    """Write the step-1 structure lookup used by weak query-time corrections."""

    with context.paths.name_to_formula_index_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "exact_name",
                "compact_name",
                "formula_keys",
                "formula_count",
                "inchi_key_prefixes",
                "inchi_key_prefix_count",
                "inchi_key_fulls",
                "inchi_key_full_count",
                "evidence_sources",
            ],
            delimiter="\t",
        )
        writer.writeheader()
        for key in sorted(rows):
            entry = rows[key]
            writer.writerow(
                {
                    "exact_name": entry["exact_name"],
                    "compact_name": entry["compact_name"],
                    "formula_keys": ";".join(sorted(entry["formula_keys"])),
                    "formula_count": len(entry["formula_keys"]),
                    "inchi_key_prefixes": ";".join(sorted(entry["inchi_key_prefixes"])),
                    "inchi_key_prefix_count": len(entry["inchi_key_prefixes"]),
                    "inchi_key_fulls": ";".join(sorted(entry["inchi_key_fulls"])),
                    "inchi_key_full_count": len(entry["inchi_key_fulls"]),
                    "evidence_sources": ";".join(sorted(entry["evidence_sources"])),
                }
            )
    return len(rows)


def run(context: PipelineContext) -> PipelineContext:
    """Load alias sources and write the normalized alias table.

    The returned context contains:
    - core ChEBI compounds
    - base ChEBI aliases
    - structure / formula / cross-reference evidence
    - PlantCyc / AraCyc / LIPID MAPS / PubChem support resources
    - a per-compound CompoundContext with all merged aliases
    - the step-1 alias output TSV
    """

    paths = context.paths

    # Fail early if any required input is missing. Step 1 depends on nearly all
    # reference inputs because alias expansion already pulls in external naming
    # sources such as PlantCyc, LIPID MAPS, and PubChem.
    paths.ensure_required_inputs()

    # Load the starting ChEBI compound table. This is the master list of
    # compounds that the whole pipeline iterates over.
    context.compounds = load_compounds(paths.compounds_path)

    # comments.tsv is not used as an alias source, but we still profile it so
    # the final summary can explain why it was excluded.
    context.comments_profile = load_comments_profile(paths.comments_path)

    # Load the standard ChEBI names and approved ChEBI alias types from
    # names.tsv.gz. This is the primary alias source before external expansion.
    context.base_aliases = load_base_aliases(context.compounds, paths.chebi_names_path)

    # Load curated ChEBI cross-references such as KEGG and PubChem IDs. These
    # IDs are later used both for strong mapping evidence and for synonym pulls.
    context.xrefs = load_xrefs(paths.chebi_database_accession_path)

    # Load formula information first, because structure loading enriches each
    # structure record with the preferred formula / formula_key when available.
    context.formulas = load_formula_info(paths.chebi_chemical_data_path)
    context.structures = load_structures(paths.chebi_structures_path, context.formulas)

    # Load PlantCyc and AraCyc compound tables. These provide additional names,
    # pathway hints, and external identifiers that may strengthen a compound's
    # support context even before KEGG matching begins.
    (
        context.plantcyc_records,
        context.plantcyc_indexes,
        plantcyc_pubchem_cids,
    ) = load_plantcyc_compounds(
        [
            ("AraCyc", paths.aracyc_compounds_path),
            ("PlantCyc", paths.plantcyc_compounds_path),
        ]
    )

    # Load LIPID MAPS records. This is especially important for lipid-like
    # compounds where names alone are often too ambiguous.
    (
        context.lipidmaps_records,
        context.lipidmaps_indexes,
        lipidmaps_pubchem_cids,
    ) = load_lipidmaps_records(paths.lipidmaps_sdf_path)

    # Build a normalized name -> formula index spanning ChEBI and selected
    # external sources. Step 2 uses this to guard typo-like fuzzy corrections:
    # if two near-identical names have incompatible formulas, they should not be
    # auto-corrected into the same compound.
    context.name_formula_index = build_name_formula_index(
        context.base_aliases,
        context.structures,
        context.plantcyc_records,
        context.lipidmaps_records,
    )

    # Collect the set of PubChem CIDs that are relevant to the current corpus.
    # We only load synonyms for CIDs that we can actually reach through ChEBI,
    # PlantCyc, or LIPID MAPS, which keeps the synonym expansion bounded.
    target_pubchem_cids = set(plantcyc_pubchem_cids) | set(lipidmaps_pubchem_cids)
    for info in context.xrefs.values():
        target_pubchem_cids.update(info.pubchem_cids)
    context.pubchem_synonyms, context.pubchem_stats = load_pubchem_synonyms(
        paths.pubchem_synonyms_path,
        target_pubchem_cids,
    )

    # Step 1 writes the alias audit table directly, because this table is useful
    # on its own and does not need to wait for KEGG matching.
    alias_rows = 0
    compound_contexts = {}
    with paths.alias_output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "compound_id",
                "chebi_accession",
                "chebi_name",
                "alias",
                "source_type",
                "normalized_name",
                "compact_name",
                "singular_name",
                "stereo_stripped_name",
            ],
            delimiter="\t",
        )
        writer.writeheader()

        # Build a full alias/support context for every ChEBI compound. This is
        # where base ChEBI aliases are merged with external names from
        # PlantCyc/AraCyc, LIPID MAPS, and PubChem.
        for compound_id in sorted(context.compounds, key=int):
            compound = context.compounds[compound_id]
            compound_context = build_compound_context(
                compound=compound,
                structure=context.structures.get(compound_id),
                xrefs=context.xrefs.get(compound_id, XrefInfo()),
                base_aliases=context.base_aliases.get(compound_id, []),
                plantcyc_records=context.plantcyc_records,
                plantcyc_indexes=context.plantcyc_indexes,
                lipidmaps_records=context.lipidmaps_records,
                lipidmaps_indexes=context.lipidmaps_indexes,
                pubchem_synonyms=context.pubchem_synonyms,
            )

            # Cache the merged context in memory so step 2 can reuse it without
            # having to rebuild aliases or re-scan external tables.
            compound_contexts[compound_id] = compound_context

            # Write one row per alias variant source. The output table keeps the
            # raw alias plus several normalized forms used in the matcher.
            for alias in sorted(compound_context.all_aliases, key=lambda item: (item.source_type, item.raw_name)):
                writer.writerow(
                    {
                        "compound_id": compound_id,
                        "chebi_accession": compound.chebi_accession,
                        "chebi_name": compound.name,
                        "alias": alias.raw_name,
                        "source_type": alias.source_type,
                        "normalized_name": alias.exact,
                        "compact_name": alias.compact,
                        "singular_name": alias.singular,
                        "stereo_stripped_name": alias.stereo_stripped,
                    }
                )
                alias_rows += 1

    # Persist the per-compound merged contexts and alias row count for later
    # steps and for the final JSON summary.
    context.compound_contexts = compound_contexts
    context.alias_rows = alias_rows
    context.preprocess_counts["compounds_total"] = len(context.compounds)
    context.preprocess_counts["comments_profile_keys"] = len(context.comments_profile)
    context.preprocess_counts["name_normalization_index"] = write_name_normalization_index(context)
    formula_rows = aggregate_name_formula_rows(context)
    context.preprocess_counts["name_to_formula_index"] = write_name_to_formula_index(context, formula_rows)
    context.preprocess_counts["pubchem_target_cids"] = context.pubchem_stats.get("target_cids", 0)
    context.add_note("Step 1 wrote the standardized alias table before KEGG mapping.")
    return context


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for standalone step-1 execution."""

    return build_parser(
        description="Run pathway pipeline step 1 only: alias normalization and alias-table writing.",
        default_output_tag="step1_cli",
    ).parse_args()


def main() -> None:
    """Run step 1 as a standalone terminal command."""

    args = parse_args()
    context = build_context(workdir=args.workdir, output_tag=args.output_tag)
    run(context)
    print_summary(
        "Step 1 completed.",
        [
            f"Alias table: {context.paths.alias_output_path}",
            f"Name normalization index: {context.paths.name_normalization_index_path}",
            f"Name-to-formula index: {context.paths.name_to_formula_index_path}",
            f"Compounds loaded: {len(context.compounds)}",
            f"Alias rows written: {context.alias_rows}",
            f"PubChem synonym CIDs loaded: {context.pubchem_stats.get('target_cids', 0)}",
        ],
    )


if __name__ == "__main__":
    main()
