"""Step 7 (AraCyc-first): write the final ranked pathway table and metadata.

Adapted from the KEGG-first step7_output_results.py with AraCyc-native columns.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from process_chebi_to_pathways_v2 import build_summary

from pathway_pipeline.context import AraCycRankedRow, PipelineContext

PREPROCESS_VERSION = "aracyc_first_v1"

ARACYC_OUTPUT_FIELDS = [
    "compound_id",
    "chebi_accession",
    "chebi_name",
    "pathway_rank",
    "score",
    "confidence_level",
    "evidence_type",
    "match_method",
    "match_score",
    "aracyc_compound_id",
    "aracyc_compound_name",
    "source_db",
    "pathway_id",
    "pathway_name",
    "pathway_category",
    "gene_count",
    "gene_ids",
    "ec_numbers",
    "reaction_count",
    "compound_count",
    "go_best_term",
    "go_best_fdr",
    "plant_reactome_best_category",
    "plant_reactome_alignment_confidence",
    "annotation_confidence",
    "chebi_xref_direct",
    "structure_validated",
    "cofactor_like",
    "top_positive_features",
    "top_negative_features",
    "feature_contributions_json",
    "reason",
]


def build_preprocess_metadata(context: PipelineContext) -> dict[str, object]:
    """Build metadata for the AraCyc-first preprocess run."""

    total = context.aracyc_reference_total()
    with_pathway = len({
        matches[0].reference_compound_key
        for compound_id, rows in context.aracyc_ranked_rows.items()
        if rows
        for matches in [context.aracyc_matches_by_compound.get(compound_id, [])]
        if matches and matches[0].source_db == "AraCyc"
    })
    total_rows = sum(len(rows) for rows in context.aracyc_ranked_rows.values())

    return {
        "preprocess_version": PREPROCESS_VERSION,
        "pipeline": "aracyc_first",
        "coverage_basis": "aracyc_reference_compounds",
        "total_compounds": total,
        "chebi_input_total": len(context.compounds),
        "compounds_with_pathway": with_pathway,
        "coverage_pct": round(100 * with_pathway / max(total, 1), 2),
        "total_pathway_rows": total_rows,
        "aracyc_pathway_count": len(context.aracyc_pathway_info),
        "aracyc_mapping_status": dict(context.aracyc_mapping_status),
        "pathway_status": dict(context.pathway_status),
        "expanded_candidates": context.expanded_candidates_count,
        "ml_predictions": context.ml_predictions_count,
        "preprocess_counts": dict(context.preprocess_counts),
        "step_notes": context.step_notes,
    }


def run(context: PipelineContext) -> PipelineContext:
    """Write the final AraCyc-first ranked pathway table and summary."""

    print("  Step 7a: Writing AraCyc pathway output...", flush=True)

    row_count = 0
    with context.paths.aracyc_pathway_output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=ARACYC_OUTPUT_FIELDS, delimiter="\t")
        writer.writeheader()

        for compound_id in sorted(context.compounds, key=int):
            for row in context.aracyc_ranked_rows.get(compound_id, []):
                writer.writerow({
                    "compound_id": row.compound_id,
                    "chebi_accession": row.chebi_accession,
                    "chebi_name": row.chebi_name,
                    "pathway_rank": row.pathway_rank,
                    "score": f"{row.score:.4f}",
                    "confidence_level": row.confidence_level,
                    "evidence_type": row.evidence_type,
                    "match_method": row.match_method,
                    "match_score": f"{row.match_score:.4f}",
                    "aracyc_compound_id": row.aracyc_compound_id,
                    "aracyc_compound_name": row.aracyc_compound_name,
                    "source_db": row.source_db,
                    "pathway_id": row.pathway_id,
                    "pathway_name": row.pathway_name,
                    "pathway_category": row.pathway_category,
                    "gene_count": row.gene_count,
                    "gene_ids": row.gene_ids,
                    "ec_numbers": row.ec_numbers,
                    "reaction_count": row.reaction_count,
                    "compound_count": row.compound_count,
                    "go_best_term": row.go_best_term,
                    "go_best_fdr": f"{row.go_best_fdr:.6g}" if row.go_best_fdr and row.go_best_fdr < 1.0 else "",
                    "plant_reactome_best_category": row.plant_reactome_best_category,
                    "plant_reactome_alignment_confidence": row.plant_reactome_alignment_confidence,
                    "annotation_confidence": row.annotation_confidence,
                    "chebi_xref_direct": str(row.chebi_xref_direct).lower(),
                    "structure_validated": str(row.structure_validated).lower(),
                    "cofactor_like": str(row.cofactor_like).lower(),
                    "top_positive_features": row.top_positive_features,
                    "top_negative_features": row.top_negative_features,
                    "feature_contributions_json": row.feature_contributions_json,
                    "reason": row.reason,
                })
                row_count += 1

    print(f"    Output: {row_count} rows → {context.paths.aracyc_pathway_output_path}", flush=True)

    # Build and write summary
    summary = build_summary(
        compounds=context.compounds,
        mapping_status=context.aracyc_mapping_status,
        pathway_status=context.pathway_status,
        alias_rows=context.alias_rows,
        comments_profile=context.comments_profile,
        plantcyc_records=context.plantcyc_records,
        lipidmaps_records=context.lipidmaps_records,
        pubchem_stats=context.pubchem_stats,
    )
    summary["compounds_total"] = context.aracyc_reference_total()
    summary["chebi_input_total"] = len(context.compounds)
    summary["coverage_basis"] = "aracyc_reference_compounds"
    if context.step_notes:
        summary.setdefault("notes", []).extend(context.step_notes)

    with context.paths.summary_output_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    # Write preprocess metadata
    context.preprocess_version = PREPROCESS_VERSION
    metadata = build_preprocess_metadata(context)
    with context.paths.preprocess_metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)

    print(f"    Metadata → {context.paths.preprocess_metadata_path}", flush=True)

    context.preprocess_counts["step7a_output_rows"] = row_count
    return context
