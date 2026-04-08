"""Step 6: score annotated pathways with explicit rule contributions.

This module converts annotated pathway hits into ranked recommendation rows.
The scoring system is intentionally rule-based and fully decomposed into named
feature contributions so the output can later be explained directly or replaced
by a model with SHAP-style attribution.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from process_chebi_to_pathways_v2 import (
    GENERIC_KEGG_MAPS,
    build_pathway_explanation,
    pathway_support_bonus,
    top_feature_labels,
)

from pathway_pipeline.cli_utils import build_context, build_parser, format_counter, print_summary
from pathway_pipeline.context import PipelineContext, RankedPathwayRow
from pathway_pipeline.step1_alias_standardization import run as run_step1
from pathway_pipeline.step2_map_names_to_kegg import run as run_step2
from pathway_pipeline.step3_link_compounds_to_pathways import run as run_step3
from pathway_pipeline.step4_map_to_ath import run as run_step4
from pathway_pipeline.step5_annotate_pathways import run as run_step5


def run(context: PipelineContext) -> PipelineContext:
    """Score pathway hits and organize ranked rows per compound."""

    # Track pathway coverage statistics for the final summary JSON.
    pathway_status = Counter()
    ranked_rows_by_compound = {}

    # Precompute normalization denominators used in the specificity / gene
    # support features so every pathway is scored on the same scale.
    max_map_count = max(context.map_pathway_compound_counts.values(), default=1)
    max_gene_count = max(context.ath_gene_counts.values(), default=1)

    # Score each input compound independently. This keeps ranking local to one
    # compound and avoids mixing evidence across compounds.
    for compound_id in sorted(context.compounds, key=int):
        selected = context.selected_by_compound.get(compound_id, [])
        if not selected:
            pathway_status["without_mapping"] += 1
            ranked_rows_by_compound[compound_id] = []
            continue

        compound = context.compounds[compound_id]
        support_context = context.support_contexts.get(compound_id)

        # Multiple KEGG compounds can converge on the same pathway target. We
        # aggregate them first, then keep the strongest-scoring explanation for
        # that pathway target while still remembering all supporting KEGG IDs.
        aggregated = {}

        for hit in context.annotated_pathway_hits.get(compound_id, []):
            # Smaller pathways are treated as more specific, so their score gets
            # a mild boost compared with huge global maps.
            specificity = 1 - math.log1p(hit.map_pathway_compound_count) / math.log1p(max_map_count)

            # ath gene count is normalized separately. It only applies when an
            # ath pathway exists.
            gene_ratio = math.log1p(hit.ath_gene_count) / math.log1p(max_gene_count) if hit.ath_gene_count else 0.0

            # PlantCyc/AraCyc support is turned into a compact bonus plus human-
            # readable evidence strings.
            plant_bonus, plant_source, plant_examples, _plant_gene_count = pathway_support_bonus(
                pathway_name=hit.pathway_name,
                support_context=support_context,
                plantcyc_pathway_stats=context.plantcyc_pathway_stats,
            )

            # Very broad KEGG overview maps are useful as context but should not
            # outrank more specific pathways, so they receive a penalty.
            generic_penalty = -0.06 if hit.map_pathway_id in GENERIC_KEGG_MAPS or hit.pathway_category == "Global and overview maps" else 0.0

            # Make every scoring factor explicit. These values are later written
            # as JSON so the user can inspect why a pathway scored well or badly.
            contributions = {
                "mapping_confidence": round(0.45 * hit.mapping.final_score, 3),
                "pathway_specificity": round(0.15 * specificity, 3),
                "ath_exists": 0.12 if hit.ath_pathway_id else 0.0,
                "ath_gene_support": round(0.10 * gene_ratio, 3),
                "direct_kegg_xref": 0.05 if hit.mapping.direct_kegg_xref else 0.0,
                "structure_support": 0.05 if hit.mapping.has_structure_evidence else 0.0,
                "direct_link_bonus": 0.12 if hit.direct_link else 0.0,
                "substrate_role_bonus": 0.12 if hit.has_substrate_role else 0.0,
                "product_role_bonus": 0.06 if hit.has_product_role else 0.0,
                "both_role_bonus": 0.03 if hit.has_both_role else 0.0,
                "cofactor_penalty": -0.15 if hit.cofactor_like else 0.0,
                "reaction_support_bonus": round(min(0.12, 0.05 * math.log1p(hit.support_reaction_count)), 3),
                "plantcyc_support": round(plant_bonus, 3),
                "generic_pathway_penalty": generic_penalty,
                "map_fallback_penalty": -0.10 if not hit.ath_pathway_id else 0.0,
                "annotation_quality_bonus": 0.03 if hit.annotation_confidence == "high" else 0.01 if hit.annotation_confidence == "medium" else 0.0,
                "plant_reactome_alignment_bonus": 0.02 if hit.plant_reactome_alignment_confidence == "high" else 0.01 if hit.plant_reactome_alignment_confidence == "medium" else 0.0,
            }

            # Clamp the final score to a bounded interval so downstream
            # confidence thresholds stay simple and stable.
            score = max(0.0, min(sum(contributions.values()), 0.999))

            # Confidence is intentionally conservative: "high" requires both a
            # strong score and an ath pathway with gene support.
            confidence_level = "high" if score >= 0.70 and hit.ath_pathway_id and hit.ath_gene_count > 0 else "medium" if score >= 0.40 else "low"

            # Serialize a small Reactome context string instead of carrying the
            # tuple structure into the final table.
            reactome_text = ";".join(f"{pathway_id}|{species}" for pathway_id, species in hit.reactome_matches[:3])

            # Also keep short positive/negative feature summaries for quick
            # reading in spreadsheets or terminal output.
            top_positive, top_negative = top_feature_labels(contributions)

            # Build the human-readable explanation string used in the result TSV.
            explanation = build_pathway_explanation(
                mapping=hit.mapping,
                pathway_name=hit.pathway_name,
                target_type=hit.pathway_target_type,
                gene_count=hit.ath_gene_count,
                plant_source=plant_source,
                plant_examples=plant_examples,
                score=score,
                confidence_level=confidence_level,
                contributions=contributions,
            )
            role_notes = []
            if hit.direct_link:
                role_notes.append("direct KEGG compound-pathway link")
            if hit.has_substrate_role or hit.has_product_role or hit.has_both_role:
                role_notes.append(f"reaction roles: {hit.role_summary}")
            if hit.cofactor_like:
                role_notes.append("cofactor-like participation lowered the score")
            if hit.relation_vstamp:
                role_notes.append(f"relation_vstamp={hit.relation_vstamp}")
            if role_notes:
                explanation = f"{explanation}; " + "; ".join(role_notes)
            annotation_notes = []
            if hit.brite_l1 or hit.brite_l2 or hit.brite_l3:
                annotation_notes.append(
                    "brite=" + " / ".join(value for value in (hit.brite_l1, hit.brite_l2, hit.brite_l3) if value)
                )
            if hit.go_best_term:
                annotation_notes.append(f"go_best_term={hit.go_best_term}")
            if hit.plant_context_tags:
                annotation_notes.append(f"plant_tags={','.join(hit.plant_context_tags)}")
            if hit.plant_reactome_best_category:
                note = f"plant_reactome={hit.plant_reactome_best_category}"
                if hit.plant_reactome_alignment_confidence:
                    note += f" ({hit.plant_reactome_alignment_confidence})"
                annotation_notes.append(note)
            if annotation_notes:
                explanation = f"{explanation}; " + "; ".join(annotation_notes)

            # Aggregate rows by their final target pathway so different KEGG
            # supporting compounds can contribute to the same pathway entry.
            entry = aggregated.setdefault(
                hit.pathway_target_id,
                {
                    "score": -1.0,
                    "confidence_level": confidence_level,
                    "mapping_confidence": 0.0,
                    "support_kegg_ids": set(),
                    "support_kegg_names": set(),
                    "pathway_target_id": hit.pathway_target_id,
                    "pathway_target_type": hit.pathway_target_type,
                    "ath_pathway_id": hit.ath_pathway_id,
                    "map_pathway_id": hit.map_pathway_id,
                    "relation_vstamp": hit.relation_vstamp,
                    "pathway_name": hit.pathway_name,
                    "map_id": hit.map_id,
                    "kegg_name": hit.kegg_name,
                    "brite_l1": hit.brite_l1,
                    "brite_l2": hit.brite_l2,
                    "brite_l3": hit.brite_l3,
                    "pathway_group": hit.pathway_group,
                    "pathway_category": hit.pathway_category,
                    "map_pathway_compound_count": hit.map_pathway_compound_count,
                    "ath_gene_count": hit.ath_gene_count,
                    "go_best_term": hit.go_best_term,
                    "go_best_fdr": hit.go_best_fdr,
                    "plant_context_tags": ";".join(hit.plant_context_tags),
                    "plant_reactome_best_category": hit.plant_reactome_best_category,
                    "plant_reactome_alignment_confidence": hit.plant_reactome_alignment_confidence,
                    "annotation_confidence": hit.annotation_confidence,
                    "support_reaction_count": hit.support_reaction_count,
                    "role_summary": hit.role_summary,
                    "plantcyc_support_source": plant_source,
                    "plantcyc_support_examples": plant_examples,
                    "reactome_matches": reactome_text,
                    "top_positive_features": top_positive,
                    "top_negative_features": top_negative,
                    "feature_contributions_json": json.dumps(contributions, ensure_ascii=False, sort_keys=True),
                    "reason": explanation,
                },
            )

            # If the current hit is stronger than the previously stored best hit
            # for the same pathway target, replace the score-bearing fields but
            # keep the cumulative support KEGG IDs/names.
            if score > float(entry["score"]):
                entry["score"] = score
                entry["confidence_level"] = confidence_level
                entry["mapping_confidence"] = hit.mapping.final_score
                entry["ath_gene_count"] = hit.ath_gene_count
                entry["relation_vstamp"] = hit.relation_vstamp
                entry["map_id"] = hit.map_id
                entry["kegg_name"] = hit.kegg_name
                entry["brite_l1"] = hit.brite_l1
                entry["brite_l2"] = hit.brite_l2
                entry["brite_l3"] = hit.brite_l3
                entry["go_best_term"] = hit.go_best_term
                entry["go_best_fdr"] = hit.go_best_fdr
                entry["plant_context_tags"] = ";".join(hit.plant_context_tags)
                entry["plant_reactome_best_category"] = hit.plant_reactome_best_category
                entry["plant_reactome_alignment_confidence"] = hit.plant_reactome_alignment_confidence
                entry["annotation_confidence"] = hit.annotation_confidence
                entry["support_reaction_count"] = hit.support_reaction_count
                entry["role_summary"] = hit.role_summary
                entry["plantcyc_support_source"] = plant_source
                entry["plantcyc_support_examples"] = plant_examples
                entry["reactome_matches"] = reactome_text
                entry["top_positive_features"] = top_positive
                entry["top_negative_features"] = top_negative
                entry["feature_contributions_json"] = json.dumps(contributions, ensure_ascii=False, sort_keys=True)
                entry["reason"] = explanation

            # Always accumulate all supporting KEGG identifiers, even when the
            # visible score/explanation comes from only the strongest hit.
            entry["support_kegg_ids"].add(hit.mapping.kegg_compound_id)
            entry["support_kegg_names"].add(hit.mapping.kegg_primary_name)

        if not aggregated:
            pathway_status["mapped_without_pathway"] += 1
            ranked_rows_by_compound[compound_id] = []
            continue

        # Rank pathway entries for the current compound. Score dominates, then
        # gene support, then pathway size as a final deterministic tie-breaker.
        ranked_entries = sorted(
            aggregated.values(),
            key=lambda item: (
                float(item["score"]),
                int(item["ath_gene_count"]),
                -int(item["map_pathway_compound_count"]),
                item["pathway_target_id"],
            ),
            reverse=True,
        )
        pathway_status["with_pathway"] += 1
        pathway_status["pathway_rows"] += len(ranked_entries)

        rows = []

        # Convert the aggregated dictionaries into the strongly typed row object
        # used by step 7 when writing the final TSV.
        for rank, entry in enumerate(ranked_entries, start=1):
            rows.append(
                RankedPathwayRow(
                    compound_id=compound_id,
                    chebi_accession=compound.chebi_accession,
                    chebi_name=compound.name,
                    pathway_rank=rank,
                    score=float(entry["score"]),
                    confidence_level=str(entry["confidence_level"]),
                    evidence_type="direct_compound_pathway",
                    mapping_confidence=float(entry["mapping_confidence"]),
                    support_kegg_compound_ids=tuple(sorted(entry["support_kegg_ids"])),
                    support_kegg_names=tuple(sorted(entry["support_kegg_names"])),
                    pathway_target_id=str(entry["pathway_target_id"]),
                    pathway_target_type=str(entry["pathway_target_type"]),
                    ath_pathway_id=str(entry["ath_pathway_id"]),
                    map_pathway_id=str(entry["map_pathway_id"]),
                    relation_vstamp=str(entry["relation_vstamp"]),
                    pathway_name=str(entry["pathway_name"]),
                    map_id=str(entry["map_id"]),
                    kegg_name=str(entry["kegg_name"]),
                    brite_l1=str(entry["brite_l1"]),
                    brite_l2=str(entry["brite_l2"]),
                    brite_l3=str(entry["brite_l3"]),
                    pathway_group=str(entry["pathway_group"]),
                    pathway_category=str(entry["pathway_category"]),
                    map_pathway_compound_count=int(entry["map_pathway_compound_count"]),
                    ath_gene_count=int(entry["ath_gene_count"]),
                    go_best_term=str(entry["go_best_term"]),
                    go_best_fdr=float(entry["go_best_fdr"]),
                    plant_context_tags=str(entry["plant_context_tags"]),
                    plant_reactome_best_category=str(entry["plant_reactome_best_category"]),
                    plant_reactome_alignment_confidence=str(entry["plant_reactome_alignment_confidence"]),
                    annotation_confidence=str(entry["annotation_confidence"]),
                    support_reaction_count=int(entry["support_reaction_count"]),
                    role_summary=str(entry["role_summary"]),
                    plantcyc_support_source=str(entry["plantcyc_support_source"]),
                    plantcyc_support_examples=str(entry["plantcyc_support_examples"]),
                    reactome_matches=str(entry["reactome_matches"]),
                    top_positive_features=str(entry["top_positive_features"]),
                    top_negative_features=str(entry["top_negative_features"]),
                    feature_contributions_json=str(entry["feature_contributions_json"]),
                    reason=str(entry["reason"]),
                )
            )
        ranked_rows_by_compound[compound_id] = rows

    # Persist per-compound ranked rows and global pathway coverage counters.
    context.pathway_status = pathway_status
    context.ranked_pathway_rows = ranked_rows_by_compound
    return context


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for standalone step-6 execution."""

    return build_parser(
        description="Run pathway pipeline steps 1-6: score annotated pathways but do not write final result files.",
        default_output_tag="step6_cli",
    ).parse_args()


def main() -> None:
    """Run steps 1-6 and report in-memory scoring coverage."""

    args = parse_args()
    context = build_context(workdir=args.workdir, output_tag=args.output_tag)
    run_step1(context)
    run_step2(context)
    run_step3(context)
    run_step4(context)
    run_step5(context)
    run(context)
    ranked_row_count = sum(len(rows) for rows in context.ranked_pathway_rows.values())
    print_summary(
        "Step 6 completed.",
        [
            f"Ranked pathway rows in memory: {ranked_row_count}",
            f"Pathway status counts: {format_counter(context.pathway_status)}",
            "No final TSV/JSON is written at step 6; use step 7 to materialize outputs.",
        ],
    )


if __name__ == "__main__":
    main()
