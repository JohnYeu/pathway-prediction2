"""Step 5 (AraCyc-first): score annotated AraCyc pathway hits.

Scoring uses AraCyc-native features. Every contribution is explicit and
decomposed for downstream inspection or model replacement.
"""

from __future__ import annotations

import json
import math
import sys
from collections import Counter
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from pathway_pipeline.context import (
    AnnotatedAraCycHit,
    AraCycRankedRow,
    PipelineContext,
)


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------


def top_feature_labels(contributions: dict[str, float]) -> tuple[str, str]:
    """Return short comma-separated labels for the top positive and negative features."""
    pos = sorted(((v, k) for k, v in contributions.items() if v > 0), reverse=True)
    neg = sorted(((v, k) for k, v in contributions.items() if v < 0))
    pos_str = ", ".join(f"{k}({v:+.3f})" for v, k in pos[:3])
    neg_str = ", ".join(f"{k}({v:+.3f})" for v, k in neg[:3])
    return pos_str, neg_str


def score_hit(
    hit: AnnotatedAraCycHit,
    max_compound_count: int,
    max_gene_count: int,
    plantcyc_corroborated_pathway_ids: set[str],
) -> tuple[float, str, dict[str, float]]:
    """Score one annotated AraCyc pathway hit.

    Returns (score, confidence_level, contributions).
    """
    # Pathway specificity: smaller pathways are more specific
    specificity = 1 - math.log1p(hit.compound_count) / math.log1p(max_compound_count) if max_compound_count > 0 else 0.5

    # Gene support ratio
    gene_ratio = math.log1p(hit.gene_count) / math.log1p(max_gene_count) if max_gene_count > 0 else 0.0

    # Multi-reaction bonus (capped)
    reaction_bonus = min(0.05, 0.02 * math.log1p(hit.reaction_count))

    # PlantCyc corroboration: pathway also exists in PlantCyc
    plantcyc_corroboration = (
        hit.source_db == "AraCyc" and hit.pathway_id in plantcyc_corroborated_pathway_ids
    )

    contributions = {
        "match_confidence": round(0.30 * hit.match.match_score, 4),
        "chebi_xref_direct": 0.12 if hit.match.chebi_xref_direct else 0.0,
        "structure_validated": 0.08 if hit.match.structure_validated else 0.0,
        "gene_support": round(0.12 * gene_ratio, 4),
        "pathway_specificity": round(0.10 * specificity, 4),
        "ec_support": 0.06 if hit.ec_numbers else 0.0,
        "reaction_present": 0.04 if hit.reaction_count > 0 else 0.0,
        "multi_reaction_bonus": round(reaction_bonus, 4),
        "cofactor_penalty": -0.12 if hit.cofactor_like else 0.0,
        "plantcyc_corroboration": 0.05 if plantcyc_corroboration else 0.0,
        "plant_reactome_alignment": (
            0.03 if hit.plant_reactome_alignment_confidence == "high"
            else 0.015 if hit.plant_reactome_alignment_confidence == "medium"
            else 0.0
        ),
        "go_enrichment_bonus": 0.02 if hit.go_best_term else 0.0,
        "annotation_quality": (
            0.03 if hit.annotation_confidence == "high"
            else 0.015 if hit.annotation_confidence == "medium"
            else 0.0
        ),
    }

    score = max(0.0, min(sum(contributions.values()), 0.999))

    # Confidence level
    if score >= 0.65 and hit.gene_count > 0 and hit.match.match_score >= 0.90:
        confidence_level = "high"
    elif score >= 0.35:
        confidence_level = "medium"
    else:
        confidence_level = "low"

    return score, confidence_level, contributions


def build_reason(hit: AnnotatedAraCycHit, score: float, confidence_level: str) -> str:
    """Build a human-readable explanation string."""
    parts = [
        f"AraCyc match: {hit.match.aracyc_compound_id} ({hit.match.match_method}, "
        f"score={hit.match.match_score:.2f})",
        f"pathway={hit.pathway_name}",
        f"category={hit.pathway_category}",
    ]
    if hit.gene_count:
        parts.append(f"genes={hit.gene_count}")
    if hit.ec_numbers:
        parts.append(f"EC={';'.join(hit.ec_numbers[:3])}")
    if hit.cofactor_like:
        parts.append("cofactor_like")
    parts.append(f"final_score={score:.3f} ({confidence_level})")
    return "; ".join(parts)


# ---------------------------------------------------------------------------
# Step runner
# ---------------------------------------------------------------------------


def run(context: PipelineContext) -> PipelineContext:
    """Score AraCyc pathway hits and build ranked rows per compound."""

    print("  Step 5a: Scoring AraCyc pathway hits...", flush=True)

    pathway_status = Counter()

    # Compute normalization denominators
    max_compound_count = max(context.aracyc_pathway_compound_counts.values(), default=1)
    max_gene_count = max(
        (len(info.gene_ids) for info in context.aracyc_pathway_info.values()),
        default=1,
    )

    # Collect pathways that are corroborated by both AraCyc and PlantCyc.
    plantcyc_corroborated_pathway_ids = {
        pid
        for pid, info in context.aracyc_pathway_info.items()
        if "AraCyc" in info.source_dbs and "PlantCyc" in info.source_dbs
    }

    for compound_id in sorted(context.compounds, key=int):
        hits = context.annotated_aracyc_hits.get(compound_id, [])
        if not hits:
            pathway_status["without_pathway"] += 1
            context.aracyc_ranked_rows[compound_id] = []
            continue

        compound = context.compounds[compound_id]

        # Score and aggregate by pathway_id to handle multiple hits to same pathway
        aggregated: dict[str, dict] = {}
        for hit in hits:
            score, confidence_level, contributions = score_hit(
                hit, max_compound_count, max_gene_count, plantcyc_corroborated_pathway_ids
            )
            top_pos, top_neg = top_feature_labels(contributions)
            reason = build_reason(hit, score, confidence_level)

            key = hit.pathway_id or hit.pathway_name
            existing = aggregated.get(key)
            if existing is None or score > existing["score"]:
                aggregated[key] = {
                    "hit": hit,
                    "score": score,
                    "confidence_level": confidence_level,
                    "contributions": contributions,
                    "top_pos": top_pos,
                    "top_neg": top_neg,
                    "reason": reason,
                }

        # Sort by score descending and build ranked rows
        sorted_entries = sorted(aggregated.values(), key=lambda e: e["score"], reverse=True)
        rows: list[AraCycRankedRow] = []
        for rank, entry in enumerate(sorted_entries, 1):
            hit = entry["hit"]
            rows.append(AraCycRankedRow(
                compound_id=compound_id,
                chebi_accession=f"CHEBI:{compound_id}",
                chebi_name=compound.name,
                pathway_rank=rank,
                score=round(entry["score"], 4),
                confidence_level=entry["confidence_level"],
                evidence_type="primary_aracyc" if hit.source_db == "AraCyc" else "plantcyc_fallback",
                match_method=hit.match.match_method,
                match_score=hit.match.match_score,
                aracyc_compound_id=hit.match.aracyc_compound_id,
                aracyc_compound_name=hit.match.aracyc_common_name,
                source_db=hit.source_db,
                pathway_id=hit.pathway_id,
                pathway_name=hit.pathway_name,
                pathway_category=hit.pathway_category,
                gene_count=hit.gene_count,
                gene_ids=";".join(hit.gene_ids[:10]),
                ec_numbers=";".join(hit.ec_numbers),
                reaction_count=hit.reaction_count,
                compound_count=hit.compound_count,
                go_best_term=hit.go_best_term,
                go_best_fdr=hit.go_best_fdr,
                plant_reactome_best_category=hit.plant_reactome_best_name,
                plant_reactome_alignment_confidence=hit.plant_reactome_alignment_confidence,
                annotation_confidence=hit.annotation_confidence,
                chebi_xref_direct=hit.match.chebi_xref_direct,
                structure_validated=hit.match.structure_validated,
                cofactor_like=hit.cofactor_like,
                top_positive_features=entry["top_pos"],
                top_negative_features=entry["top_neg"],
                feature_contributions_json=json.dumps(entry["contributions"]),
                reason=entry["reason"],
            ))

        context.aracyc_ranked_rows[compound_id] = rows
        if rows:
            pathway_status["with_pathway"] += 1
        else:
            pathway_status["without_pathway"] += 1

    total = context.aracyc_reference_total()
    with_pw = len({
        hit.match.reference_compound_key
        for hits in context.annotated_aracyc_hits.values()
        for hit in hits
        if hit.source_db == "AraCyc"
    })
    context.pathway_status = Counter({
        "with_pathway": with_pw,
        "without_pathway": max(total - with_pw, 0),
    })
    print(f"    With pathways: {with_pw}/{total} ({100*with_pw/max(total, 1):.2f}%)", flush=True)
    print(f"    Pathway status: {dict(context.pathway_status)}", flush=True)

    context.preprocess_counts["step5a_with_pathway"] = with_pw
    return context
