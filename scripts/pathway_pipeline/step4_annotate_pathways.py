"""Step 4 (AraCyc-first): annotate AraCyc pathway hits with gene, GO, and classification context.

AraCyc pathways already carry gene/EC/reaction information from the pathway file.
This step enriches hits with:
  - Gene counts and IDs from aracyc_pathways
  - Pathway category extraction from pathway names
  - GO enrichment (reusing existing infrastructure)
  - Plant Reactome alignment (reusing existing infrastructure)
  - Cofactor-like detection
"""

from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from process_chebi_to_pathways_v2 import normalize_name

from pathway_pipeline.context import (
    AnnotatedAraCycHit,
    AraCycPathwayHit,
    AraCycPathwayInfo,
    PipelineContext,
)
from pathway_pipeline.pathway_annotation_support import (
    build_go_term_gene_sets,
    compute_pathway_go_enrichment,
    load_gene_to_go_bp,
    match_plant_reactome_context,
    load_plant_reactome_pathways,
    load_plant_reactome_gene_index,
    token_sorensen_dice,
    ensure_plant_reactome_refs,
)


# ---------------------------------------------------------------------------
# Pathway category extraction
# ---------------------------------------------------------------------------

# Patterns for classifying AraCyc pathway names
_CATEGORY_PATTERNS = [
    (re.compile(r"\bbiosynthesis\b", re.I), "Biosynthesis"),
    (re.compile(r"\bdegradation\b", re.I), "Degradation"),
    (re.compile(r"\bdetoxification\b", re.I), "Detoxification"),
    (re.compile(r"\binactivation\b", re.I), "Inactivation"),
    (re.compile(r"\bmetabolism\b", re.I), "Metabolism"),
    (re.compile(r"\bsignaling\b|\bsignal\b", re.I), "Signaling"),
    (re.compile(r"\btransport\b", re.I), "Transport"),
    (re.compile(r"\bsalvage\b", re.I), "Salvage"),
    (re.compile(r"\binterconversion\b|\bconversion\b", re.I), "Interconversion"),
    (re.compile(r"\bmodification\b|\bmethylation\b|\bacylation\b|\bglycosylation\b", re.I), "Modification"),
    (re.compile(r"\bcycle\b", re.I), "Cycle"),
    (re.compile(r"\bphotosynthesis\b|\bphotorespiration\b", re.I), "Photosynthesis"),
    (re.compile(r"\bfermentation\b", re.I), "Fermentation"),
    (re.compile(r"\bfixation\b", re.I), "Fixation"),
    (re.compile(r"\bassimilation\b", re.I), "Assimilation"),
]


def classify_pathway(pathway_name: str) -> str:
    """Extract a coarse category from an AraCyc pathway name."""
    for pattern, category in _CATEGORY_PATTERNS:
        if pattern.search(pathway_name):
            return category
    return "Other"


# ---------------------------------------------------------------------------
# Cofactor detection
# ---------------------------------------------------------------------------

COFACTOR_ARACYC_IDS = {
    "WATER", "OXYGEN-MOLECULE", "CARBON-DIOXIDE", "PROTON",
    "ATP", "ADP", "AMP", "NAD", "NADH", "NAD-P-OR-NOP",
    "NADP", "NADPH", "CO-A", "Pi", "PPI", "FAD", "FADH2",
}

COFACTOR_NAME_TOKENS = {
    "atp", "adp", "amp", "nadh", "nadph", "nadp", "nad", "water",
    "oxygen", "coenzyme a", "phosphate", "pyrophosphate", "carbon dioxide",
    "proton", "h+", "h2o", "o2", "co2",
}


def is_cofactor_like(aracyc_compound_id: str, common_name: str) -> bool:
    """Flag ubiquitous helper molecules."""
    if aracyc_compound_id in COFACTOR_ARACYC_IDS:
        return True
    normalized = common_name.lower()
    return any(token in normalized for token in COFACTOR_NAME_TOKENS)


# ---------------------------------------------------------------------------
# Annotation confidence for AraCyc pipeline
# ---------------------------------------------------------------------------


def build_aracyc_annotation_confidence(
    *,
    gene_count: int,
    go_best_term: str,
    plant_reactome_alignment_confidence: str,
    match_score: float,
) -> str:
    """Assign confidence to the annotation bundle."""
    strong_evidence = sum([
        gene_count > 0,
        bool(go_best_term),
        plant_reactome_alignment_confidence in ("high", "medium"),
        match_score >= 0.90,
    ])
    if strong_evidence >= 3:
        return "high"
    if strong_evidence >= 1:
        return "medium"
    return "low"


# ---------------------------------------------------------------------------
# Index writing
# ---------------------------------------------------------------------------


def write_pathway_annotation_index(context: PipelineContext) -> int:
    """Write AraCyc pathway annotation index."""
    row_count = 0
    with context.paths.aracyc_pathway_annotation_index_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "compound_id",
                "chebi_accession",
                "aracyc_compound_id",
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
                "plant_reactome_best_name",
                "plant_reactome_alignment_confidence",
                "annotation_confidence",
                "cofactor_like",
            ],
            delimiter="\t",
        )
        writer.writeheader()
        for compound_id in sorted(context.annotated_aracyc_hits, key=int):
            for hit in context.annotated_aracyc_hits[compound_id]:
                writer.writerow({
                    "compound_id": compound_id,
                    "chebi_accession": f"CHEBI:{compound_id}",
                    "aracyc_compound_id": hit.match.aracyc_compound_id,
                    "source_db": hit.source_db,
                    "pathway_id": hit.pathway_id,
                    "pathway_name": hit.pathway_name,
                    "pathway_category": hit.pathway_category,
                    "gene_count": hit.gene_count,
                    "gene_ids": ";".join(hit.gene_ids[:10]),
                    "ec_numbers": ";".join(hit.ec_numbers),
                    "reaction_count": hit.reaction_count,
                    "compound_count": hit.compound_count,
                    "go_best_term": hit.go_best_term,
                    "go_best_fdr": f"{hit.go_best_fdr:.6f}" if hit.go_best_fdr < 1.0 else "",
                    "plant_reactome_best_name": hit.plant_reactome_best_name,
                    "plant_reactome_alignment_confidence": hit.plant_reactome_alignment_confidence,
                    "annotation_confidence": hit.annotation_confidence,
                    "cofactor_like": str(hit.cofactor_like).lower(),
                })
                row_count += 1
    return row_count


# ---------------------------------------------------------------------------
# Step runner
# ---------------------------------------------------------------------------


def run(context: PipelineContext) -> PipelineContext:
    """Annotate AraCyc pathway hits with gene, GO, Reactome, and classification data."""

    print("  Step 4a: Annotating AraCyc pathway hits...", flush=True)

    # Load GO enrichment data (reuse existing infrastructure)
    go_gene_index = context.go_gene_index
    go_term_namespace = context.go_term_namespace
    if not go_gene_index:
        try:
            go_gene_index, go_term_namespace, go_vstamp = load_gene_to_go_bp(
                context.paths.gene_association_tair_path,
                context.paths.go_basic_obo_path,
            )
            context.go_gene_index = go_gene_index
            context.go_term_namespace = go_term_namespace
            context.go_vstamp = go_vstamp
        except Exception as e:
            print(f"    Warning: GO data unavailable: {e}", flush=True)
            go_gene_index = {}

    # Load Plant Reactome data (reuse existing infrastructure)
    plant_reactome_pathways = context.plant_reactome_pathways
    plant_reactome_gene_sets = context.plant_reactome_gene_sets
    if not plant_reactome_pathways:
        try:
            ensure_plant_reactome_refs(
                context.paths.plant_reactome_pathways_path,
                context.paths.plant_reactome_gene_pathway_path,
                context.paths.plant_reactome_version_path,
            )
            plant_reactome_pathways = load_plant_reactome_pathways(
                context.paths.plant_reactome_pathways_path
            )
            plant_reactome_gene_sets = load_plant_reactome_gene_index(
                context.paths.plant_reactome_gene_pathway_path
            )
            context.plant_reactome_pathways = plant_reactome_pathways
            context.plant_reactome_gene_sets = plant_reactome_gene_sets
        except Exception as e:
            print(f"    Warning: Plant Reactome data unavailable: {e}", flush=True)

    # Compute GO enrichment for AraCyc pathways using their gene sets
    pathway_go_enrichment = context.pathway_go_enrichment
    if not pathway_go_enrichment and go_gene_index:
        # Build GO term→gene and background gene sets
        term_to_genes, go_names = build_go_term_gene_sets(go_gene_index)
        background_genes: set[str] = set()
        for gene_set in term_to_genes.values():
            background_genes.update(gene_set)

        # Compute enrichment per AraCyc pathway
        for pid, info in context.aracyc_pathway_info.items():
            if info.gene_ids and len(info.gene_ids) >= 3:
                pathway_go_enrichment[pid] = compute_pathway_go_enrichment(
                    info.gene_ids,
                    term_to_genes,
                    go_names,
                    background_genes,
                )
        context.pathway_go_enrichment = pathway_go_enrichment
        print(f"    GO enrichment computed for {len(pathway_go_enrichment)} pathways", flush=True)

    # Annotate each compound's pathway hits
    print("  Step 4a: Building annotated hits...", flush=True)
    total_annotated = 0

    for compound_id in sorted(context.aracyc_pathway_hits, key=int):
        hits = context.aracyc_pathway_hits[compound_id]
        annotated: list[AnnotatedAraCycHit] = []

        for hit in hits:
            # Get pathway metadata
            pinfo = context.aracyc_pathway_info.get(hit.pathway_id)

            gene_ids = tuple(sorted(pinfo.gene_ids)) if pinfo else ()
            gene_names = tuple(sorted(pinfo.gene_names)) if pinfo else ()
            gene_count = len(gene_ids)
            reaction_count = len(pinfo.reaction_ids) if pinfo else len(hit.reaction_ids)
            compound_count = context.aracyc_pathway_compound_counts.get(
                hit.pathway_id or hit.pathway_name, 0
            )

            # Pathway category
            pathway_category = classify_pathway(hit.pathway_name)

            # GO enrichment for this pathway
            go_best_term = ""
            go_best_fdr = 1.0
            go_data = pathway_go_enrichment.get(hit.pathway_id, {})
            if go_data:
                go_best_term = go_data.get("go_best_term", "")
                raw_fdr = go_data.get("go_best_fdr", "")
                go_best_fdr = float(raw_fdr) if raw_fdr != "" else 1.0

            # Plant Reactome alignment
            pr_best_id = ""
            pr_best_name = ""
            pr_alignment_score = 0.0
            pr_alignment_confidence = ""
            if plant_reactome_pathways:
                best_dice = 0.0
                for pr_id, pr_info in plant_reactome_pathways.items():
                    pr_name = pr_info.get("name", "")
                    dice = token_sorensen_dice(hit.pathway_name, pr_name)
                    if dice > best_dice:
                        best_dice = dice
                        pr_best_id = pr_id
                        pr_best_name = pr_name
                if best_dice >= 0.6:
                    pr_alignment_score = best_dice
                    pr_alignment_confidence = "high" if best_dice >= 0.8 else "medium"
                else:
                    pr_best_id = ""
                    pr_best_name = ""

            # Cofactor detection
            cofactor_like = is_cofactor_like(
                hit.match.aracyc_compound_id, hit.match.aracyc_common_name
            )

            # Annotation confidence
            annotation_confidence = build_aracyc_annotation_confidence(
                gene_count=gene_count,
                go_best_term=go_best_term,
                plant_reactome_alignment_confidence=pr_alignment_confidence,
                match_score=hit.match.match_score,
            )

            annotated.append(AnnotatedAraCycHit(
                compound_id=compound_id,
                match=hit.match,
                pathway_id=hit.pathway_id,
                pathway_name=hit.pathway_name,
                pathway_category=pathway_category,
                ec_numbers=hit.ec_numbers,
                reaction_ids=hit.reaction_ids,
                reaction_count=reaction_count,
                compound_count=compound_count,
                gene_ids=gene_ids,
                gene_names=gene_names,
                gene_count=gene_count,
                go_best_term=go_best_term,
                go_best_fdr=go_best_fdr,
                plant_reactome_best_id=pr_best_id,
                plant_reactome_best_name=pr_best_name,
                plant_reactome_alignment_score=pr_alignment_score,
                plant_reactome_alignment_confidence=pr_alignment_confidence,
                annotation_confidence=annotation_confidence,
                source_db=hit.source_db,
                cofactor_like=cofactor_like,
            ))

        if annotated:
            context.annotated_aracyc_hits[compound_id] = annotated
            total_annotated += len(annotated)

    print(f"    Annotated hits: {total_annotated}", flush=True)

    # Write annotation index
    n_idx = write_pathway_annotation_index(context)
    print(f"    aracyc_pathway_annotation_index: {n_idx} rows", flush=True)

    context.preprocess_counts["step4a_annotated_hits"] = total_annotated
    return context
