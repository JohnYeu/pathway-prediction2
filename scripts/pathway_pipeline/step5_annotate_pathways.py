"""Step 5: annotate pathways with KEGG BRITE, GO BP, PMN, and Plant Reactome.

This step upgrades the pathway annotation layer from:

- pathway name
- coarse KEGG category
- ath gene count
- exact-name Reactome context

to a richer, reproducible annotation bundle that also includes:

- full KEGG BRITE hierarchy (L1/L2/L3)
- pathway-level Arabidopsis GO BP enrichment
- PMN/AraCyc-derived plant evidence and lightweight plant tags
- Plant Reactome alignment, category, and description context
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from process_chebi_to_pathways_v2 import (
    build_annotation_confidence,
    build_go_term_gene_sets,
    compute_pathway_go_enrichment,
    ensure_go_annotations,
    ensure_plant_reactome_refs,
    infer_plant_context_tags,
    iso_mtime,
    load_ath_gene_counts,
    load_gene_to_go_bp,
    load_map_pathways,
    load_pathway_categories,
    load_plant_reactome_gene_index,
    load_plant_reactome_pathways,
    load_plantcyc_pathway_stats,
    load_reactome_pathways,
    match_plant_reactome_context,
    normalize_name,
)

from pathway_pipeline.cli_utils import build_context, build_parser, print_summary
from pathway_pipeline.context import PipelineContext, Step5AnnotatedPathwayHit
from pathway_pipeline.step1_alias_standardization import run as run_step1
from pathway_pipeline.step2_map_names_to_kegg import run as run_step2
from pathway_pipeline.step3_link_compounds_to_pathways import run as run_step3
from pathway_pipeline.step4_map_to_ath import run as run_step4


def load_ath_gene_sets(path: Path) -> dict[str, set[str]]:
    """Load ath pathway -> AGI gene sets from the KEGG ath gene table."""

    gene_sets: dict[str, set[str]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            gene_ref, pathway_ref = line.rstrip("\n").split("\t", 1)
            pathway_id = pathway_ref.replace("path:", "")
            gene_id = gene_ref.replace("ath:", "").upper()
            gene_sets.setdefault(pathway_id, set()).add(gene_id)
    return gene_sets


def write_pathway_annotation_index(rows: list[dict[str, object]], context: PipelineContext) -> int:
    """Write the final Step 5 pathway annotation index."""

    fieldnames = [
        "pathway_id",
        "pathway_target_type",
        "map_pathway_id",
        "ath_pathway_id",
        "pathway_name",
        "map_id",
        "kegg_name",
        "brite_l1",
        "brite_l2",
        "brite_l3",
        "pathway_group",
        "pathway_category",
        "kegg_vstamp",
        "map_pathway_compound_count",
        "ath_gene_count",
        "go_top_terms_json",
        "go_best_term",
        "go_best_fdr",
        "go_vstamp",
        "plant_context_tags",
        "plant_evidence_sources",
        "aracyc_evidence_score",
        "reactome_matches",
        "plant_reactome_matches_json",
        "plant_reactome_best_id",
        "plant_reactome_best_name",
        "plant_reactome_best_category",
        "plant_reactome_best_description",
        "plant_reactome_alignment_score",
        "plant_reactome_alignment_confidence",
        "plant_reactome_tags",
        "plant_reactome_vstamp",
        "annotation_confidence",
    ]
    with context.paths.pathway_annotation_index_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def write_pathway_go_enrichment(context: PipelineContext) -> int:
    """Persist pathway-level GO enrichment results."""

    fieldnames = [
        "pathway_id",
        "map_id",
        "ath_id",
        "gene_count",
        "background_gene_count",
        "go_top_terms_json",
        "go_best_term",
        "go_best_fdr",
        "go_vstamp",
    ]
    rows = []
    for ath_id, payload in sorted(context.pathway_go_enrichment.items()):
        rows.append(
            {
                "pathway_id": ath_id,
                "map_id": f"map{ath_id[3:]}" if ath_id.startswith("ath") else "",
                "ath_id": ath_id,
                "gene_count": payload["gene_count"],
                "background_gene_count": payload["background_gene_count"],
                "go_top_terms_json": json.dumps(payload["terms"], ensure_ascii=False, sort_keys=True),
                "go_best_term": payload["go_best_term"],
                "go_best_fdr": f"{payload['go_best_fdr']:.6g}" if payload["go_best_fdr"] != "" else "",
                "go_vstamp": context.go_vstamp,
            }
        )
    with context.paths.pathway_go_enrichment_index_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def write_gene_to_go_index(context: PipelineContext) -> int:
    """Persist the preprocessed AGI -> GO BP index used in Step 5."""

    rows = []
    with context.paths.gene_to_go_index_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["gene_id", "go_id", "go_name", "namespace", "go_vstamp"],
            delimiter="\t",
        )
        writer.writeheader()
        for gene_id in sorted(context.go_gene_index):
            for go_id, go_name in sorted(context.go_gene_index[gene_id]):
                rows.append(
                    {
                        "gene_id": gene_id,
                        "go_id": go_id,
                        "go_name": go_name,
                        "namespace": context.go_term_namespace.get(go_id, ""),
                        "go_vstamp": context.go_vstamp,
                    }
                )
        writer.writerows(rows)
    return len(rows)


def write_plant_reactome_indexes(context: PipelineContext) -> tuple[int, int]:
    """Persist normalized Plant Reactome pathway and gene-set tables."""

    pathway_rows = []
    with context.paths.plant_reactome_pathway_index_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "pathway_id",
                "pathway_name",
                "species",
                "top_level_category",
                "description",
                "release_date",
                "go_biological_process_id",
                "go_biological_process_name",
                "name_normalized",
                "name_compact",
                "plant_reactome_vstamp",
            ],
            delimiter="\t",
        )
        writer.writeheader()
        for pathway_id, row in sorted(context.plant_reactome_pathways.items()):
            payload = dict(row)
            payload["plant_reactome_vstamp"] = context.plant_reactome_vstamp
            pathway_rows.append(payload)
        writer.writerows(pathway_rows)

    gene_rows = []
    with context.paths.plant_reactome_gene_index_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["pathway_id", "gene_id", "plant_reactome_vstamp"],
            delimiter="\t",
        )
        writer.writeheader()
        for pathway_id, genes in sorted(context.plant_reactome_gene_sets.items()):
            for gene_id in sorted(genes):
                gene_rows.append(
                    {
                        "pathway_id": pathway_id,
                        "gene_id": gene_id,
                        "plant_reactome_vstamp": context.plant_reactome_vstamp,
                    }
                )
        writer.writerows(gene_rows)
    return len(pathway_rows), len(gene_rows)


def write_plant_reactome_alignment(context: PipelineContext) -> int:
    """Persist KEGG pathway -> Plant Reactome alignment candidates."""

    rows = []
    with context.paths.plant_reactome_alignment_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "pathway_id",
                "map_id",
                "ath_id",
                "plant_reactome_id",
                "plant_reactome_name",
                "species",
                "top_level_category",
                "description",
                "alignment_score",
                "alignment_confidence",
                "gene_overlap_count",
                "gene_jaccard",
                "overlap_ratio_kegg",
                "name_similarity",
                "plant_reactome_vstamp",
            ],
            delimiter="\t",
        )
        writer.writeheader()
        for pathway_id, matches in sorted(context.plant_reactome_alignments.items()):
            ath_id = pathway_id if pathway_id.startswith("ath") else ""
            map_id = f"map{ath_id[3:]}" if ath_id else pathway_id
            for match in matches:
                rows.append(
                    {
                        "pathway_id": pathway_id,
                        "map_id": map_id,
                        "ath_id": ath_id,
                        "plant_reactome_id": match["plant_reactome_id"],
                        "plant_reactome_name": match["pathway_name"],
                        "species": match["species"],
                        "top_level_category": match["top_level_category"],
                        "description": match["description"],
                        "alignment_score": f"{match['alignment_score']:.4f}",
                        "alignment_confidence": match["alignment_confidence"],
                        "gene_overlap_count": match["gene_overlap_count"],
                        "gene_jaccard": f"{match['gene_jaccard']:.4f}",
                        "overlap_ratio_kegg": f"{match['overlap_ratio_kegg']:.4f}",
                        "name_similarity": f"{match['name_similarity']:.4f}",
                        "plant_reactome_vstamp": context.plant_reactome_vstamp,
                    }
                )
        writer.writerows(rows)
    return len(rows)


def write_plant_evidence_index(context: PipelineContext) -> int:
    """Write PMN support used for KEGG boosting and direct PMN fallback."""

    row_count = 0
    with context.paths.plant_evidence_index_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "compound_id",
                "chebi_accession",
                "canonical_name",
                "pathway_name_normalized",
                "plant_support_source",
                "plant_pathway_ids",
                "plant_support_examples",
                "plant_support_bonus",
                "plant_support_gene_count",
            ],
            delimiter="\t",
        )
        writer.writeheader()
        for compound_id in sorted(context.compounds, key=int):
            compound = context.compounds[compound_id]
            compound_context = context.compound_contexts[compound_id]

            for normalized_name, names in sorted(compound_context.aracyc_pathways.items()):
                pathway_stat = context.plantcyc_pathway_stats["AraCyc"].get(normalized_name, {})
                gene_count = len(pathway_stat.get("gene_ids", set()))
                writer.writerow(
                    {
                        "compound_id": compound_id,
                        "chebi_accession": compound.chebi_accession,
                        "canonical_name": compound.name,
                        "pathway_name_normalized": normalized_name,
                        "plant_support_source": "AraCyc",
                        "plant_pathway_ids": ";".join(sorted(pathway_stat.get("pathway_ids", set()))),
                        "plant_support_examples": ";".join(sorted(names)[:3]),
                        "plant_support_bonus": "0.060",
                        "plant_support_gene_count": gene_count,
                    }
                )
                row_count += 1

            for normalized_name, names in sorted(compound_context.plantcyc_pathways.items()):
                pathway_stat = context.plantcyc_pathway_stats["PlantCyc"].get(normalized_name, {})
                gene_count = len(pathway_stat.get("gene_ids", set()))
                writer.writerow(
                    {
                        "compound_id": compound_id,
                        "chebi_accession": compound.chebi_accession,
                        "canonical_name": compound.name,
                        "pathway_name_normalized": normalized_name,
                        "plant_support_source": "PlantCyc",
                        "plant_pathway_ids": ";".join(sorted(pathway_stat.get("pathway_ids", set()))),
                        "plant_support_examples": ";".join(sorted(names)[:3]),
                        "plant_support_bonus": "0.040",
                        "plant_support_gene_count": gene_count,
                    }
                )
                row_count += 1
    return row_count


def build_annotation_row(
    *,
    context: PipelineContext,
    pathway_id: str,
    pathway_target_type: str,
    map_pathway_id: str,
    ath_pathway_id: str,
    pathway_name: str,
    ath_gene_sets: dict[str, set[str]],
) -> tuple[dict[str, object], Step5AnnotatedPathwayHit]:
    """Build one fully annotated pathway record and its in-memory counterpart."""

    brite_l1, brite_l2, brite_l3 = context.pathway_categories.get(map_pathway_id, ("", "", ""))
    normalized_name = normalize_name(pathway_name)
    reactome_matches = tuple(context.reactome_pathways.get(normalized_name, [])[:3])
    has_aracyc = normalized_name in context.plantcyc_pathway_stats["AraCyc"]
    has_plantcyc = normalized_name in context.plantcyc_pathway_stats["PlantCyc"]
    plant_evidence_sources = tuple(
        source
        for source, enabled in (("AraCyc", has_aracyc), ("PlantCyc", has_plantcyc))
        if enabled
    )
    aracyc_evidence_score = 1.0 if has_aracyc else 0.5 if has_plantcyc else 0.0

    if ath_pathway_id:
        go_payload = context.pathway_go_enrichment.get(
            ath_pathway_id,
            {
                "terms": [],
                "go_best_term": "",
                "go_best_fdr": "",
                "gene_count": 0,
                "background_gene_count": 0,
            },
        )
        gene_ids = ath_gene_sets.get(ath_pathway_id, set())
    else:
        go_payload = {
            "terms": [],
            "go_best_term": "",
            "go_best_fdr": "",
            "gene_count": 0,
            "background_gene_count": 0,
        }
        gene_ids = set()

    plant_reactome_matches = match_plant_reactome_context(
        pathway_name=pathway_name,
        gene_ids=gene_ids,
        brite_l1=brite_l1,
        brite_l2=brite_l2,
        brite_l3=brite_l3,
        plant_reactome_pathways=context.plant_reactome_pathways,
        plant_reactome_gene_sets=context.plant_reactome_gene_sets,
    )
    context.plant_reactome_alignments[pathway_id] = plant_reactome_matches
    top_reactome = plant_reactome_matches[0] if plant_reactome_matches else {}
    plant_reactome_tags = infer_plant_context_tags(
        brite_l1="",
        brite_l2=top_reactome.get("top_level_category", ""),
        brite_l3="",
        pathway_name=top_reactome.get("pathway_name", ""),
        plant_reactome_texts=(top_reactome.get("description", ""),),
    )
    go_texts = tuple(item["go_name"] for item in go_payload["terms"])
    plant_context_tags = infer_plant_context_tags(
        brite_l1=brite_l1,
        brite_l2=brite_l2,
        brite_l3=brite_l3,
        pathway_name=pathway_name,
        has_aracyc=has_aracyc,
        has_plantcyc=has_plantcyc,
        plant_reactome_texts=(
            top_reactome.get("pathway_name", ""),
            top_reactome.get("top_level_category", ""),
            top_reactome.get("description", ""),
        ),
        go_texts=go_texts,
    )
    annotation_confidence = build_annotation_confidence(
        brite_l1=brite_l1,
        go_best_term=go_payload["go_best_term"],
        aracyc_evidence_score=aracyc_evidence_score,
        reactome_matches=reactome_matches,
        plant_evidence_sources=plant_evidence_sources,
        plant_reactome_alignment_confidence=top_reactome.get("alignment_confidence", ""),
    )

    row = {
        "pathway_id": pathway_id,
        "pathway_target_type": pathway_target_type,
        "map_pathway_id": map_pathway_id,
        "ath_pathway_id": ath_pathway_id,
        "pathway_name": pathway_name,
        "map_id": map_pathway_id,
        "kegg_name": pathway_name,
        "brite_l1": brite_l1,
        "brite_l2": brite_l2,
        "brite_l3": brite_l3 or pathway_name,
        "pathway_group": brite_l1,
        "pathway_category": brite_l2,
        "kegg_vstamp": context.step3_vstamp or "unknown",
        "map_pathway_compound_count": context.map_pathway_compound_counts.get(map_pathway_id, 0),
        "ath_gene_count": context.ath_gene_counts.get(ath_pathway_id, 0) if ath_pathway_id else 0,
        "go_top_terms_json": json.dumps(go_payload["terms"], ensure_ascii=False, sort_keys=True),
        "go_best_term": go_payload["go_best_term"],
        "go_best_fdr": f"{go_payload['go_best_fdr']:.6g}" if go_payload["go_best_fdr"] != "" else "",
        "go_vstamp": context.go_vstamp,
        "plant_context_tags": ";".join(plant_context_tags),
        "plant_evidence_sources": ";".join(plant_evidence_sources),
        "aracyc_evidence_score": f"{aracyc_evidence_score:.3f}",
        "reactome_matches": ";".join(f"{pathway_ref}|{species}" for pathway_ref, species in reactome_matches),
        "plant_reactome_matches_json": json.dumps(plant_reactome_matches, ensure_ascii=False, sort_keys=True),
        "plant_reactome_best_id": top_reactome.get("plant_reactome_id", ""),
        "plant_reactome_best_name": top_reactome.get("pathway_name", ""),
        "plant_reactome_best_category": top_reactome.get("top_level_category", ""),
        "plant_reactome_best_description": top_reactome.get("description", ""),
        "plant_reactome_alignment_score": f"{top_reactome.get('alignment_score', 0.0):.4f}" if top_reactome else "",
        "plant_reactome_alignment_confidence": top_reactome.get("alignment_confidence", ""),
        "plant_reactome_tags": ";".join(plant_reactome_tags),
        "plant_reactome_vstamp": context.plant_reactome_vstamp,
        "annotation_confidence": annotation_confidence,
    }
    return row, Step5AnnotatedPathwayHit(
        compound_id="",
        mapping=None,  # placeholder, replaced for per-hit rows below
        map_pathway_id=map_pathway_id,
        ath_pathway_id=ath_pathway_id,
        pathway_target_id=pathway_id,
        pathway_target_type=pathway_target_type,
        pathway_name=pathway_name,
        map_id=map_pathway_id,
        kegg_name=pathway_name,
        brite_l1=brite_l1,
        brite_l2=brite_l2,
        brite_l3=brite_l3 or pathway_name,
        kegg_vstamp=context.step3_vstamp or "unknown",
        pathway_group=brite_l1,
        pathway_category=brite_l2,
        map_pathway_compound_count=context.map_pathway_compound_counts.get(map_pathway_id, 0),
        ath_gene_count=context.ath_gene_counts.get(ath_pathway_id, 0) if ath_pathway_id else 0,
        go_top_terms_json=json.dumps(go_payload["terms"], ensure_ascii=False, sort_keys=True),
        go_best_term=go_payload["go_best_term"],
        go_best_fdr=float(go_payload["go_best_fdr"]) if go_payload["go_best_fdr"] != "" else 0.0,
        go_vstamp=context.go_vstamp,
        plant_context_tags=plant_context_tags,
        plant_evidence_sources=plant_evidence_sources,
        aracyc_evidence_score=aracyc_evidence_score,
        reactome_matches=reactome_matches,
        plant_reactome_matches_json=json.dumps(plant_reactome_matches, ensure_ascii=False, sort_keys=True),
        plant_reactome_best_id=top_reactome.get("plant_reactome_id", ""),
        plant_reactome_best_name=top_reactome.get("pathway_name", ""),
        plant_reactome_best_category=top_reactome.get("top_level_category", ""),
        plant_reactome_best_description=top_reactome.get("description", ""),
        plant_reactome_alignment_score=float(top_reactome.get("alignment_score", 0.0)) if top_reactome else 0.0,
        plant_reactome_alignment_confidence=top_reactome.get("alignment_confidence", ""),
        plant_reactome_tags=plant_reactome_tags,
        plant_reactome_vstamp=context.plant_reactome_vstamp,
        annotation_confidence=annotation_confidence,
        relation_vstamp="",
        direct_link=False,
        support_reaction_count=0,
        support_rids=(),
        has_substrate_role=False,
        has_product_role=False,
        has_both_role=False,
        cofactor_like=False,
        role_summary="",
        reaction_role_score=0.0,
    )


def run(context: PipelineContext) -> PipelineContext:
    """Add pathway metadata needed for scoring and reporting."""

    ensure_go_annotations(context.paths.gene_association_tair_path)
    context.go_vstamp = iso_mtime(context.paths.gene_association_tair_path)
    context.plant_reactome_vstamp = ensure_plant_reactome_refs(
        context.paths.plant_reactome_pathways_path,
        context.paths.plant_reactome_gene_pathway_path,
        context.paths.plant_reactome_version_path,
    )

    context.map_pathways = load_map_pathways(context.paths.kegg_pathway_map_path)
    context.pathway_categories = load_pathway_categories(context.paths.kegg_pathway_hierarchy_path)
    context.ath_gene_counts = load_ath_gene_counts(context.paths.kegg_ath_gene_pathway_path)
    context.reactome_pathways = load_reactome_pathways(context.paths.reactome_pathways_path)
    context.plantcyc_pathway_stats = load_plantcyc_pathway_stats(
        [
            ("AraCyc", context.paths.aracyc_pathways_path),
            ("PlantCyc", context.paths.plantcyc_pathways_path),
        ]
    )
    context.go_gene_index, context.go_term_namespace, context.go_vstamp = load_gene_to_go_bp(
        context.paths.gene_association_tair_path,
        context.paths.go_basic_obo_path,
    )
    term_to_genes, go_names = build_go_term_gene_sets(context.go_gene_index)
    context.plant_reactome_pathways = load_plant_reactome_pathways(context.paths.plant_reactome_pathways_path)
    context.plant_reactome_gene_sets = load_plant_reactome_gene_index(context.paths.plant_reactome_gene_pathway_path)

    ath_gene_sets = load_ath_gene_sets(context.paths.kegg_ath_gene_pathway_path)
    kegg_ath_genes = set()
    for genes in ath_gene_sets.values():
        kegg_ath_genes.update(genes)
    background_genes = set(context.go_gene_index) & kegg_ath_genes

    pathway_go_enrichment = {}
    for ath_id, gene_ids in ath_gene_sets.items():
        pathway_go_enrichment[ath_id] = compute_pathway_go_enrichment(
            gene_ids,
            term_to_genes,
            go_names,
            background_genes,
        )
    context.pathway_go_enrichment = pathway_go_enrichment

    annotation_rows = []
    annotation_lookup: dict[str, Step5AnnotatedPathwayHit] = {}
    map_ids = sorted(set(context.map_pathways) | set(context.map_to_ath))
    for map_pathway_id in map_ids:
        map_name = context.map_pathways.get(map_pathway_id, map_pathway_id)
        row, template = build_annotation_row(
            context=context,
            pathway_id=map_pathway_id,
            pathway_target_type="map_fallback",
            map_pathway_id=map_pathway_id,
            ath_pathway_id="",
            pathway_name=map_name,
            ath_gene_sets=ath_gene_sets,
        )
        annotation_rows.append(row)
        annotation_lookup[map_pathway_id] = template

        ath_pathway_id = context.map_to_ath.get(map_pathway_id, "")
        if not ath_pathway_id:
            continue
        ath_name = context.ath_pathways.get(ath_pathway_id, ath_pathway_id)
        row, template = build_annotation_row(
            context=context,
            pathway_id=ath_pathway_id,
            pathway_target_type="ath",
            map_pathway_id=map_pathway_id,
            ath_pathway_id=ath_pathway_id,
            pathway_name=ath_name,
            ath_gene_sets=ath_gene_sets,
        )
        annotation_rows.append(row)
        annotation_lookup[ath_pathway_id] = template

    annotated_hits: dict[str, list[Step5AnnotatedPathwayHit]] = {}
    for compound_id, hits in context.resolved_pathway_hits.items():
        annotated = []
        for hit in hits:
            template = annotation_lookup[hit.pathway_target_id]
            annotated.append(
                Step5AnnotatedPathwayHit(
                    compound_id=compound_id,
                    mapping=hit.mapping,
                    map_pathway_id=hit.map_pathway_id,
                    ath_pathway_id=hit.ath_pathway_id,
                    pathway_target_id=hit.pathway_target_id,
                    pathway_target_type=hit.pathway_target_type,
                    pathway_name=template.pathway_name,
                    map_id=template.map_id,
                    kegg_name=template.kegg_name,
                    brite_l1=template.brite_l1,
                    brite_l2=template.brite_l2,
                    brite_l3=template.brite_l3,
                    kegg_vstamp=template.kegg_vstamp,
                    pathway_group=template.pathway_group,
                    pathway_category=template.pathway_category,
                    map_pathway_compound_count=template.map_pathway_compound_count,
                    ath_gene_count=template.ath_gene_count,
                    go_top_terms_json=template.go_top_terms_json,
                    go_best_term=template.go_best_term,
                    go_best_fdr=template.go_best_fdr,
                    go_vstamp=template.go_vstamp,
                    plant_context_tags=template.plant_context_tags,
                    plant_evidence_sources=template.plant_evidence_sources,
                    aracyc_evidence_score=template.aracyc_evidence_score,
                    reactome_matches=template.reactome_matches,
                    plant_reactome_matches_json=template.plant_reactome_matches_json,
                    plant_reactome_best_id=template.plant_reactome_best_id,
                    plant_reactome_best_name=template.plant_reactome_best_name,
                    plant_reactome_best_category=template.plant_reactome_best_category,
                    plant_reactome_best_description=template.plant_reactome_best_description,
                    plant_reactome_alignment_score=template.plant_reactome_alignment_score,
                    plant_reactome_alignment_confidence=template.plant_reactome_alignment_confidence,
                    plant_reactome_tags=template.plant_reactome_tags,
                    plant_reactome_vstamp=template.plant_reactome_vstamp,
                    annotation_confidence=template.annotation_confidence,
                    relation_vstamp=hit.relation_vstamp,
                    direct_link=hit.direct_link,
                    support_reaction_count=hit.support_reaction_count,
                    support_rids=hit.support_rids,
                    has_substrate_role=hit.has_substrate_role,
                    has_product_role=hit.has_product_role,
                    has_both_role=hit.has_both_role,
                    cofactor_like=hit.cofactor_like,
                    role_summary=hit.role_summary,
                    reaction_role_score=hit.reaction_role_score,
                )
            )
        annotated_hits[compound_id] = annotated

    context.annotated_pathway_hits = annotated_hits
    context.preprocess_counts["pathway_annotation_index"] = write_pathway_annotation_index(annotation_rows, context)
    context.preprocess_counts["pathway_go_enrichment"] = write_pathway_go_enrichment(context)
    context.preprocess_counts["gene_to_go_index"] = write_gene_to_go_index(context)
    plant_reactome_pathway_count, plant_reactome_gene_count = write_plant_reactome_indexes(context)
    context.preprocess_counts["plant_reactome_pathway_index"] = plant_reactome_pathway_count
    context.preprocess_counts["plant_reactome_gene_index"] = plant_reactome_gene_count
    context.preprocess_counts["plant_reactome_alignment"] = write_plant_reactome_alignment(context)
    if context.compounds and context.compound_contexts:
        context.preprocess_counts["plant_evidence_index"] = write_plant_evidence_index(context)
    elif context.paths.plant_evidence_index_path.exists():
        with context.paths.plant_evidence_index_path.open(encoding="utf-8", newline="") as handle:
            existing_rows = max(sum(1 for _ in handle) - 1, 0)
        context.preprocess_counts["plant_evidence_index"] = existing_rows
        context.add_note("Step 5 kept the existing plant_evidence_index because compound-level PMN contexts were not loaded.")
    else:
        context.preprocess_counts["plant_evidence_index"] = 0
        context.add_note("Step 5 skipped plant_evidence_index because compound-level PMN contexts were not loaded.")
    context.add_note(
        "Step 5 enriches pathways with KEGG BRITE, Arabidopsis GO BP enrichment, PMN plant evidence, and Plant Reactome local alignment context."
    )
    return context


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for standalone step-5 execution."""

    return build_parser(
        description="Run pathway pipeline steps 1-5: annotate resolved pathways with KEGG BRITE, GO, PMN, and Plant Reactome context.",
        default_output_tag="step5_cli",
    ).parse_args()


def main() -> None:
    """Run steps 1-5 and report annotation coverage."""

    args = parse_args()
    context = build_context(workdir=args.workdir, output_tag=args.output_tag)
    run_step1(context)
    run_step2(context)
    run_step3(context)
    run_step4(context)
    run(context)
    annotated_count = sum(len(hits) for hits in context.annotated_pathway_hits.values())
    print_summary(
        "Step 5 completed.",
        [
            f"Annotated pathway hits: {annotated_count}",
            f"Pathway annotation index: {context.paths.pathway_annotation_index_path}",
            f"GO enrichment index: {context.paths.pathway_go_enrichment_index_path}",
            f"Plant evidence index: {context.paths.plant_evidence_index_path}",
            f"Plant Reactome alignment index: {context.paths.plant_reactome_alignment_path}",
        ],
    )


if __name__ == "__main__":
    main()
