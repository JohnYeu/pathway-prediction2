"""Step 5: annotate pathways with names, categories, genes, and Reactome context.

This step enriches the resolved pathway hits with metadata needed for both
human-readable output and scoring:

- display name
- KEGG group/category
- ath gene support count
- exact-name Reactome context
- PlantCyc/AraCyc pathway statistics used later in scoring
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from process_chebi_to_pathways_v2 import (
    load_ath_gene_counts,
    load_map_pathways,
    load_pathway_categories,
    load_plantcyc_pathway_stats,
    load_reactome_pathways,
    normalize_name,
)

from pathway_pipeline.cli_utils import build_context, build_parser, print_summary
from pathway_pipeline.context import PipelineContext, Step5AnnotatedPathwayHit
from pathway_pipeline.step1_alias_standardization import run as run_step1
from pathway_pipeline.step2_map_names_to_kegg import run as run_step2
from pathway_pipeline.step3_link_compounds_to_pathways import run as run_step3
from pathway_pipeline.step4_map_to_ath import run as run_step4


def write_pathway_annotation_index(context: PipelineContext) -> int:
    """Write a unified map/ath annotation table consumed by query step 5."""

    row_count = 0
    map_ids = sorted(set(context.map_pathways) | set(context.map_to_ath))
    with context.paths.pathway_annotation_index_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "pathway_id",
                "pathway_target_type",
                "map_pathway_id",
                "ath_pathway_id",
                "pathway_name",
                "pathway_group",
                "pathway_category",
                "map_pathway_compound_count",
                "ath_gene_count",
                "reactome_matches",
            ],
            delimiter="\t",
        )
        writer.writeheader()
        for map_pathway_id in map_ids:
            group, category, _unused = context.pathway_categories.get(map_pathway_id, ("", "", ""))
            map_name = context.map_pathways.get(map_pathway_id, map_pathway_id)
            writer.writerow(
                {
                    "pathway_id": map_pathway_id,
                    "pathway_target_type": "map_fallback",
                    "map_pathway_id": map_pathway_id,
                    "ath_pathway_id": "",
                    "pathway_name": map_name,
                    "pathway_group": group,
                    "pathway_category": category,
                    "map_pathway_compound_count": context.map_pathway_compound_counts.get(map_pathway_id, 0),
                    "ath_gene_count": 0,
                    "reactome_matches": ";".join(
                        f"{pathway_id}|{species}"
                        for pathway_id, species in context.reactome_pathways.get(normalize_name(map_name), [])[:3]
                    ),
                }
            )
            row_count += 1

            ath_pathway_id = context.map_to_ath.get(map_pathway_id, "")
            if not ath_pathway_id:
                continue
            ath_name = context.ath_pathways.get(ath_pathway_id, ath_pathway_id)
            writer.writerow(
                {
                    "pathway_id": ath_pathway_id,
                    "pathway_target_type": "ath",
                    "map_pathway_id": map_pathway_id,
                    "ath_pathway_id": ath_pathway_id,
                    "pathway_name": ath_name,
                    "pathway_group": group,
                    "pathway_category": category,
                    "map_pathway_compound_count": context.map_pathway_compound_counts.get(map_pathway_id, 0),
                    "ath_gene_count": context.ath_gene_counts.get(ath_pathway_id, 0),
                    "reactome_matches": ";".join(
                        f"{pathway_id}|{species}"
                        for pathway_id, species in context.reactome_pathways.get(normalize_name(ath_name), [])[:3]
                    ),
                }
            )
            row_count += 1
    return row_count


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


def run(context: PipelineContext) -> PipelineContext:
    """Add pathway metadata needed for scoring and reporting."""

    # Load the static metadata tables that describe pathways rather than
    # compound-to-pathway relations.
    context.map_pathways = load_map_pathways(context.paths.kegg_pathway_map_path)
    context.pathway_categories = load_pathway_categories(context.paths.kegg_pathway_hierarchy_path)
    context.ath_gene_counts = load_ath_gene_counts(context.paths.kegg_ath_gene_pathway_path)
    context.reactome_pathways = load_reactome_pathways(context.paths.reactome_pathways_path)

    # PlantCyc/AraCyc pathway stats are loaded here rather than in step 1
    # because they support pathway annotation/ranking, not alias expansion.
    context.plantcyc_pathway_stats = load_plantcyc_pathway_stats(
        [
            ("AraCyc", context.paths.aracyc_pathways_path),
            ("PlantCyc", context.paths.plantcyc_pathways_path),
        ]
    )

    annotated_hits = {}

    # Attach annotation fields to every resolved pathway hit.
    for compound_id, hits in context.resolved_pathway_hits.items():
        annotated = []
        for hit in hits:
            # Choose the most specific available display name. ath pathway names
            # take precedence; otherwise we fall back to the generic map name.
            pathway_name = context.ath_pathways.get(
                hit.ath_pathway_id,
                context.map_pathways.get(hit.map_pathway_id, hit.map_pathway_id),
            )

            # KEGG hierarchy is keyed by map pathway ID, so category lookup
            # always uses the original map ID even if an ath pathway exists.
            pathway_group, pathway_category, _unused_name = context.pathway_categories.get(hit.map_pathway_id, ("", "", ""))

            # Reactome is used only as exact-name contextual metadata. No ID
            # alignment is attempted here; the goal is to surface parallel
            # pathway labels that may help interpretation later.
            reactome_matches = tuple(context.reactome_pathways.get(normalize_name(pathway_name), []))
            annotated.append(
                Step5AnnotatedPathwayHit(
                    compound_id=compound_id,
                    mapping=hit.mapping,
                    map_pathway_id=hit.map_pathway_id,
                    ath_pathway_id=hit.ath_pathway_id,
                    pathway_target_id=hit.pathway_target_id,
                    pathway_target_type=hit.pathway_target_type,
                    pathway_name=pathway_name,
                    pathway_group=pathway_group,
                    pathway_category=pathway_category,
                    map_pathway_compound_count=context.map_pathway_compound_counts.get(hit.map_pathway_id, 0),
                    ath_gene_count=context.ath_gene_counts.get(hit.ath_pathway_id, 0) if hit.ath_pathway_id else 0,
                    reactome_matches=reactome_matches,
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

    # Persist the metadata-enriched pathway hits for step 6 scoring.
    context.annotated_pathway_hits = annotated_hits
    context.preprocess_counts["pathway_annotation_index"] = write_pathway_annotation_index(context)
    context.preprocess_counts["plant_evidence_index"] = write_plant_evidence_index(context)
    return context


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for standalone step-5 execution."""

    return build_parser(
        description="Run pathway pipeline steps 1-5: annotate resolved pathways with names, categories, genes, and Reactome context.",
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
            f"Plant evidence index: {context.paths.plant_evidence_index_path}",
            f"Pathway category table: {context.paths.kegg_pathway_hierarchy_path}",
            f"Reactome context table: {context.paths.reactome_pathways_path}",
        ],
    )


if __name__ == "__main__":
    main()
