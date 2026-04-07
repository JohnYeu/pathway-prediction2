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
                )
            )
        annotated_hits[compound_id] = annotated

    # Persist the metadata-enriched pathway hits for step 6 scoring.
    context.annotated_pathway_hits = annotated_hits
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
            f"Pathway category table: {context.paths.kegg_pathway_hierarchy_path}",
            f"Reactome context table: {context.paths.reactome_pathways_path}",
        ],
    )


if __name__ == "__main__":
    main()
