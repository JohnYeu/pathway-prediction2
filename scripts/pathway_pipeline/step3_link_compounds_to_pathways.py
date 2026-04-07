"""Step 3: expand selected KEGG compounds to raw map pathways.

This step is intentionally narrow: it does not rank or filter pathways yet.
It only answers the question "for each selected KEGG compound, which KEGG map
pathways are linked to it?".
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from process_chebi_to_pathways_v2 import load_kegg_pathway_links

from pathway_pipeline.cli_utils import build_context, build_parser, print_summary
from pathway_pipeline.context import PipelineContext, Step3PathwayHit
from pathway_pipeline.step1_alias_standardization import run as run_step1
from pathway_pipeline.step2_map_names_to_kegg import run as run_step2


def run(context: PipelineContext) -> PipelineContext:
    """Translate selected KEGG compound IDs into raw map pathway hits.

    The output keeps one record per:
    - input ChEBI compound
    - selected KEGG compound mapping
    - linked KEGG map pathway

    Step 4 will decide whether each map pathway can be upgraded to an ath
    pathway, so step 3 deliberately keeps only the raw map-level relation.
    """

    # Load the canonical KEGG compound -> pathway relation table. At this stage
    # we do not need ath conversion yet, so an empty map_to_ath table is passed
    # on purpose and only the map IDs are used.
    context.kegg_to_pathways, context.map_pathway_compound_counts = load_kegg_pathway_links(
        context.paths.kegg_compound_pathway_path,
        {},
    )

    raw_hits = {}

    # Expand every selected KEGG compound to all map pathways directly linked
    # to it in KEGG.
    for compound_id, selected_mappings in context.selected_by_compound.items():
        hits = []
        for mapping in selected_mappings:
            # Keep the originating mapping on each hit so later steps can still
            # trace pathway evidence back to the exact KEGG compound candidate.
            for map_pathway_id, _unused_ath in context.kegg_to_pathways.get(mapping.kegg_compound_id, []):
                hits.append(
                    Step3PathwayHit(
                        compound_id=compound_id,
                        mapping=mapping,
                        map_pathway_id=map_pathway_id,
                    )
                )
        raw_hits[compound_id] = hits

    # Store the raw, unresolved pathway hits for step 4.
    context.raw_pathway_hits = raw_hits
    return context


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for standalone step-3 execution."""

    return build_parser(
        description="Run pathway pipeline steps 1-3: stop after KEGG compound-to-pathway linking.",
        default_output_tag="step3_cli",
    ).parse_args()


def main() -> None:
    """Run steps 1-3 and report raw pathway-link counts."""

    args = parse_args()
    context = build_context(workdir=args.workdir, output_tag=args.output_tag)
    run_step1(context)
    run_step2(context)
    run(context)
    raw_hit_count = sum(len(hits) for hits in context.raw_pathway_hits.values())
    print_summary(
        "Step 3 completed.",
        [
            f"Mapping summary: {context.paths.mapping_summary_path}",
            f"Selected mappings: {context.paths.mapping_selected_path}",
            f"Compounds with selected mappings: {sum(1 for hits in context.selected_by_compound.values() if hits)}",
            f"Raw map-pathway hits: {raw_hit_count}",
        ],
    )


if __name__ == "__main__":
    main()
