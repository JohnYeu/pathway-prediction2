"""Step 4: resolve map pathways to ath pathways with fallback handling.

The pathway table in step 3 is still at KEGG map-level resolution. This step
adds the plant-specific rule from the design table:

- if a map pathway has an ath-specific counterpart, prefer the ath pathway
- otherwise keep the original map pathway as a lower-confidence fallback
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from process_chebi_to_pathways_v2 import load_ath_pathways

from pathway_pipeline.cli_utils import build_context, build_parser, print_summary
from pathway_pipeline.context import PipelineContext, Step4ResolvedPathwayHit
from pathway_pipeline.step1_alias_standardization import run as run_step1
from pathway_pipeline.step2_map_names_to_kegg import run as run_step2
from pathway_pipeline.step3_link_compounds_to_pathways import run as run_step3


def run(context: PipelineContext) -> PipelineContext:
    """Prefer ath pathways when they exist, otherwise keep map fallbacks."""

    # Load both the ath pathway names and the mapXXXX -> athXXXX conversion
    # table derived from the KEGG ath pathway list.
    context.ath_pathways, context.map_to_ath = load_ath_pathways(context.paths.kegg_pathway_ath_path)

    resolved_hits = {}

    # Convert each raw map hit into either:
    # - an ath-resolved hit
    # - or a map fallback hit when no ath pathway exists
    for compound_id, hits in context.raw_pathway_hits.items():
        resolved = []
        for hit in hits:
            ath_pathway_id = context.map_to_ath.get(hit.map_pathway_id, "")

            # The target ID is the one that downstream ranking/reporting will
            # display. It is ath-prefixed when available, otherwise the map ID.
            pathway_target_id = ath_pathway_id or hit.map_pathway_id

            # Keep the target type explicit so step 6 can penalize fallback rows
            # without losing the original evidence.
            pathway_target_type = "ath" if ath_pathway_id else "map_fallback"
            resolved.append(
                Step4ResolvedPathwayHit(
                    compound_id=compound_id,
                    mapping=hit.mapping,
                    map_pathway_id=hit.map_pathway_id,
                    ath_pathway_id=ath_pathway_id,
                    pathway_target_id=pathway_target_id,
                    pathway_target_type=pathway_target_type,
                )
            )
        resolved_hits[compound_id] = resolved

    # Persist the resolved view for annotation.
    context.resolved_pathway_hits = resolved_hits
    return context


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for standalone step-4 execution."""

    return build_parser(
        description="Run pathway pipeline steps 1-4: resolve KEGG map pathways to ath with fallback handling.",
        default_output_tag="step4_cli",
    ).parse_args()


def main() -> None:
    """Run steps 1-4 and report ath-vs-fallback counts."""

    args = parse_args()
    context = build_context(workdir=args.workdir, output_tag=args.output_tag)
    run_step1(context)
    run_step2(context)
    run_step3(context)
    run(context)
    ath_count = sum(1 for hits in context.resolved_pathway_hits.values() for hit in hits if hit.pathway_target_type == "ath")
    fallback_count = sum(1 for hits in context.resolved_pathway_hits.values() for hit in hits if hit.pathway_target_type == "map_fallback")
    print_summary(
        "Step 4 completed.",
        [
            f"Resolved ath pathway hits: {ath_count}",
            f"Map fallback hits: {fallback_count}",
            f"ath pathway reference table: {context.paths.kegg_pathway_ath_path}",
        ],
    )


if __name__ == "__main__":
    main()
