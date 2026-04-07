"""Step 9 placeholder: dedicated PlantCyc re-ranking hook.

The table also reserves an optional late-stage PlantCyc enhancement step. The
current pipeline already uses PlantCyc/AraCyc evidence earlier in the process,
but this module keeps a dedicated extension point for any future post-ranking
boost or re-ranking rule set.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from pathway_pipeline.cli_utils import build_context, build_parser, print_summary
from pathway_pipeline.context import PipelineContext
from pathway_pipeline.step1_alias_standardization import run as run_step1
from pathway_pipeline.step2_map_names_to_kegg import run as run_step2
from pathway_pipeline.step3_link_compounds_to_pathways import run as run_step3
from pathway_pipeline.step4_map_to_ath import run as run_step4
from pathway_pipeline.step5_annotate_pathways import run as run_step5
from pathway_pipeline.step6_score_pathways import run as run_step6
from pathway_pipeline.step7_output_results import run as run_step7


def run(context: PipelineContext, enabled: bool = False) -> PipelineContext:
    """Reserve a dedicated post-annotation PlantCyc enhancement stage.

    For now this hook is documentation-friendly scaffolding: it confirms that
    the pipeline has a stable place for future PlantCyc-specific post-processing
    without changing the current ranked outputs.
    """

    # Just like step 8, keep the default behavior side-effect free until the
    # enhancement logic is specified and implemented.
    if enabled:
        context.add_note(
            "Step 9 placeholder was enabled. Current PlantCyc/AraCyc evidence is already used during alias expansion, annotation, and scoring, but no extra post-ranking enhancement module has been added yet."
        )
    return context


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for standalone step-9 execution."""

    return build_parser(
        description="Run the pipeline with optional step 9 enabled, then write the final outputs.",
        default_output_tag="step9_cli",
    ).parse_args()


def main() -> None:
    """Run the full pipeline with the step-9 placeholder enabled."""

    args = parse_args()
    context = build_context(workdir=args.workdir, output_tag=args.output_tag)
    run_step1(context)
    run_step2(context)
    run_step3(context)
    run_step4(context)
    run_step5(context)
    run(context, enabled=True)
    run_step6(context)
    run_step7(context)
    print_summary(
        "Step 9 placeholder pipeline completed.",
        [
            f"Ranked pathways: {context.paths.pathway_output_path}",
            f"Summary JSON: {context.paths.summary_output_path}",
            "Step 9 currently records a note only and does not change scores.",
        ],
    )


if __name__ == "__main__":
    main()
