"""Step 8 placeholder: similar-metabolite fallback hook.

The design table reserves this step for "no direct pathway hit" scenarios where
similar metabolites may be used as a lower-confidence fallback. The refactored
pipeline exposes the hook now so the control flow is stable even before the
full RDKit / PubChem similarity logic is added.
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
    """Reserve a stable hook for future similarity-based pathway fallback.

    This function currently does not mutate rankings. It only records a note
    when the placeholder is explicitly enabled, so users can see that step 8
    was requested but not yet implemented.
    """

    # Keep the default behavior as a no-op so step 8 can stay in the main
    # execution order without changing current results.
    if enabled:
        context.add_note(
            "Step 8 placeholder was enabled, but no RDKit/PubChem similarity fallback has been implemented in the refactored pipeline yet."
        )
    return context


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for standalone step-8 execution."""

    return build_parser(
        description="Run the pipeline with optional step 8 enabled, then write the final outputs.",
        default_output_tag="step8_cli",
    ).parse_args()


def main() -> None:
    """Run the full pipeline with the step-8 placeholder enabled."""

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
        "Step 8 placeholder pipeline completed.",
        [
            f"Ranked pathways: {context.paths.pathway_output_path}",
            f"Summary JSON: {context.paths.summary_output_path}",
            "Step 8 currently records a note only and does not change scores.",
        ],
    )


if __name__ == "__main__":
    main()
