#!/usr/bin/env python3
"""Thin CLI entrypoint for the active step-wise preprocessing flow.

The project now uses a single AraCyc-first preprocessing chain under
``scripts/pathway_pipeline`` so the code layout matches the step table
directly.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from pathway_pipeline.cli_utils import format_counter
from pathway_pipeline.context import PipelineContext, PipelinePaths

from pathway_pipeline import (
    step1_standardize_names,
    step2_match_compounds,
    step3_link_compounds_to_pathways,
    step4_annotate_pathways,
    step5_score_pathways,
    step7_output_results,
)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for full preprocessing."""

    parser = argparse.ArgumentParser(description="Build all preprocessed indexes for the name-first pathway query flow.")
    parser.add_argument("--workdir", default=".", help="Workspace containing compounds.tsv, comments.tsv, refs/, and outputs/.")
    return parser.parse_args()


def run_pipeline(context: PipelineContext) -> None:
    """Run the active AraCyc-first preprocessing pipeline."""

    print("=== AraCyc-first pipeline ===", flush=True)
    print("Loading primary datasets...", flush=True)
    step1_standardize_names.run(context)
    print("Matching ChEBI compounds to AraCyc/PlantCyc...", flush=True)
    step2_match_compounds.run(context)
    print("Extracting AraCyc compound-pathway links...", flush=True)
    step3_link_compounds_to_pathways.run(context)
    print("Annotating AraCyc pathway hits...", flush=True)
    step4_annotate_pathways.run(context)
    print("Scoring AraCyc pathway hits...", flush=True)
    step5_score_pathways.run(context)
    print("Writing outputs...", flush=True)
    step7_output_results.run(context)


def main() -> None:
    """Run the step-wise preprocessing pipeline."""

    args = parse_args()
    context = PipelineContext(
        paths=PipelinePaths.from_workdir(Path(args.workdir), output_tag="refactored"),
    )
    run_pipeline(context)

    print("Preprocess completed.", flush=True)
    print("- Pipeline: aracyc", flush=True)
    print(f"- Output directory: {context.paths.preprocessed_dir}", flush=True)
    for key in sorted(context.preprocess_counts):
        print(f"- {key}: {context.preprocess_counts[key]}", flush=True)
    print(f"- Pathway status counts: {format_counter(context.pathway_status)}", flush=True)


if __name__ == "__main__":
    main()
