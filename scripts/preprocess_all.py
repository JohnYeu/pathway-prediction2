#!/usr/bin/env python3
"""Thin CLI entrypoint for the active step-wise preprocessing flow.

The real preprocessing implementation now lives under ``scripts/pathway_pipeline``
so the code layout matches the table steps directly:

1. alias normalization
2. name -> KEGG compound mapping
3. compound -> pathway linking
4. map -> ath resolution
5. pathway annotation + PMN evidence
6. pathway scoring
7. output writing + preprocess metadata
"""

from __future__ import annotations

import argparse
from pathlib import Path

from pathway_pipeline.cli_utils import format_counter
from pathway_pipeline.context import PipelineContext, PipelinePaths
from pathway_pipeline import (
    step1_alias_standardization,
    step2_map_names_to_kegg,
    step3_link_compounds_to_pathways,
    step4_map_to_ath,
    step5_annotate_pathways,
    step6_score_pathways,
    step7_output_results,
    step8_expanded_recall,
)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for full preprocessing."""

    parser = argparse.ArgumentParser(description="Build all preprocessed indexes for the name-first pathway query flow.")
    parser.add_argument("--workdir", default=".", help="Workspace containing compounds.tsv, comments.tsv, refs/, and outputs/.")
    parser.add_argument(
        "--refresh-step3-kegg",
        action="store_true",
        help="Refresh KEGG step-3 refs (info, compound->reaction, reaction->pathway, reaction details) before preprocessing.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the step-wise preprocessing pipeline through step 7."""

    args = parse_args()
    context = PipelineContext(
        paths=PipelinePaths.from_workdir(Path(args.workdir), output_tag="refactored"),
        refresh_step3_kegg=args.refresh_step3_kegg,
    )

    print("Loading primary datasets...", flush=True)
    step1_alias_standardization.run(context)
    print("Building per-compound contexts and KEGG selections...", flush=True)
    step2_map_names_to_kegg.run(context)
    print("Resolving compound-to-pathway relations...", flush=True)
    step3_link_compounds_to_pathways.run(context)
    step4_map_to_ath.run(context)
    step5_annotate_pathways.run(context)
    print("Scoring and writing outputs...", flush=True)
    step6_score_pathways.run(context)
    step8_expanded_recall.run(context)
    step7_output_results.run(context)

    print("Preprocess completed.", flush=True)
    print(f"- Output directory: {context.paths.preprocessed_dir}", flush=True)
    for key in sorted(context.preprocess_counts):
        print(f"- {key}: {context.preprocess_counts[key]}", flush=True)
    print(f"- Pathway status counts: {format_counter(context.pathway_status)}", flush=True)


if __name__ == "__main__":
    main()
