#!/usr/bin/env python3
"""Run the refactored step-wise ChEBI to pathway pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

from pathway_pipeline.context import PipelineContext, PipelinePaths
from pathway_pipeline import (
    step1_alias_standardization,
    step2_map_names_to_kegg,
    step3_link_compounds_to_pathways,
    step4_map_to_ath,
    step5_annotate_pathways,
    step6_score_pathways,
    step7_output_results,
    step8_similar_metabolite_fallback,
    step9_plantcyc_enhancement,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the refactored pipeline."""

    parser = argparse.ArgumentParser(description="Run the step-wise ChEBI to KEGG/AraCyc/PlantCyc pathway pipeline.")
    parser.add_argument("--workdir", default=".", help="Workspace containing compounds.tsv, comments.tsv, refs/, and outputs/.")
    parser.add_argument("--output-tag", default="refactored", help="Suffix added to the generated output files.")
    parser.add_argument("--enable-step8", action="store_true", help="Enable the optional similarity-fallback placeholder hook.")
    parser.add_argument("--enable-step9", action="store_true", help="Enable the optional PlantCyc-enhancement placeholder hook.")
    return parser.parse_args()


def main() -> None:
    """Execute the pipeline in the same order as the table steps."""

    args = parse_args()
    context = PipelineContext(
        paths=PipelinePaths.from_workdir(Path(args.workdir), output_tag=args.output_tag)
    )

    step1_alias_standardization.run(context)
    step2_map_names_to_kegg.run(context)
    step3_link_compounds_to_pathways.run(context)
    step4_map_to_ath.run(context)
    step5_annotate_pathways.run(context)
    step8_similar_metabolite_fallback.run(context, enabled=args.enable_step8)
    step9_plantcyc_enhancement.run(context, enabled=args.enable_step9)
    step6_score_pathways.run(context)
    step7_output_results.run(context)


if __name__ == "__main__":
    main()
