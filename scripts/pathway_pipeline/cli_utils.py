"""Shared CLI helpers for standalone step execution."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from pathway_pipeline.context import PipelineContext, PipelinePaths


def build_parser(*, description: str, default_output_tag: str) -> argparse.ArgumentParser:
    """Create a consistent CLI parser for standalone step commands."""

    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--workdir", default=".", help="Workspace containing compounds.tsv, comments.tsv, refs/, and outputs/.")
    parser.add_argument(
        "--output-tag",
        default=default_output_tag,
        help="Suffix added to the generated output files for this standalone run.",
    )
    return parser


def build_context(*, workdir: str, output_tag: str) -> PipelineContext:
    """Construct a fresh pipeline context for a standalone step run."""

    return PipelineContext(
        paths=PipelinePaths.from_workdir(Path(workdir), output_tag=output_tag)
    )


def format_counter(counter: Counter[str] | dict[str, int]) -> str:
    """Render counters compactly for terminal summaries."""

    items = sorted(counter.items())
    return ", ".join(f"{key}={value}" for key, value in items) if items else "none"


def print_summary(title: str, lines: list[str]) -> None:
    """Emit a short human-readable CLI summary."""

    print(title)
    for line in lines:
        print(f"- {line}")
