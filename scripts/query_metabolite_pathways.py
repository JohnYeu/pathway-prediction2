#!/usr/bin/env python3
"""Interactive terminal query for metabolite -> standard name -> top pathways.

This script reads the already-generated v2 outputs and provides a lightweight
REPL for end users:

1. the user types a metabolite name
2. the backend resolves it to the best supported standard compound name
3. the script prints the top-ranked pathways for that compound

It is intentionally read-only and fast enough for repeated terminal use.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from process_chebi_to_pathways_v2 import ALIAS_SOURCE_WEIGHTS, build_variants


VARIANT_PRIORITY = {
    "exact": 1.00,
    "compact": 0.97,
    "singular": 0.95,
    "stereo_stripped": 0.93,
}

GENERIC_PATHWAY_CATEGORY = "Global and overview maps"


@dataclass(slots=True)
class AliasHit:
    compound_id: str
    chebi_accession: str
    chebi_name: str
    alias: str
    source_type: str


@dataclass(slots=True)
class MappingInfo:
    kegg_compound_id: str
    kegg_primary_name: str
    mapping_score: float
    mapping_confidence_level: str
    mapping_method: str


@dataclass(slots=True)
class PathwayInfo:
    pathway_rank: int
    score: float
    confidence_level: str
    pathway_target_id: str
    pathway_name: str
    pathway_category: str
    reason: str


@dataclass(slots=True)
class ResolvedCompound:
    compound_id: str
    chebi_accession: str
    chebi_name: str
    matched_alias: str
    alias_source: str
    matched_variant: str
    candidate_score: float
    mapping: MappingInfo | None
    pathways: list[PathwayInfo]


def ensure_exists(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")


def load_alias_indexes(
    path: Path,
) -> dict[str, dict[str, dict[str, AliasHit]]]:
    """Load alias output into variant-specific indexes.

    Structure:
    variant_name -> normalized_value -> compound_id -> AliasHit
    """

    indexes = {
        "exact": defaultdict(dict),
        "compact": defaultdict(dict),
        "singular": defaultdict(dict),
        "stereo_stripped": defaultdict(dict),
    }
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            hit = AliasHit(
                compound_id=row["compound_id"],
                chebi_accession=row["chebi_accession"],
                chebi_name=row["chebi_name"],
                alias=row["alias"],
                source_type=row["source_type"],
            )
            for variant_name, field_name in (
                ("exact", "normalized_name"),
                ("compact", "compact_name"),
                ("singular", "singular_name"),
                ("stereo_stripped", "stereo_stripped_name"),
            ):
                value = row[field_name]
                if not value:
                    continue
                existing = indexes[variant_name][value].get(hit.compound_id)
                if existing is None or ALIAS_SOURCE_WEIGHTS.get(hit.source_type, 0.0) > ALIAS_SOURCE_WEIGHTS.get(existing.source_type, 0.0):
                    indexes[variant_name][value][hit.compound_id] = hit
    return indexes


def load_mapping_info(path: Path) -> dict[str, MappingInfo]:
    """Load one best KEGG mapping row per compound."""

    info_by_compound: dict[str, MappingInfo] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            mapping = MappingInfo(
                kegg_compound_id=row["kegg_compound_id"],
                kegg_primary_name=row["kegg_primary_name"],
                mapping_score=float(row["mapping_score"]),
                mapping_confidence_level=row["mapping_confidence_level"],
                mapping_method=row["mapping_method"],
            )
            current = info_by_compound.get(row["compound_id"])
            if current is None or mapping.mapping_score > current.mapping_score:
                info_by_compound[row["compound_id"]] = mapping
    return info_by_compound


def load_pathway_info(path: Path) -> dict[str, list[PathwayInfo]]:
    """Load ranked pathways grouped by compound."""

    pathways: defaultdict[str, list[PathwayInfo]] = defaultdict(list)
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            pathways[row["compound_id"]].append(
                PathwayInfo(
                    pathway_rank=int(row["pathway_rank"]),
                    score=float(row["score"]),
                    confidence_level=row["confidence_level"],
                    pathway_target_id=row["pathway_target_id"],
                    pathway_name=row["pathway_name"],
                    pathway_category=row["pathway_category"],
                    reason=row["reason"],
                )
            )
    for compound_id, rows in pathways.items():
        rows.sort(key=lambda item: (-item.score, item.pathway_rank, item.pathway_target_id))
    return dict(pathways)


def choose_display_pathways(pathways: list[PathwayInfo], top_k: int, include_generic: bool) -> list[PathwayInfo]:
    """Prefer specific pathways, then fall back to generic ones if needed."""

    if include_generic:
        return pathways[:top_k]
    specific = [row for row in pathways if row.pathway_category != GENERIC_PATHWAY_CATEGORY]
    if len(specific) >= top_k:
        return specific[:top_k]
    combined = specific + [row for row in pathways if row.pathway_category == GENERIC_PATHWAY_CATEGORY]
    return combined[:top_k]


def resolve_compound(
    query: str,
    alias_indexes: dict[str, dict[str, dict[str, AliasHit]]],
    mapping_info: dict[str, MappingInfo],
    pathway_info: dict[str, list[PathwayInfo]],
) -> ResolvedCompound | None:
    """Resolve one free-text query to the best supported compound."""

    variants = build_variants(query)
    candidates: dict[str, ResolvedCompound] = {}
    for variant_name in ("exact", "compact", "singular", "stereo_stripped"):
        value = variants[variant_name]
        if not value:
            continue
        hits = alias_indexes[variant_name].get(value)
        if not hits:
            continue
        for compound_id, hit in hits.items():
            mapping = mapping_info.get(compound_id)
            pathways = pathway_info.get(compound_id, [])
            top_pathway_score = pathways[0].score if pathways else 0.0
            score = (
                VARIANT_PRIORITY[variant_name]
                + ALIAS_SOURCE_WEIGHTS.get(hit.source_type, 0.0)
                + (mapping.mapping_score * 0.10 if mapping else 0.0)
                + (top_pathway_score * 0.05 if pathways else 0.0)
            )
            current = candidates.get(compound_id)
            if current is None or score > current.candidate_score:
                candidates[compound_id] = ResolvedCompound(
                    compound_id=compound_id,
                    chebi_accession=hit.chebi_accession,
                    chebi_name=hit.chebi_name,
                    matched_alias=hit.alias,
                    alias_source=hit.source_type,
                    matched_variant=variant_name,
                    candidate_score=score,
                    mapping=mapping,
                    pathways=pathways,
                )
        # Stop at the first variant tier that produced hits.
        break
    if not candidates:
        return None
    ranked = sorted(
        candidates.values(),
        key=lambda item: (
            item.candidate_score,
            item.mapping.mapping_score if item.mapping else 0.0,
            item.pathways[0].score if item.pathways else 0.0,
            item.chebi_name,
        ),
        reverse=True,
    )
    return ranked[0]


def print_result(result: ResolvedCompound, top_k: int, include_generic: bool) -> None:
    """Render one query result in a terminal-friendly format."""

    print(f"Standard name: {result.chebi_name} ({result.chebi_accession})")
    print(
        "Matched by: "
        f"{result.matched_alias} "
        f"[source={result.alias_source}, variant={result.matched_variant}]"
    )
    if result.mapping:
        print(
            "KEGG: "
            f"{result.mapping.kegg_compound_id} {result.mapping.kegg_primary_name} "
            f"[score={result.mapping.mapping_score:.3f}, method={result.mapping.mapping_method}]"
        )
    else:
        print("KEGG: no selected mapping")

    pathways = choose_display_pathways(result.pathways, top_k=top_k, include_generic=include_generic)
    if not pathways:
        print("Top pathways: no ranked pathway available")
        return

    print(f"Top {len(pathways)} pathways:")
    for index, pathway in enumerate(pathways, start=1):
        print(
            f"{index}. {pathway.pathway_name} "
            f"[{pathway.pathway_target_id}, score={pathway.score:.3f}, confidence={pathway.confidence_level}]"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Query top-ranked pathways for a metabolite name.")
    parser.add_argument("--workdir", default=".", help="Workspace containing outputs/ from the v2 pipeline.")
    parser.add_argument("--name", help="One-shot metabolite query. If omitted, start an interactive prompt.")
    parser.add_argument("--top-k", type=int, default=3, help="Number of pathway names to print.")
    parser.add_argument(
        "--include-generic",
        action="store_true",
        help="Include generic KEGG pathways such as 'Metabolic pathways' in the top-k output.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    workdir = Path(args.workdir).resolve()
    outputs = workdir / "outputs"

    alias_path = outputs / "chebi_aliases_standardized_v2.tsv"
    mapping_path = outputs / "chebi_kegg_selected_v2.tsv"
    pathway_path = outputs / "chebi_pathways_ranked_v2.tsv"

    for path in (alias_path, mapping_path, pathway_path):
        ensure_exists(path)

    alias_indexes = load_alias_indexes(alias_path)
    mapping_info = load_mapping_info(mapping_path)
    pathway_info = load_pathway_info(pathway_path)

    def handle_query(query: str) -> None:
        result = resolve_compound(query, alias_indexes, mapping_info, pathway_info)
        if result is None:
            print("No matched metabolite found.")
            return
        print_result(result, top_k=args.top_k, include_generic=args.include_generic)

    if args.name:
        handle_query(args.name.strip())
        return

    print("Enter a metabolite name. Type 'exit' or press Enter on an empty line to quit.")
    while True:
        try:
            query = input("metabolite> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not query or query.lower() in {"exit", "quit", "q"}:
            break
        handle_query(query)
        print()


if __name__ == "__main__":
    main()
