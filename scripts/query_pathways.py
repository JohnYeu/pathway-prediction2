#!/usr/bin/env python3
"""Query-time pathway lookup for the active AraCyc-first pipeline.

Usage:
    python3 query_pathways.py --workdir ..
    python3 query_pathways.py --workdir .. --query "glucose"
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from process_chebi_to_pathways_v2 import build_variants, normalize_name


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class AraCycMappingRecord:
    """One ChEBI → AraCyc compound match from the precomputed index."""

    compound_id: str
    chebi_accession: str
    chebi_name: str
    aracyc_compound_id: str
    aracyc_common_name: str
    source_db: str
    match_method: str
    match_score: float
    chebi_xref_direct: bool
    structure_validated: bool
    pathway_count: int
    pathways: str


@dataclass(slots=True)
class AraCycRankedPathway:
    """One scored pathway from the precomputed output."""

    compound_id: str
    chebi_accession: str
    chebi_name: str
    pathway_rank: int
    score: float
    confidence_level: str
    evidence_type: str
    match_method: str
    aracyc_compound_id: str
    source_db: str
    pathway_id: str
    pathway_name: str
    pathway_category: str
    gene_count: int
    ec_numbers: str
    annotation_confidence: str
    reason: str


@dataclass(slots=True)
class ExpandedPathway:
    """One ML-predicted expanded pathway."""

    compound_id: str
    chebi_name: str
    pathway_name: str
    pathway_source: str
    ml_score: float
    ml_confidence: str
    reason: str


# ---------------------------------------------------------------------------
# Index loading
# ---------------------------------------------------------------------------


def load_name_to_aracyc(path: Path) -> dict[str, list[AraCycMappingRecord]]:
    """Load name_to_aracyc_index.tsv into a lookup by chebi_name variants."""
    by_name: dict[str, list[AraCycMappingRecord]] = {}
    by_compound_id: dict[str, list[AraCycMappingRecord]] = {}

    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            record = AraCycMappingRecord(
                compound_id=row["compound_id"],
                chebi_accession=row["chebi_accession"],
                chebi_name=row["chebi_name"],
                aracyc_compound_id=row["aracyc_compound_id"],
                aracyc_common_name=row["aracyc_common_name"],
                source_db=row["source_db"],
                match_method=row["match_method"],
                match_score=float(row["match_score"]),
                chebi_xref_direct=row.get("chebi_xref_direct", "") == "true",
                structure_validated=row.get("structure_validated", "") == "true",
                pathway_count=int(row.get("pathway_count", 0)),
                pathways=row.get("pathways", ""),
            )
            by_compound_id.setdefault(row["compound_id"], []).append(record)

            # Index by normalized name variants
            for name in (row["chebi_name"], row["aracyc_common_name"]):
                variants = build_variants(name)
                for vtype, vtext in variants.items():
                    if vtext:
                        by_name.setdefault(vtext, []).append(record)

    return by_name, by_compound_id


def load_pathway_output(path: Path) -> dict[str, list[AraCycRankedPathway]]:
    """Load the ranked pathway output TSV."""
    result: dict[str, list[AraCycRankedPathway]] = {}

    if not path.exists():
        return result

    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            pathway = AraCycRankedPathway(
                compound_id=row["compound_id"],
                chebi_accession=row["chebi_accession"],
                chebi_name=row["chebi_name"],
                pathway_rank=int(row["pathway_rank"]),
                score=float(row["score"]),
                confidence_level=row["confidence_level"],
                evidence_type=row["evidence_type"],
                match_method=row["match_method"],
                aracyc_compound_id=row["aracyc_compound_id"],
                source_db=row["source_db"],
                pathway_id=row["pathway_id"],
                pathway_name=row["pathway_name"],
                pathway_category=row["pathway_category"],
                gene_count=int(row.get("gene_count", 0)),
                ec_numbers=row.get("ec_numbers", ""),
                annotation_confidence=row.get("annotation_confidence", ""),
                reason=row.get("reason", ""),
            )
            result.setdefault(row["compound_id"], []).append(pathway)

    return result


def load_expanded_predictions(path: Path) -> dict[str, list[ExpandedPathway]]:
    """Load ML-predicted expanded pathways."""
    result: dict[str, list[ExpandedPathway]] = {}

    if not path.exists():
        return result

    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            ep = ExpandedPathway(
                compound_id=row["compound_id"],
                chebi_name=row.get("chebi_name", ""),
                pathway_name=row["pathway_name"],
                pathway_source=row.get("pathway_source", ""),
                ml_score=float(row.get("ml_score", 0)),
                ml_confidence=row.get("ml_confidence", ""),
                reason=row.get("reason", ""),
            )
            result.setdefault(row["compound_id"], []).append(ep)

    return result


# ---------------------------------------------------------------------------
# Query logic
# ---------------------------------------------------------------------------


def resolve_query(
    query: str,
    name_index: dict[str, list[AraCycMappingRecord]],
) -> list[AraCycMappingRecord]:
    """Resolve a query string to AraCyc mapping records."""
    variants = build_variants(query)
    seen_compound_ids: set[str] = set()
    results: list[AraCycMappingRecord] = []

    for vtype in ("exact", "compact", "singular", "stereo_stripped"):
        vtext = variants.get(vtype, "")
        if not vtext:
            continue
        for record in name_index.get(vtext, []):
            if record.compound_id not in seen_compound_ids:
                seen_compound_ids.add(record.compound_id)
                results.append(record)

    # Sort by match score descending
    results.sort(key=lambda r: r.match_score, reverse=True)
    return results


def run_query(
    query: str,
    state: dict,
    top_k: int = 10,
) -> tuple[list[AraCycMappingRecord], list[AraCycRankedPathway], list[ExpandedPathway]]:
    """Run a full query: resolve name → find mappings → rank pathways."""

    name_index = state["name_index"]
    pathway_index = state["pathway_index"]
    expanded_index = state["expanded_index"]

    # Resolve query to AraCyc mappings
    mappings = resolve_query(query, name_index)

    # Collect pathways for all matched compounds
    pathways: list[AraCycRankedPathway] = []
    expanded: list[ExpandedPathway] = []

    for mapping in mappings[:5]:  # limit to top 5 compound matches
        compound_pathways = pathway_index.get(mapping.compound_id, [])
        pathways.extend(compound_pathways)

        compound_expanded = expanded_index.get(mapping.compound_id, [])
        expanded.extend(compound_expanded)

    # Sort pathways by score, take top_k
    pathways.sort(key=lambda p: p.score, reverse=True)
    pathways = pathways[:top_k]

    # Sort expanded by ml_score
    expanded.sort(key=lambda e: e.ml_score, reverse=True)
    expanded = expanded[:top_k]

    return mappings, pathways, expanded


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------


def print_result(
    query: str,
    mappings: list[AraCycMappingRecord],
    pathways: list[AraCycRankedPathway],
    expanded: list[ExpandedPathway],
    top_k: int = 10,
) -> None:
    """Pretty-print query results."""

    print(f"\n{'='*70}")
    print(f"Query: {query}")
    print(f"{'='*70}")

    if not mappings:
        print("  No AraCyc/PlantCyc mapping found.")
        return

    print(f"\nAraCyc/PlantCyc Mappings ({len(mappings)} found):")
    for i, m in enumerate(mappings[:5], 1):
        xref = " [ChEBI xref]" if m.chebi_xref_direct else ""
        struct = " [structure]" if m.structure_validated else ""
        print(f"  {i}. CHEBI:{m.compound_id} ({m.chebi_name})")
        print(f"     → {m.aracyc_compound_id} ({m.aracyc_common_name}) [{m.source_db}]")
        print(f"     method={m.match_method}, score={m.match_score:.3f}{xref}{struct}")
        print(f"     pathways: {m.pathway_count}")

    if pathways:
        print(f"\nRanked Pathways (top {min(top_k, len(pathways))}):")
        for pw in pathways[:top_k]:
            genes = f", genes={pw.gene_count}" if pw.gene_count else ""
            ec = f", EC={pw.ec_numbers}" if pw.ec_numbers else ""
            print(f"  #{pw.pathway_rank} [{pw.confidence_level}] {pw.pathway_name}")
            print(f"     score={pw.score:.3f}, category={pw.pathway_category}{genes}{ec}")
            print(f"     via {pw.aracyc_compound_id} [{pw.source_db}, {pw.match_method}]")
    else:
        print("\n  No pathways found in primary chain.")

    if expanded:
        print(f"\nExpanded Pathways [experimental] ({len(expanded)}):")
        for ep in expanded[:5]:
            print(f"  - {ep.pathway_name} (ml_score={ep.ml_score:.3f}, {ep.ml_confidence})")


# ---------------------------------------------------------------------------
# State loading
# ---------------------------------------------------------------------------


def load_preprocessed_state(workdir: Path, verbose: bool = True) -> dict:
    """Load all preprocessed indexes for query-time use."""

    preprocessed_dir = workdir / "outputs" / "preprocessed"
    outputs_dir = workdir / "outputs"

    name_to_aracyc_path = preprocessed_dir / "name_to_aracyc_index.tsv"
    pathway_output_path = outputs_dir / "chebi_pathways_aracyc_refactored.tsv"
    expanded_path = preprocessed_dir / "ml_pathway_predictions.tsv"

    if verbose:
        print(f"Loading AraCyc indexes from {preprocessed_dir}...", flush=True)

    if not name_to_aracyc_path.exists():
        print(f"  Warning: {name_to_aracyc_path} not found. Run preprocess_all.py first.", file=sys.stderr)
        return {"name_index": {}, "compound_index": {}, "pathway_index": {}, "expanded_index": {}}

    name_index, compound_index = load_name_to_aracyc(name_to_aracyc_path)
    pathway_index = load_pathway_output(pathway_output_path)
    expanded_index = load_expanded_predictions(expanded_path)

    if verbose:
        print(f"  Name index: {len(name_index)} entries", flush=True)
        print(f"  Compound index: {len(compound_index)} compounds", flush=True)
        print(f"  Pathway index: {len(pathway_index)} compounds with pathways", flush=True)
        print(f"  Expanded index: {len(expanded_index)} compounds", flush=True)

    return {
        "name_index": name_index,
        "compound_index": compound_index,
        "pathway_index": pathway_index,
        "expanded_index": expanded_index,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def repl(state: dict, top_k: int) -> None:
    """Interactive REPL for pathway queries."""

    print("\nAraCyc-first pathway query REPL. Type a compound name or 'quit' to exit.")
    while True:
        try:
            query = input("\nquery> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not query or query.lower() in ("quit", "exit", "q"):
            break
        mappings, pathways, expanded = run_query(query, state, top_k)
        print_result(query, mappings, pathways, expanded, top_k)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Query AraCyc-first pathway mappings.")
    parser.add_argument("--workdir", default=".", help="Workspace root directory.")
    parser.add_argument("--query", "-q", default="", help="Single query (non-interactive).")
    parser.add_argument("--top-k", type=int, default=10, help="Max pathways to show.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    workdir = Path(args.workdir).resolve()
    state = load_preprocessed_state(workdir)

    if args.query:
        mappings, pathways, expanded = run_query(args.query, state, args.top_k)
        print_result(args.query, mappings, pathways, expanded, args.top_k)
    else:
        repl(state, args.top_k)


if __name__ == "__main__":
    main()
