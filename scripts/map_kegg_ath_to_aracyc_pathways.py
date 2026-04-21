#!/usr/bin/env python3
"""Build a scored KEGG ath pathway -> AraCyc pathway alignment table.

The goal here is not to declare strict biological equivalence. Instead, the
script builds a practical alignment table that can be used for:

- display-time support annotations
- cross-database sanity checks
- future external validation analyses

Each edge is scored from pathway name similarity, Arabidopsis gene overlap, and
compound overlap after mapping KEGG compounds into the project's ChEBI-centric
compound key space.
"""

from __future__ import annotations

import argparse
import csv
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from pathlib import Path


DEFAULT_TOP_K = 5


@dataclass(slots=True)
class AlignmentRow:
    ath_pathway_id: str
    ath_pathway_name: str
    rank: int
    aracyc_pathway_id: str
    aracyc_pathway_name: str
    alignment_score: float
    alignment_confidence: str
    name_exact: int
    name_similarity: float
    shared_gene_count: int
    ath_gene_count: int
    aracyc_gene_count: int
    gene_jaccard: float
    gene_overlap_coefficient: float
    shared_compound_count: int
    ath_compound_count: int
    aracyc_compound_count: int
    compound_jaccard: float
    compound_overlap_coefficient: float


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for KEGG-ath to AraCyc alignment generation."""

    parser = argparse.ArgumentParser(
        description="Build a scored alignment between KEGG ath pathways and AraCyc pathways."
    )
    parser.add_argument("--workdir", default=".", help="Workspace root directory.")
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help="Number of AraCyc candidates to keep per KEGG ath pathway.",
    )
    return parser.parse_args()


def _latest_versioned_ref(refs: Path, prefix: str) -> Path:
    candidates = sorted(refs.glob(f"{prefix}.*"))
    if not candidates:
        raise FileNotFoundError(f"No versioned ref found for {prefix} in {refs}")
    return candidates[-1]


def _normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text.strip().lower())
    normalized = normalized.encode("ascii", "ignore").decode("ascii")
    normalized = re.sub(r"[^a-z0-9]+", " ", normalized)
    return " ".join(normalized.split())


def _name_similarity(left: str, right: str) -> float:
    left_norm = _normalize_text(left)
    right_norm = _normalize_text(right)
    if not left_norm or not right_norm:
        return 0.0
    return SequenceMatcher(None, left_norm, right_norm).ratio()


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 0.0
    union = left | right
    if not union:
        return 0.0
    return len(left & right) / len(union)


def _overlap_coefficient(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    denominator = min(len(left), len(right))
    if denominator == 0:
        return 0.0
    return len(left & right) / denominator


def _confidence_level(
    *,
    name_exact: int,
    alignment_score: float,
    shared_gene_count: int,
    shared_compound_count: int,
) -> str:
    if name_exact == 1 or (alignment_score >= 6.0 and shared_gene_count > 0):
        return "high"
    if alignment_score >= 4.0 and (shared_gene_count > 0 or shared_compound_count > 0):
        return "medium"
    return "low"


def _load_kegg_ath_pathways(path: Path) -> tuple[dict[str, str], dict[str, str]]:
    ath_pathways: dict[str, str] = {}
    map_to_ath: dict[str, str] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            pathway_id, raw_name = line.rstrip("\n").split("\t", 1)
            pathway_name = raw_name.split(" - ", 1)[0]
            ath_pathways[pathway_id] = pathway_name
            map_to_ath[f"map{pathway_id[3:]}"] = pathway_id
    return ath_pathways, map_to_ath


def _load_kegg_ath_gene_sets(path: Path) -> dict[str, set[str]]:
    gene_sets: dict[str, set[str]] = defaultdict(set)
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            gene_ref, pathway_ref = line.rstrip("\n").split("\t", 1)
            pathway_id = pathway_ref.replace("path:", "")
            gene_sets[pathway_id].add(gene_ref.replace("ath:", "").upper())
    return {pathway_id: set(genes) for pathway_id, genes in gene_sets.items()}


def _load_aracyc_pathway_metadata(path: Path) -> tuple[dict[str, str], dict[str, set[str]]]:
    pathway_names: dict[str, str] = {}
    pathway_gene_sets: dict[str, set[str]] = defaultdict(set)
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            pathway_id = (row.get("Pathway-id") or "").strip()
            pathway_name = (row.get("Pathway-name") or "").strip()
            gene_id = (row.get("Gene-id") or "").strip().upper()
            if not pathway_id or not pathway_name:
                continue
            pathway_names[pathway_id] = pathway_name
            if gene_id:
                pathway_gene_sets[pathway_id].add(gene_id)
    return pathway_names, {pathway_id: set(genes) for pathway_id, genes in pathway_gene_sets.items()}


def _load_kegg_to_compound_ids(path: Path) -> dict[str, set[str]]:
    mapping: dict[str, set[str]] = defaultdict(set)
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            compound_id = (row.get("compound_id") or "").strip()
            if not compound_id:
                continue
            for kegg_id in (row.get("kegg_ids") or "").split(";"):
                kegg_id = kegg_id.strip()
                if kegg_id:
                    mapping[kegg_id].add(compound_id)
    return {kegg_id: set(compound_ids) for kegg_id, compound_ids in mapping.items()}


def _load_kegg_ath_compound_sets(
    pathway_links_path: Path,
    map_to_ath: dict[str, str],
    kegg_to_compound_ids: dict[str, set[str]],
) -> dict[str, set[str]]:
    pathway_compounds: dict[str, set[str]] = defaultdict(set)
    with pathway_links_path.open(encoding="utf-8") as handle:
        for line in handle:
            compound_ref, pathway_ref = line.rstrip("\n").split("\t", 1)
            map_pathway_id = pathway_ref.replace("path:", "")
            ath_pathway_id = map_to_ath.get(map_pathway_id)
            if not ath_pathway_id:
                continue
            kegg_compound_id = compound_ref.replace("cpd:", "")
            pathway_compounds[ath_pathway_id].update(kegg_to_compound_ids.get(kegg_compound_id, ()))
    return {pathway_id: set(compound_ids) for pathway_id, compound_ids in pathway_compounds.items()}


def _load_aracyc_compound_sets(path: Path) -> dict[str, set[str]]:
    pathway_compounds: dict[str, set[str]] = defaultdict(set)
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            if (row.get("source_db") or "").strip() != "AraCyc":
                continue
            pathway_id = (row.get("pathway_id") or "").strip()
            compound_id = (row.get("compound_id") or "").strip()
            if pathway_id and compound_id:
                pathway_compounds[pathway_id].add(compound_id)
    return {pathway_id: set(compound_ids) for pathway_id, compound_ids in pathway_compounds.items()}


def _score_alignment(
    *,
    ath_pathway_id: str,
    ath_pathway_name: str,
    aracyc_pathway_id: str,
    aracyc_pathway_name: str,
    ath_gene_sets: dict[str, set[str]],
    aracyc_gene_sets: dict[str, set[str]],
    ath_compound_sets: dict[str, set[str]],
    aracyc_compound_sets: dict[str, set[str]],
) -> AlignmentRow:
    """Score one KEGG-ath / AraCyc pathway pair using fixed evidence weights."""

    ath_genes = ath_gene_sets.get(ath_pathway_id, set())
    aracyc_genes = aracyc_gene_sets.get(aracyc_pathway_id, set())
    ath_compounds = ath_compound_sets.get(ath_pathway_id, set())
    aracyc_compounds = aracyc_compound_sets.get(aracyc_pathway_id, set())

    name_exact = int(_normalize_text(ath_pathway_name) == _normalize_text(aracyc_pathway_name))
    name_similarity = _name_similarity(ath_pathway_name, aracyc_pathway_name)

    shared_genes = len(ath_genes & aracyc_genes)
    gene_jaccard = _jaccard(ath_genes, aracyc_genes)
    gene_overlap_coefficient = _overlap_coefficient(ath_genes, aracyc_genes)

    shared_compounds = len(ath_compounds & aracyc_compounds)
    compound_jaccard = _jaccard(ath_compounds, aracyc_compounds)
    compound_overlap_coefficient = _overlap_coefficient(ath_compounds, aracyc_compounds)

    alignment_score = (
        6.0 * name_exact
        + 2.0 * name_similarity
        + 4.0 * gene_jaccard
        + 3.0 * gene_overlap_coefficient
        + 4.0 * compound_jaccard
        + 3.0 * compound_overlap_coefficient
        + 1.0 * int(shared_genes > 0 and shared_compounds > 0)
    )

    return AlignmentRow(
        ath_pathway_id=ath_pathway_id,
        ath_pathway_name=ath_pathway_name,
        rank=0,
        aracyc_pathway_id=aracyc_pathway_id,
        aracyc_pathway_name=aracyc_pathway_name,
        alignment_score=alignment_score,
        alignment_confidence=_confidence_level(
            name_exact=name_exact,
            alignment_score=alignment_score,
            shared_gene_count=shared_genes,
            shared_compound_count=shared_compounds,
        ),
        name_exact=name_exact,
        name_similarity=name_similarity,
        shared_gene_count=shared_genes,
        ath_gene_count=len(ath_genes),
        aracyc_gene_count=len(aracyc_genes),
        gene_jaccard=gene_jaccard,
        gene_overlap_coefficient=gene_overlap_coefficient,
        shared_compound_count=shared_compounds,
        ath_compound_count=len(ath_compounds),
        aracyc_compound_count=len(aracyc_compounds),
        compound_jaccard=compound_jaccard,
        compound_overlap_coefficient=compound_overlap_coefficient,
    )


def _format_row(row: AlignmentRow) -> dict[str, str]:
    payload = asdict(row)
    return {
        key: f"{value:.6f}"
        if isinstance(value, float)
        else str(value)
        for key, value in payload.items()
    }


def _write_rows(path: Path, rows: list[AlignmentRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(asdict(rows[0]).keys()) if rows else list(AlignmentRow.__dataclass_fields__.keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow(_format_row(row))


def main() -> None:
    """Generate and write the full alignment table plus one best match per KEGG pathway."""

    args = parse_args()
    workdir = Path(args.workdir).resolve()
    refs = workdir / "refs"
    preprocessed = workdir / "outputs" / "preprocessed"
    top_k = max(1, int(args.top_k))

    kegg_ath_path = refs / "kegg_pathway_ath.tsv"
    kegg_ath_gene_path = refs / "kegg_ath_gene_pathway.tsv"
    kegg_compound_pathway_path = refs / "kegg_compound_pathway.tsv"
    aracyc_pathways_path = _latest_versioned_ref(refs, "aracyc_pathways")
    step1_result_path = preprocessed / "step1_result.tsv"
    aracyc_compound_pathway_index_path = preprocessed / "aracyc_compound_pathway_index.tsv"

    ath_pathways, map_to_ath = _load_kegg_ath_pathways(kegg_ath_path)
    ath_gene_sets = _load_kegg_ath_gene_sets(kegg_ath_gene_path)
    aracyc_pathway_names, aracyc_gene_sets = _load_aracyc_pathway_metadata(aracyc_pathways_path)
    kegg_to_compound_ids = _load_kegg_to_compound_ids(step1_result_path)
    ath_compound_sets = _load_kegg_ath_compound_sets(
        kegg_compound_pathway_path,
        map_to_ath,
        kegg_to_compound_ids,
    )
    aracyc_compound_sets = _load_aracyc_compound_sets(aracyc_compound_pathway_index_path)

    alignment_rows: list[AlignmentRow] = []
    best_rows: list[AlignmentRow] = []

    for ath_pathway_id, ath_pathway_name in sorted(ath_pathways.items()):
        ranked_rows = [
            _score_alignment(
                ath_pathway_id=ath_pathway_id,
                ath_pathway_name=ath_pathway_name,
                aracyc_pathway_id=aracyc_pathway_id,
                aracyc_pathway_name=aracyc_pathway_name,
                ath_gene_sets=ath_gene_sets,
                aracyc_gene_sets=aracyc_gene_sets,
                ath_compound_sets=ath_compound_sets,
                aracyc_compound_sets=aracyc_compound_sets,
            )
            for aracyc_pathway_id, aracyc_pathway_name in aracyc_pathway_names.items()
        ]
        ranked_rows.sort(
            key=lambda row: (
                row.alignment_score,
                row.name_exact,
                row.shared_gene_count,
                row.shared_compound_count,
                row.name_similarity,
                -row.aracyc_gene_count,
                row.aracyc_pathway_id,
            ),
            reverse=True,
        )
        top_rows = ranked_rows[:top_k]
        for rank, row in enumerate(top_rows, start=1):
            row.rank = rank
        alignment_rows.extend(top_rows)
        best_rows.append(top_rows[0])

    alignment_path = preprocessed / "kegg_ath_to_aracyc_pathway_alignment.tsv"
    best_path = preprocessed / "kegg_ath_to_aracyc_pathway_best.tsv"
    _write_rows(alignment_path, alignment_rows)
    _write_rows(best_path, best_rows)

    confidence_counts = Counter(row.alignment_confidence for row in best_rows)
    exact_name_best = sum(1 for row in best_rows if row.name_exact == 1)
    gene_evidence_best = sum(1 for row in best_rows if row.shared_gene_count > 0)
    compound_evidence_best = sum(1 for row in best_rows if row.shared_compound_count > 0)

    print(f"KEGG ath pathways: {len(ath_pathways)}", flush=True)
    print(f"Exact-name best matches: {exact_name_best}", flush=True)
    print(f"Best matches with gene evidence: {gene_evidence_best}", flush=True)
    print(f"Best matches with compound evidence: {compound_evidence_best}", flush=True)
    print(
        "Confidence levels: "
        f"high={confidence_counts.get('high', 0)}, "
        f"medium={confidence_counts.get('medium', 0)}, "
        f"low={confidence_counts.get('low', 0)}",
        flush=True,
    )
    print(f"Alignment table written: {alignment_path}", flush=True)
    print(f"Best-match table written: {best_path}", flush=True)


if __name__ == "__main__":
    main()
