#!/usr/bin/env python3
"""Interactive/file-driven AraCyc pathway enrichment for a compound set."""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from compound_set_rerank_shared import (
    ONLINE_CANDIDATE_POLICY,
    load_online_model_bundle,
    load_parent_map,
    predict_rerank_scores,
    query_level_features_from_compound_ids,
    rerank_feature_values,
)
from pathway_query_shared import (
    KeggPathwaySupport,
    ensure_chebi_structure_lookup,
    load_preprocessed_state,
    resolve_primary_compound,
)


DONE_SENTINEL = "DONE"
SMALL_U_WARNING = 10
SMALL_S_WARNING = 3


@dataclass(slots=True)
class InputMetaboliteRow:
    input_order: int
    input_name: str
    resolution_status: str
    compound_id: str
    chebi_accession: str
    chebi_name: str
    match_type: str
    match_score: float
    resolution_path: str
    note: str
    did_you_mean: str
    included_in_target: bool


@dataclass(slots=True)
class CompoundTargetRecord:
    compound_id: str
    chebi_name: str
    match_type: str
    match_score: float


@dataclass(slots=True)
class PathwayEnrichmentRow:
    pathway_id: str
    pathway_name: str
    x_i: int
    K_i: int
    n: int
    N: int
    target_rate: float
    background_rate: float
    expected_hits: float
    excess_hits: float
    enrichment_factor: float
    coverage: float
    p_value: float
    fdr: float
    neg_log10_p: float
    neg_log10_fdr: float
    mapping_confidence_mean: float
    mapping_confidence_min: float
    mapping_confidence_max: float
    direct_hit_count: int
    fuzzy_hit_count: int
    recovered_hit_count: int
    exact_structure_hit_count: int
    structural_neighbor_hit_count: int
    final_score: float
    confidence_level: str
    hit_compound_ids: tuple[str, ...]
    hit_compound_names: tuple[str, ...]
    kegg_support_present: bool = False
    kegg_ath_pathway_id: str = ""
    kegg_ath_pathway_name: str = ""
    kegg_alignment_score: float = 0.0
    kegg_alignment_confidence: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run AraCyc-only pathway enrichment for a set of compounds.")
    parser.add_argument("--workdir", default=".", help="Workspace root directory.")
    parser.add_argument("--out-prefix", default="", help="Optional output prefix.")
    parser.add_argument("--input", default="", help="Optional text file with one compound name per line.")
    parser.add_argument("--no-fuzzy", action="store_true", help="Disable typo correction during resolution.")
    return parser.parse_args()


def _load_input_names(input_path: Path | None) -> list[str]:
    names: list[str] = []
    if input_path is not None:
        with input_path.open(encoding="utf-8") as handle:
            for line in handle:
                text = line.strip()
                if text:
                    names.append(text)
        return names

    print(
        "Enter one compound name per line. Type DONE on its own line to run enrichment.",
        flush=True,
    )
    while True:
        try:
            text = input("> ").strip()
        except EOFError:
            if names:
                break
            return []
        if not text:
            continue
        if text.upper() == DONE_SENTINEL:
            break
        names.append(text)
    return names


def _write_input_names(path: Path, names: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for name in names:
            handle.write(f"{name}\n")


def _load_pathway_membership(path: Path) -> tuple[dict[str, str], dict[str, set[str]], set[str]]:
    pathway_names: dict[str, str] = {}
    pathway_to_tokens: dict[str, set[str]] = {}
    universe_tokens: set[str] = set()
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            if row.get("source_db") != "AraCyc":
                continue
            compound_id = (row.get("compound_id") or "").strip()
            pathway_id = (row.get("pathway_id") or "").strip()
            pathway_name = (row.get("pathway_name") or "").strip()
            if not compound_id or not pathway_id:
                continue
            token = f"cmp:{compound_id}"
            universe_tokens.add(token)
            pathway_names[pathway_id] = pathway_name or pathway_id
            pathway_to_tokens.setdefault(pathway_id, set()).add(token)
    return pathway_names, pathway_to_tokens, universe_tokens


def _log_choose(n: int, k: int) -> float:
    if k < 0 or k > n:
        return float("-inf")
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


def _hypergeom_p_value(N: int, K: int, n: int, x: int) -> float:
    if x <= 0:
        return 1.0
    upper = min(K, n)
    if x > upper:
        return 0.0
    denominator = _log_choose(N, n)
    log_terms: list[float] = []
    for k in range(x, upper + 1):
        log_terms.append(_log_choose(K, k) + _log_choose(N - K, n - k) - denominator)
    max_log = max(log_terms)
    total = sum(math.exp(term - max_log) for term in log_terms)
    return min(1.0, math.exp(max_log) * total)


def _benjamini_hochberg(p_values: list[float]) -> list[float]:
    if not p_values:
        return []
    m = len(p_values)
    order = sorted(range(m), key=lambda idx: p_values[idx])
    adjusted = [1.0] * m
    running = 1.0
    for rank, idx in enumerate(reversed(order), start=1):
        orig_rank = m - rank + 1
        running = min(running, p_values[idx] * m / orig_rank)
        adjusted[idx] = min(running, 1.0)
    return adjusted


def _normalized_ef(values: dict[str, float]) -> dict[str, float]:
    if not values:
        return {}
    min_value = min(values.values())
    max_value = max(values.values())
    if len(values) == 1 or math.isclose(min_value, max_value):
        return {key: 1.0 for key in values}
    normalized = {}
    for key, value in values.items():
        scaled = (value - min_value) / (max_value - min_value)
        normalized[key] = min(1.0, max(0.0, scaled))
    return normalized


def _confidence_level(x_i: int, fdr: float) -> str:
    if x_i >= 3 and fdr < 0.01:
        return "high"
    if x_i >= 2 and fdr < 0.05:
        return "medium"
    return "low"


def _match_type_priority(match_type: str) -> int:
    order = {
        "direct": 5,
        "exact_structure": 4,
        "fuzzy": 3,
        "recovered_from_chebi": 2,
        "structural_neighbor": 1,
    }
    return order.get(match_type, 0)


def _resolve_input_rows(
    input_names: list[str],
    *,
    state: dict[str, Any],
    universe_tokens: set[str],
    no_fuzzy: bool,
) -> list[InputMetaboliteRow]:
    rows: list[InputMetaboliteRow] = []
    for input_order, raw_name in enumerate(input_names, start=1):
        resolution = resolve_primary_compound(raw_name, state, no_fuzzy=no_fuzzy)
        compound_token = f"cmp:{resolution.compound_id}" if resolution.compound_id else ""
        included_in_target = resolution.status == "resolved" and compound_token in universe_tokens
        note = resolution.note
        if resolution.status == "resolved" and not included_in_target:
            extra = "Resolved compound is not present in the AraCyc pathway membership universe."
            note = f"{note} {extra}".strip()

        rows.append(
            InputMetaboliteRow(
                input_order=input_order,
                input_name=raw_name,
                resolution_status=resolution.status,
                compound_id=resolution.compound_id,
                chebi_accession=resolution.chebi_accession,
                chebi_name=resolution.chebi_name,
                match_type=resolution.match_type,
                match_score=resolution.match_score,
                resolution_path=resolution.resolution_path,
                note=note,
                did_you_mean="; ".join(resolution.did_you_mean),
                included_in_target=included_in_target,
            )
        )
    return rows


def _build_target_compounds(rows: list[InputMetaboliteRow]) -> tuple[set[str], dict[str, CompoundTargetRecord]]:
    target_tokens: set[str] = set()
    compound_records: dict[str, CompoundTargetRecord] = {}
    for row in rows:
        if not row.included_in_target or not row.compound_id:
            continue
        token = f"cmp:{row.compound_id}"
        target_tokens.add(token)
        current = compound_records.get(row.compound_id)
        candidate = CompoundTargetRecord(
            compound_id=row.compound_id,
            chebi_name=row.chebi_name,
            match_type=row.match_type,
            match_score=row.match_score,
        )
        if current is None:
            compound_records[row.compound_id] = candidate
            continue
        if candidate.match_score > current.match_score:
            compound_records[row.compound_id] = candidate
            continue
        if math.isclose(candidate.match_score, current.match_score) and (
            _match_type_priority(candidate.match_type) > _match_type_priority(current.match_type)
        ):
            compound_records[row.compound_id] = candidate
    return target_tokens, compound_records


def _compute_enrichment(
    pathway_names: dict[str, str],
    pathway_to_tokens: dict[str, set[str]],
    universe_tokens: set[str],
    target_tokens: set[str],
    compound_records: dict[str, CompoundTargetRecord],
    kegg_support_lookup: dict[str, KeggPathwaySupport] | None = None,
) -> list[PathwayEnrichmentRow]:
    N = len(universe_tokens)
    n = len(target_tokens)
    raw_rows: list[dict[str, Any]] = []

    for pathway_id, member_tokens in pathway_to_tokens.items():
        K_i = len(member_tokens)
        x_tokens = member_tokens & target_tokens
        x_i = len(x_tokens)
        if x_i < 2 or K_i == 0 or N == 0 or n == 0:
            continue

        target_rate = x_i / n
        background_rate = K_i / N
        expected_hits = n * K_i / N
        excess_hits = x_i - expected_hits
        enrichment_factor = (target_rate / background_rate) if background_rate > 0 else 0.0
        coverage = x_i / K_i
        p_value = _hypergeom_p_value(N, K_i, n, x_i)

        hit_compound_ids = tuple(sorted(token.split(":", 1)[1] for token in x_tokens if token.startswith("cmp:")))
        hit_records = [compound_records[cid] for cid in hit_compound_ids if cid in compound_records]
        hit_scores = [record.match_score for record in hit_records]
        mapping_confidence_mean = sum(hit_scores) / len(hit_scores) if hit_scores else 0.0
        mapping_confidence_min = min(hit_scores) if hit_scores else 0.0
        mapping_confidence_max = max(hit_scores) if hit_scores else 0.0

        direct_hit_count = sum(1 for record in hit_records if record.match_type == "direct")
        fuzzy_hit_count = sum(1 for record in hit_records if record.match_type == "fuzzy")
        recovered_hit_count = sum(1 for record in hit_records if record.match_type == "recovered_from_chebi")
        exact_structure_hit_count = sum(1 for record in hit_records if record.match_type == "exact_structure")
        structural_neighbor_hit_count = sum(1 for record in hit_records if record.match_type == "structural_neighbor")

        raw_rows.append(
            {
                "pathway_id": pathway_id,
                "pathway_name": pathway_names.get(pathway_id, pathway_id),
                "x_i": x_i,
                "K_i": K_i,
                "n": n,
                "N": N,
                "target_rate": target_rate,
                "background_rate": background_rate,
                "expected_hits": expected_hits,
                "excess_hits": excess_hits,
                "enrichment_factor": enrichment_factor,
                "coverage": coverage,
                "p_value": p_value,
                "hit_compound_ids": hit_compound_ids,
                "hit_compound_names": tuple(record.chebi_name or f"CHEBI:{record.compound_id}" for record in hit_records),
                "mapping_confidence_mean": mapping_confidence_mean,
                "mapping_confidence_min": mapping_confidence_min,
                "mapping_confidence_max": mapping_confidence_max,
                "direct_hit_count": direct_hit_count,
                "fuzzy_hit_count": fuzzy_hit_count,
                "recovered_hit_count": recovered_hit_count,
                "exact_structure_hit_count": exact_structure_hit_count,
                "structural_neighbor_hit_count": structural_neighbor_hit_count,
            }
        )

    adjusted = _benjamini_hochberg([row["p_value"] for row in raw_rows])
    for row, fdr in zip(raw_rows, adjusted, strict=False):
        row["fdr"] = fdr
        row["neg_log10_p"] = -math.log10(max(row["p_value"], 1e-300))
        row["neg_log10_fdr"] = -math.log10(max(fdr, 1e-300))

    ef_logs = {row["pathway_id"]: math.log2(max(row["enrichment_factor"], 1.0)) for row in raw_rows}
    ef_norm = _normalized_ef(ef_logs)

    support_lookup = kegg_support_lookup or {}
    results: list[PathwayEnrichmentRow] = []
    for row in raw_rows:
        final_score = (
            0.55 * ef_norm.get(row["pathway_id"], 0.0)
            + 0.30 * row["coverage"]
            + 0.15 * row["mapping_confidence_mean"]
        )
        kegg_support = support_lookup.get(row["pathway_id"])
        results.append(
            PathwayEnrichmentRow(
                pathway_id=row["pathway_id"],
                pathway_name=row["pathway_name"],
                x_i=row["x_i"],
                K_i=row["K_i"],
                n=row["n"],
                N=row["N"],
                target_rate=row["target_rate"],
                background_rate=row["background_rate"],
                expected_hits=row["expected_hits"],
                excess_hits=row["excess_hits"],
                enrichment_factor=row["enrichment_factor"],
                coverage=row["coverage"],
                p_value=row["p_value"],
                fdr=row["fdr"],
                neg_log10_p=row["neg_log10_p"],
                neg_log10_fdr=row["neg_log10_fdr"],
                mapping_confidence_mean=row["mapping_confidence_mean"],
                mapping_confidence_min=row["mapping_confidence_min"],
                mapping_confidence_max=row["mapping_confidence_max"],
                direct_hit_count=row["direct_hit_count"],
                fuzzy_hit_count=row["fuzzy_hit_count"],
                recovered_hit_count=row["recovered_hit_count"],
                exact_structure_hit_count=row["exact_structure_hit_count"],
                structural_neighbor_hit_count=row["structural_neighbor_hit_count"],
                final_score=round(final_score, 4),
                confidence_level=_confidence_level(row["x_i"], row["fdr"]),
                hit_compound_ids=row["hit_compound_ids"],
                hit_compound_names=row["hit_compound_names"],
                kegg_support_present=kegg_support is not None,
                kegg_ath_pathway_id=kegg_support.ath_pathway_id if kegg_support is not None else "",
                kegg_ath_pathway_name=kegg_support.ath_pathway_name if kegg_support is not None else "",
                kegg_alignment_score=kegg_support.alignment_score if kegg_support is not None else 0.0,
                kegg_alignment_confidence=kegg_support.alignment_confidence if kegg_support is not None else "",
            )
        )

    results.sort(
        key=lambda row: (
            {"high": 2, "medium": 1, "low": 0}[row.confidence_level],
            row.final_score,
            row.neg_log10_fdr,
            row.x_i,
        ),
        reverse=True,
    )
    return results


def _write_resolution_output(path: Path, rows: list[InputMetaboliteRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "input_order",
                "input_name",
                "resolution_status",
                "compound_id",
                "chebi_accession",
                "chebi_name",
                "match_type",
                "match_score",
                "resolution_path",
                "note",
                "did_you_mean",
                "included_in_target",
            ],
            delimiter="\t",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "input_order": row.input_order,
                    "input_name": row.input_name,
                    "resolution_status": row.resolution_status,
                    "compound_id": row.compound_id,
                    "chebi_accession": row.chebi_accession,
                    "chebi_name": row.chebi_name,
                    "match_type": row.match_type,
                    "match_score": f"{row.match_score:.4f}" if row.match_score else "",
                    "resolution_path": row.resolution_path,
                    "note": row.note,
                    "did_you_mean": row.did_you_mean,
                    "included_in_target": str(row.included_in_target).lower(),
                }
            )


def _enrichment_output_fieldnames() -> list[str]:
    return [
        "pathway_id",
        "pathway_name",
        "baseline_rank",
        "ml_score",
        "ml_rank",
        "rerank_applied",
        "rerank_reason",
        "model_kind",
        "x_i",
        "K_i",
        "n",
        "N",
        "target_rate",
        "background_rate",
        "expected_hits",
        "excess_hits",
        "enrichment_factor",
        "coverage",
        "p_value",
        "fdr",
        "neg_log10_p",
        "neg_log10_fdr",
        "mapping_confidence_mean",
        "mapping_confidence_min",
        "mapping_confidence_max",
        "direct_hit_count",
        "fuzzy_hit_count",
        "recovered_hit_count",
        "exact_structure_hit_count",
        "structural_neighbor_hit_count",
        "final_score",
        "confidence_level",
        "hit_compound_ids",
        "hit_compound_names",
        "kegg_support_present",
        "kegg_ath_pathway_id",
        "kegg_ath_pathway_name",
        "kegg_alignment_score",
        "kegg_alignment_confidence",
    ]


def _enrichment_output_row(
    row: PathwayEnrichmentRow,
    *,
    baseline_rank: int,
    ml_score: float | None,
    ml_rank: int | None,
    rerank_applied: bool,
    rerank_reason: str,
    model_kind: str,
) -> dict[str, Any]:
    return {
        "pathway_id": row.pathway_id,
        "pathway_name": row.pathway_name,
        "baseline_rank": baseline_rank,
        "ml_score": ml_score,
        "ml_rank": ml_rank,
        "rerank_applied": rerank_applied,
        "rerank_reason": rerank_reason,
        "model_kind": model_kind,
        "x_i": row.x_i,
        "K_i": row.K_i,
        "n": row.n,
        "N": row.N,
        "target_rate": row.target_rate,
        "background_rate": row.background_rate,
        "expected_hits": row.expected_hits,
        "excess_hits": row.excess_hits,
        "enrichment_factor": row.enrichment_factor,
        "coverage": row.coverage,
        "p_value": row.p_value,
        "fdr": row.fdr,
        "neg_log10_p": row.neg_log10_p,
        "neg_log10_fdr": row.neg_log10_fdr,
        "mapping_confidence_mean": row.mapping_confidence_mean,
        "mapping_confidence_min": row.mapping_confidence_min,
        "mapping_confidence_max": row.mapping_confidence_max,
        "direct_hit_count": row.direct_hit_count,
        "fuzzy_hit_count": row.fuzzy_hit_count,
        "recovered_hit_count": row.recovered_hit_count,
        "exact_structure_hit_count": row.exact_structure_hit_count,
        "structural_neighbor_hit_count": row.structural_neighbor_hit_count,
        "final_score": row.final_score,
        "confidence_level": row.confidence_level,
        "hit_compound_ids": ";".join(row.hit_compound_ids),
        "hit_compound_names": "; ".join(row.hit_compound_names),
        "kegg_support_present": row.kegg_support_present,
        "kegg_ath_pathway_id": row.kegg_ath_pathway_id,
        "kegg_ath_pathway_name": row.kegg_ath_pathway_name,
        "kegg_alignment_score": row.kegg_alignment_score,
        "kegg_alignment_confidence": row.kegg_alignment_confidence,
    }


def _write_enrichment_output(path: Path, rows: list[dict[str, Any]], warning: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        if warning:
            handle.write(f"# WARNING: {warning}\n")
        writer = csv.DictWriter(
            handle,
            fieldnames=_enrichment_output_fieldnames(),
            delimiter="\t",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "pathway_id": row["pathway_id"],
                    "pathway_name": row["pathway_name"],
                    "baseline_rank": row["baseline_rank"],
                    "ml_score": f"{float(row['ml_score']):.6g}" if row["ml_score"] is not None else "",
                    "ml_rank": row["ml_rank"] if row["ml_rank"] is not None else "",
                    "rerank_applied": str(bool(row["rerank_applied"])).lower(),
                    "rerank_reason": row["rerank_reason"],
                    "model_kind": row["model_kind"],
                    "x_i": row["x_i"],
                    "K_i": row["K_i"],
                    "n": row["n"],
                    "N": row["N"],
                    "target_rate": f"{float(row['target_rate']):.6g}",
                    "background_rate": f"{float(row['background_rate']):.6g}",
                    "expected_hits": f"{float(row['expected_hits']):.6g}",
                    "excess_hits": f"{float(row['excess_hits']):.6g}",
                    "enrichment_factor": f"{float(row['enrichment_factor']):.6g}",
                    "coverage": f"{float(row['coverage']):.6g}",
                    "p_value": f"{float(row['p_value']):.6g}",
                    "fdr": f"{float(row['fdr']):.6g}",
                    "neg_log10_p": f"{float(row['neg_log10_p']):.6g}",
                    "neg_log10_fdr": f"{float(row['neg_log10_fdr']):.6g}",
                    "mapping_confidence_mean": f"{float(row['mapping_confidence_mean']):.6g}",
                    "mapping_confidence_min": f"{float(row['mapping_confidence_min']):.6g}",
                    "mapping_confidence_max": f"{float(row['mapping_confidence_max']):.6g}",
                    "direct_hit_count": row["direct_hit_count"],
                    "fuzzy_hit_count": row["fuzzy_hit_count"],
                    "recovered_hit_count": row["recovered_hit_count"],
                    "exact_structure_hit_count": row["exact_structure_hit_count"],
                    "structural_neighbor_hit_count": row["structural_neighbor_hit_count"],
                    "final_score": f"{float(row['final_score']):.4f}",
                    "confidence_level": row["confidence_level"],
                    "hit_compound_ids": row["hit_compound_ids"],
                    "hit_compound_names": row["hit_compound_names"],
                    "kegg_support_present": str(bool(row["kegg_support_present"])).lower(),
                    "kegg_ath_pathway_id": row["kegg_ath_pathway_id"],
                    "kegg_ath_pathway_name": row["kegg_ath_pathway_name"],
                    "kegg_alignment_score": (
                        f"{float(row['kegg_alignment_score']):.6g}" if row["kegg_support_present"] else ""
                    ),
                    "kegg_alignment_confidence": row["kegg_alignment_confidence"],
                }
            )


def _default_out_prefix(workdir: Path, input_path: Path | None) -> Path:
    if input_path is not None:
        return workdir / "outputs" / "enrichment" / input_path.stem
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return workdir / "outputs" / "enrichment" / f"interactive_{stamp}"


def main() -> None:
    args = parse_args()
    workdir = Path(args.workdir).resolve()
    input_path = Path(args.input).resolve() if args.input else None
    out_prefix = Path(args.out_prefix).resolve() if args.out_prefix else _default_out_prefix(workdir, input_path)

    input_names = _load_input_names(input_path)
    if not input_names:
        raise SystemExit("No compound names provided.")

    state = load_preprocessed_state(workdir, verbose=True)
    kegg_support_lookup: dict[str, KeggPathwaySupport] = state.get("kegg_pathway_support", {})
    online_model_bundle, online_model_load_reason = load_online_model_bundle(workdir)
    pathway_index_path = workdir / "outputs" / "preprocessed" / "aracyc_compound_pathway_index.tsv"
    pathway_names, pathway_to_tokens, universe_tokens = _load_pathway_membership(pathway_index_path)

    rows = _resolve_input_rows(
        input_names,
        state=state,
        universe_tokens=universe_tokens,
        no_fuzzy=args.no_fuzzy,
    )
    target_tokens, compound_records = _build_target_compounds(rows)
    enrichment_rows = _compute_enrichment(
        pathway_names,
        pathway_to_tokens,
        universe_tokens,
        target_tokens,
        compound_records,
        kegg_support_lookup,
    )
    baseline_rank_by_id = {row.pathway_id: rank for rank, row in enumerate(enrichment_rows, start=1)}
    rerank_reason = "applied"
    rerank_applied = False
    output_rows: list[dict[str, Any]] = []
    online_model_kind = online_model_bundle["model_kind"] if online_model_bundle is not None else ""
    online_candidate_policy = (
        str(online_model_bundle["metadata"].get("online_candidate_policy") or ONLINE_CANDIDATE_POLICY)
        if online_model_bundle is not None
        else ONLINE_CANDIDATE_POLICY
    )

    if len(target_tokens) < 2:
        rerank_reason = "single_compound_query"
    elif not enrichment_rows:
        rerank_reason = "no_candidates"
    elif online_model_bundle is None:
        rerank_reason = online_model_load_reason
    else:
        structure_lookup = ensure_chebi_structure_lookup(state, verbose=False)
        parent_map = load_parent_map(workdir)
        root_cache: dict[str, str] = {}
        query_features = query_level_features_from_compound_ids(
            [record.compound_id for record in compound_records.values()],
            structure_lookup=structure_lookup,
            parent_map=parent_map,
            root_cache=root_cache,
        )
        feature_rows = [rerank_feature_values(row, query_features) for row in enrichment_rows]
        ml_scores = predict_rerank_scores(online_model_bundle, feature_rows)
        rerank_applied = True
        ranked_indices = sorted(
            range(len(enrichment_rows)),
            key=lambda idx: (float(ml_scores[idx]), float(enrichment_rows[idx].final_score)),
            reverse=True,
        )
        ml_rank_by_index = {idx: rank for rank, idx in enumerate(ranked_indices, start=1)}
        for idx, row in enumerate(enrichment_rows):
            output_rows.append(
                _enrichment_output_row(
                    row,
                    baseline_rank=baseline_rank_by_id[row.pathway_id],
                    ml_score=float(ml_scores[idx]),
                    ml_rank=ml_rank_by_index[idx],
                    rerank_applied=True,
                    rerank_reason="applied",
                    model_kind=online_model_kind,
                )
            )
        output_rows.sort(key=lambda row: (int(row["ml_rank"]), int(row["baseline_rank"])))

    if not output_rows:
        for row in enrichment_rows:
            output_rows.append(
                _enrichment_output_row(
                    row,
                    baseline_rank=baseline_rank_by_id[row.pathway_id],
                    ml_score=None,
                    ml_rank=None,
                    rerank_applied=False,
                    rerank_reason=rerank_reason,
                    model_kind=online_model_kind,
                )
            )

    input_list_path = out_prefix.with_name(out_prefix.name + "_input_compounds.txt")
    resolution_path = out_prefix.with_name(out_prefix.name + "_metabolite_resolution.tsv")
    enrichment_path = out_prefix.with_name(out_prefix.name + "_pathway_enrichment.tsv")

    warning = ""
    if len(universe_tokens) < SMALL_U_WARNING or len(target_tokens) < SMALL_S_WARNING:
        warning = (
            f"small sample size (U={len(universe_tokens)}, S={len(target_tokens)}), "
            "enrichment results may be unreliable"
        )
        print(f"WARNING: {warning}", flush=True)

    _write_input_names(input_list_path, input_names)
    _write_resolution_output(resolution_path, rows)
    _write_enrichment_output(enrichment_path, output_rows, warning=warning)

    resolved_count = sum(1 for row in rows if row.resolution_status == "resolved")
    ambiguous_count = sum(1 for row in rows if row.resolution_status == "ambiguous")
    unmapped_count = sum(1 for row in rows if row.resolution_status == "unmapped")

    print(f"Input rows: {len(rows)}", flush=True)
    print(f"Resolved: {resolved_count}, ambiguous: {ambiguous_count}, unmapped: {unmapped_count}", flush=True)
    print(f"n={len(target_tokens)}", flush=True)
    print(f"N={len(universe_tokens)}", flush=True)
    if online_model_bundle is not None:
        print(
            f"Online rerank model loaded: {online_model_kind} "
            f"(LightGBM {online_model_bundle['metadata'].get('lightgbm_version', '')})",
            flush=True,
        )
        print(f"Online candidate policy: {online_candidate_policy}", flush=True)
    if rerank_applied:
        print(f"ML rerank applied: {len(enrichment_rows)} candidate pathways reranked.", flush=True)
    else:
        print(f"ML rerank skipped: {rerank_reason}", flush=True)
    print(f"Input list written: {input_list_path}", flush=True)
    print(f"Resolution audit written: {resolution_path}", flush=True)
    print(f"Enriched pathways written: {len(output_rows)} -> {enrichment_path}", flush=True)
    if output_rows:
        print("Top enriched pathways:", flush=True)
        row_lookup = {row.pathway_id: row for row in enrichment_rows}
        for output_row in output_rows[:5]:
            baseline_row = row_lookup[output_row["pathway_id"]]
            summary = (
                f"  - {baseline_row.pathway_name} [{baseline_row.confidence_level}] "
                f"x_i={baseline_row.x_i}, K_i={baseline_row.K_i}, n={baseline_row.n}, N={baseline_row.N}, "
                f"EF={baseline_row.enrichment_factor:.4g}, p={baseline_row.p_value:.4g}, "
                f"FDR={baseline_row.fdr:.4g}, baseline_score={baseline_row.final_score:.4f}"
            )
            if output_row["rerank_applied"]:
                summary += f", ml_score={float(output_row['ml_score']):.4f}"
            print(summary, flush=True)
            print(f"    hits: {'; '.join(baseline_row.hit_compound_names)}", flush=True)
            if baseline_row.kegg_support_present:
                print(
                    "    "
                    f"KEGG ath support: {baseline_row.kegg_ath_pathway_id} ({baseline_row.kegg_ath_pathway_name}) "
                    f"[{baseline_row.kegg_alignment_confidence}, score={baseline_row.kegg_alignment_score:.4f}]",
                    flush=True,
                )
    else:
        print("No enriched pathways passed x_i >= 2.", flush=True)


if __name__ == "__main__":
    main()
