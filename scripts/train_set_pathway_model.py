#!/usr/bin/env python3
"""Interactive TSV-driven training for compound-set -> pathway reranking."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import pickle
import random
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import ndcg_score

try:  # Optional dependency; v1 falls back to sklearn when unavailable.
    import lightgbm as lgb
except Exception:  # pragma: no cover - optional dependency
    lgb = None

from enrich_pathways import (
    CompoundTargetRecord,
    PathwayEnrichmentRow,
    _benjamini_hochberg,
    _confidence_level,
    _compute_enrichment,
    _load_pathway_membership,
)
from pathway_query_shared import (
    DataStructs,
    QueryResult,
    ensure_chebi_structure_lookup,
    load_preprocessed_state,
    resolve_primary_compound,
)
from process_chebi_to_pathways_v2 import normalize_name


RANDOM_SEED = 42
DEFAULT_TRAIN_RATIO = 0.80
DEFAULT_NEGATIVE_LIMIT = 30
DEFAULT_HARD_NEGATIVE_LIMIT = 20
DEFAULT_MAX_SUBSETS_PER_PATHWAY = 50
DEFAULT_PERMUTATION_IMPORTANCE_ROWS = 2000


@dataclass(slots=True)
class ResolvedCompound:
    raw_token: str
    compound_id: str
    chebi_accession: str
    chebi_name: str
    match_type: str
    match_score: float


@dataclass(slots=True)
class ResolvedSample:
    sample_id: str
    compounds_raw: str
    pathway_ids_raw: str
    pathway_names_raw: str
    split: str
    weight: float
    source: str
    note: str
    resolved_compounds: tuple[ResolvedCompound, ...]
    unresolved_tokens: tuple[str, ...]
    pathway_ids: tuple[str, ...]
    pathway_names: tuple[str, ...]


@dataclass(slots=True)
class AuditRow:
    sample_id: str
    reason: str
    detail: str
    compounds_raw: str
    pathway_ids_raw: str
    pathway_names_raw: str
    split_raw: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a compound-set -> pathway reranker from a user TSV dataset."
    )
    parser.add_argument("--workdir", default=".", help="Workspace root directory.")
    parser.add_argument("--dataset", default="", help="Path to the TSV dataset.")
    parser.add_argument(
        "--compound-format",
        choices=("names", "chebi_ids"),
        default="",
        help="How the `compounds` column is encoded.",
    )
    parser.add_argument(
        "--use-split-column",
        action="store_true",
        help="Respect the TSV `split` column (train/test only).",
    )
    parser.add_argument(
        "--auto-split",
        action="store_true",
        help="Force automatic 80/20 split even if no `split` column is present.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="In TSV mode, drop a sample if any compound token fails to resolve.",
    )
    parser.add_argument(
        "--max-samples-per-pathway",
        type=int,
        default=DEFAULT_MAX_SUBSETS_PER_PATHWAY,
        help="Maximum number of generated compound-set samples per pathway in default AraCyc mode.",
    )
    parser.add_argument("--out-dir", default="", help="Output directory.")
    return parser.parse_args()


def _ask_text(prompt: str, *, default: str = "", allow_empty: bool = False) -> str:
    suffix = f" [{default}]" if default else ""
    while True:
        value = input(f"{prompt}{suffix}: ").strip()
        if value:
            return value
        if default:
            return default
        if allow_empty:
            return ""


def _ask_bool(prompt: str, *, default: bool) -> bool:
    default_text = "Y/n" if default else "y/N"
    while True:
        value = input(f"{prompt} [{default_text}]: ").strip().lower()
        if not value:
            return default
        if value in {"y", "yes"}:
            return True
        if value in {"n", "no"}:
            return False


def _ask_choice(prompt: str, choices: tuple[str, ...], *, default: str) -> str:
    joined = "/".join(choices)
    while True:
        value = input(f"{prompt} [{joined}] ({default}): ").strip().lower()
        if not value:
            return default
        if value in choices:
            return value


def _interactive_config(args: argparse.Namespace, workdir: Path) -> dict[str, Any]:
    dataset_mode = "custom_tsv" if args.dataset else "default_aracyc"
    dataset: Path | None = Path(args.dataset).resolve() if args.dataset else None

    if not args.dataset:
        raw_dataset = _ask_text(
            "Custom TSV dataset path (leave blank to use default AraCyc-generated dataset)",
            allow_empty=True,
        )
        if raw_dataset:
            dataset = Path(raw_dataset).expanduser().resolve()
            dataset_mode = "custom_tsv"

    compound_format = ""
    use_split_column = False
    auto_split = True
    strict = args.strict

    if dataset_mode == "custom_tsv":
        if dataset is None or not dataset.exists():
            raise SystemExit(f"Dataset not found: {dataset}")
        header = _read_header(dataset)
        compound_format = args.compound_format
        if not compound_format:
            compound_format = _ask_choice(
                "Compound encoding in `compounds`",
                ("names", "chebi_ids"),
                default="names",
            )

        use_split_column = args.use_split_column
        if "split" in header and not use_split_column and not args.auto_split:
            use_split_column = _ask_bool("Use TSV split column", default=True)
        elif "split" not in header:
            use_split_column = False

        auto_split = args.auto_split
        if not use_split_column and not auto_split:
            auto_split = _ask_bool("Use automatic 80/20 train/test split", default=True)
        if not use_split_column and not auto_split:
            raise SystemExit("No split strategy selected.")
    out_dir = Path(args.out_dir).resolve() if args.out_dir else None
    if out_dir is None:
        default_name = dataset.stem if dataset is not None else "aracyc_default"
        default_out = workdir / "outputs" / "ml_training" / default_name
        out_dir = Path(_ask_text("Output directory", default=str(default_out))).expanduser().resolve()

    return {
        "dataset_mode": dataset_mode,
        "dataset": dataset,
        "compound_format": compound_format,
        "use_split_column": use_split_column,
        "auto_split": auto_split,
        "strict": strict,
        "max_samples_per_pathway": max(1, int(args.max_samples_per_pathway)),
        "out_dir": out_dir,
    }


def _read_header(path: Path) -> list[str]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle, delimiter="\t")
        try:
            return next(reader)
        except StopIteration as exc:  # pragma: no cover - invalid input
            raise SystemExit(f"Dataset is empty: {path}") from exc


def _split_multi(value: str) -> list[str]:
    return [part.strip() for part in (value or "").split(";") if part.strip()]


def _read_step1_metadata(workdir: Path) -> tuple[dict[str, dict[str, str]], dict[str, str]]:
    metadata_by_id: dict[str, dict[str, str]] = {}
    compound_by_accession: dict[str, str] = {}
    path = workdir / "outputs" / "preprocessed" / "step1_result.tsv"
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            if row.get("entity_source") != "chebi":
                continue
            compound_id = (row.get("compound_id") or "").strip()
            chebi_accession = (row.get("chebi_accession") or "").strip()
            if not compound_id:
                continue
            metadata_by_id[compound_id] = row
            if chebi_accession:
                compound_by_accession[chebi_accession.upper()] = compound_id
    return metadata_by_id, compound_by_accession


def _load_pathway_name_index(pathway_names: dict[str, str]) -> dict[str, str]:
    grouped: dict[str, set[str]] = defaultdict(set)
    for pathway_id, pathway_name in pathway_names.items():
        grouped[normalize_name(pathway_name)].add(pathway_id)
    return {name: next(iter(ids)) for name, ids in grouped.items() if len(ids) == 1}


def _load_parent_map(workdir: Path) -> dict[str, str]:
    parent_map: dict[str, str] = {}
    with (workdir / "compounds.tsv").open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            compound_id = row["id"]
            parent_id = (row.get("parent_id") or "").strip()
            parent_map[compound_id] = parent_id
    return parent_map


def _root_ancestor(compound_id: str, parent_map: dict[str, str], cache: dict[str, str]) -> str:
    cached = cache.get(compound_id)
    if cached is not None:
        return cached
    seen: set[str] = set()
    current = compound_id
    root = compound_id
    while current and current not in seen:
        seen.add(current)
        root = current
        current = parent_map.get(current, "")
    cache[compound_id] = root
    return root


def _resolve_name_token(token: str, state: dict[str, Any]) -> tuple[ResolvedCompound | None, str]:
    resolution = resolve_primary_compound(token, state, no_fuzzy=False)
    if resolution.status != "resolved" or not resolution.compound_id:
        detail = resolution.note or resolution.status
        if resolution.did_you_mean:
            detail = f"{detail} | did_you_mean={'; '.join(resolution.did_you_mean)}"
        return None, detail
    return (
        ResolvedCompound(
            raw_token=token,
            compound_id=resolution.compound_id,
            chebi_accession=resolution.chebi_accession,
            chebi_name=resolution.chebi_name,
            match_type=resolution.match_type or "direct",
            match_score=resolution.match_score,
        ),
        "",
    )


def _resolve_id_token(
    token: str,
    metadata_by_id: dict[str, dict[str, str]],
    compound_by_accession: dict[str, str],
) -> tuple[ResolvedCompound | None, str]:
    text = token.strip()
    if not text:
        return None, "empty token"
    compound_id = ""
    if text.upper().startswith("CHEBI:"):
        compound_id = compound_by_accession.get(text.upper(), "")
    elif text.isdigit():
        compound_id = text if text in metadata_by_id else ""
    if not compound_id:
        return None, f"unknown ChEBI identifier: {text}"
    row = metadata_by_id[compound_id]
    return (
        ResolvedCompound(
            raw_token=text,
            compound_id=compound_id,
            chebi_accession=row.get("chebi_accession", ""),
            chebi_name=row.get("display_name", ""),
            match_type="direct",
            match_score=1.0,
        ),
        "",
    )


def _map_pathways(
    row: dict[str, str],
    *,
    pathway_names: dict[str, str],
    pathway_name_index: dict[str, str],
) -> tuple[list[str], list[str], list[str]]:
    mapped_ids: list[str] = []
    warnings: list[str] = []

    for raw_id in _split_multi(row.get("pathway_id", "")):
        if raw_id in pathway_names:
            mapped_ids.append(raw_id)
        else:
            warnings.append(f"unknown pathway_id={raw_id}")

    for raw_name in _split_multi(row.get("pathway_name", "")):
        normalized = normalize_name(raw_name)
        pathway_id = pathway_name_index.get(normalized, "")
        if pathway_id:
            mapped_ids.append(pathway_id)
        else:
            warnings.append(f"unmapped pathway_name={raw_name}")

    deduped_ids = tuple(sorted(set(mapped_ids)))
    names = [pathway_names[pathway_id] for pathway_id in deduped_ids]
    return list(deduped_ids), names, warnings


def _resolve_dataset(
    dataset_path: Path,
    *,
    compound_format: str,
    use_split_column: bool,
    strict: bool,
    state: dict[str, Any],
    metadata_by_id: dict[str, dict[str, str]],
    compound_by_accession: dict[str, str],
    pathway_names: dict[str, str],
    pathway_name_index: dict[str, str],
) -> tuple[list[ResolvedSample], list[AuditRow]]:
    resolved: list[ResolvedSample] = []
    audit: list[AuditRow] = []

    with dataset_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None:
            raise SystemExit(f"No header found in dataset: {dataset_path}")
        required = {"compounds"}
        if not required.issubset(reader.fieldnames):
            missing = sorted(required - set(reader.fieldnames))
            raise SystemExit(f"Missing required columns: {', '.join(missing)}")

        for row_index, row in enumerate(reader, start=1):
            sample_id = (row.get("sample_id") or f"row_{row_index}").strip() or f"row_{row_index}"
            compounds_raw = row.get("compounds", "") or ""
            pathway_ids_raw = row.get("pathway_id", "") or ""
            pathway_names_raw = row.get("pathway_name", "") or ""
            split_raw = (row.get("split", "") or "").strip().lower()
            split = split_raw if use_split_column else ""
            if use_split_column and split not in {"train", "test"}:
                audit.append(
                    AuditRow(
                        sample_id=sample_id,
                        reason="invalid_split",
                        detail=f"Unsupported split value: {split_raw!r}; v1 only supports train/test.",
                        compounds_raw=compounds_raw,
                        pathway_ids_raw=pathway_ids_raw,
                        pathway_names_raw=pathway_names_raw,
                        split_raw=split_raw,
                    )
                )
                continue

            tokens = _split_multi(compounds_raw)
            if not tokens:
                audit.append(
                    AuditRow(
                        sample_id=sample_id,
                        reason="empty_compounds",
                        detail="No compounds listed.",
                        compounds_raw=compounds_raw,
                        pathway_ids_raw=pathway_ids_raw,
                        pathway_names_raw=pathway_names_raw,
                        split_raw=split_raw,
                    )
                )
                continue

            resolved_compounds: list[ResolvedCompound] = []
            resolution_errors: list[str] = []
            seen_compounds: set[str] = set()
            for token in tokens:
                if compound_format == "names":
                    result, error = _resolve_name_token(token, state)
                else:
                    result, error = _resolve_id_token(token, metadata_by_id, compound_by_accession)
                if result is None:
                    resolution_errors.append(f"{token}: {error}")
                    continue
                if result.compound_id in seen_compounds:
                    continue
                seen_compounds.add(result.compound_id)
                resolved_compounds.append(result)

            unresolved_tokens = tuple(resolution_errors)
            if resolution_errors and strict:
                audit.append(
                    AuditRow(
                        sample_id=sample_id,
                        reason="compound_resolution_failed",
                        detail=" | ".join(resolution_errors),
                        compounds_raw=compounds_raw,
                        pathway_ids_raw=pathway_ids_raw,
                        pathway_names_raw=pathway_names_raw,
                        split_raw=split_raw,
                    )
                )
                continue
            if resolution_errors:
                audit.append(
                    AuditRow(
                        sample_id=sample_id,
                        reason="compound_resolution_partial",
                        detail=" | ".join(resolution_errors),
                        compounds_raw=compounds_raw,
                        pathway_ids_raw=pathway_ids_raw,
                        pathway_names_raw=pathway_names_raw,
                        split_raw=split_raw,
                    )
                )

            if len(resolved_compounds) < 2:
                audit.append(
                    AuditRow(
                        sample_id=sample_id,
                        reason="too_small_set",
                        detail=(
                            f"Need at least 2 unique resolved compounds, got {len(resolved_compounds)}."
                            + (f" Unresolved: {' | '.join(resolution_errors)}" if resolution_errors else "")
                        ),
                        compounds_raw=compounds_raw,
                        pathway_ids_raw=pathway_ids_raw,
                        pathway_names_raw=pathway_names_raw,
                        split_raw=split_raw,
                    )
                )
                continue

            pathway_ids, pathway_label_names, pathway_warnings = _map_pathways(
                row,
                pathway_names=pathway_names,
                pathway_name_index=pathway_name_index,
            )
            if not pathway_ids:
                audit.append(
                    AuditRow(
                        sample_id=sample_id,
                        reason="pathway_unmapped",
                        detail=" | ".join(pathway_warnings) or "No AraCyc pathway labels mapped.",
                        compounds_raw=compounds_raw,
                        pathway_ids_raw=pathway_ids_raw,
                        pathway_names_raw=pathway_names_raw,
                        split_raw=split_raw,
                    )
                )
                continue

            try:
                weight = float((row.get("weight") or "").strip() or "1.0")
            except ValueError:
                weight = 1.0

            resolved.append(
                ResolvedSample(
                    sample_id=sample_id,
                    compounds_raw=compounds_raw,
                    pathway_ids_raw=pathway_ids_raw,
                    pathway_names_raw=pathway_names_raw,
                    split=split,
                    weight=weight,
                    source=(row.get("source") or "").strip(),
                    note=(row.get("note") or "").strip(),
                    resolved_compounds=tuple(resolved_compounds),
                    unresolved_tokens=unresolved_tokens,
                    pathway_ids=tuple(pathway_ids),
                    pathway_names=tuple(pathway_label_names),
                )
            )

    return resolved, audit


def _assign_auto_split(samples: list[ResolvedSample], rng: random.Random) -> list[ResolvedSample]:
    shuffled = list(samples)
    rng.shuffle(shuffled)
    train_count = _choose_train_count(len(shuffled))
    train_ids = {sample.sample_id for sample in shuffled[:train_count]}
    updated: list[ResolvedSample] = []
    for sample in samples:
        updated.append(
            ResolvedSample(
                sample_id=sample.sample_id,
                compounds_raw=sample.compounds_raw,
                pathway_ids_raw=sample.pathway_ids_raw,
                pathway_names_raw=sample.pathway_names_raw,
                split="train" if sample.sample_id in train_ids else "test",
                weight=sample.weight,
                source=sample.source,
                note=sample.note,
                resolved_compounds=sample.resolved_compounds,
                unresolved_tokens=sample.unresolved_tokens,
                pathway_ids=sample.pathway_ids,
                pathway_names=sample.pathway_names,
            )
        )
    return updated


def _choose_train_count(total: int) -> int:
    if total <= 1:
        return total
    train_count = int(round(total * DEFAULT_TRAIN_RATIO))
    return max(1, min(total - 1, train_count))


def _assert_unique_sample_ids(samples: list[ResolvedSample]) -> None:
    counts = Counter(sample.sample_id for sample in samples)
    duplicates = sorted(sample_id for sample_id, count in counts.items() if count > 1)
    if duplicates:
        preview = ", ".join(duplicates[:10])
        suffix = "" if len(duplicates) <= 10 else f" (+{len(duplicates) - 10} more)"
        raise SystemExit(f"Duplicate sample_id values are not allowed: {preview}{suffix}")


def _sample_subset_tuples(
    compound_ids: list[str],
    *,
    rng: random.Random,
    limit: int = DEFAULT_MAX_SUBSETS_PER_PATHWAY,
) -> list[tuple[str, ...]]:
    ordered = sorted(dict.fromkeys(compound_ids))
    max_size = min(10, len(ordered))
    if max_size < 2:
        return []

    size_counts = {size: math.comb(len(ordered), size) for size in range(2, max_size + 1)}
    total = sum(size_counts.values())
    if total <= limit:
        subsets: list[tuple[str, ...]] = []
        for size in range(2, max_size + 1):
            subsets.extend(itertools.combinations(ordered, size))
        return [tuple(combo) for combo in subsets]

    target = min(limit, total)
    sizes = tuple(size_counts)
    weights = tuple(size_counts[size] for size in sizes)
    sampled: set[tuple[str, ...]] = set()
    attempts = 0
    max_attempts = target * 100
    while len(sampled) < target and attempts < max_attempts:
        size = rng.choices(sizes, weights=weights, k=1)[0]
        sampled.add(tuple(sorted(rng.sample(ordered, size))))
        attempts += 1
    return sorted(sampled)


def _generate_default_aracyc_dataset(
    *,
    pathway_names: dict[str, str],
    pathway_to_tokens: dict[str, set[str]],
    metadata_by_id: dict[str, dict[str, str]],
    max_samples_per_pathway: int,
    rng: random.Random,
) -> tuple[list[ResolvedSample], list[AuditRow], dict[str, Any]]:
    all_compounds = sorted(
        {
            token.split(":", 1)[1]
            for members in pathway_to_tokens.values()
            for token in members
            if token.startswith("cmp:") and token.split(":", 1)[1] in metadata_by_id
        }
    )
    if len(all_compounds) < 2:
        raise SystemExit("AraCyc default dataset cannot be generated: fewer than 2 compounds are available.")

    shuffled_compounds = list(all_compounds)
    rng.shuffle(shuffled_compounds)
    train_count = _choose_train_count(len(shuffled_compounds))
    train_compounds = set(shuffled_compounds[:train_count])
    test_compounds = set(shuffled_compounds[train_count:])

    pathway_to_compounds: dict[str, set[str]] = {
        pathway_id: {
            token.split(":", 1)[1]
            for token in members
            if token.startswith("cmp:") and token.split(":", 1)[1] in metadata_by_id
        }
        for pathway_id, members in pathway_to_tokens.items()
    }

    eligible_pathways: dict[str, dict[str, list[str]]] = {}
    for pathway_id, members in pathway_to_compounds.items():
        train_members = sorted(members & train_compounds)
        test_members = sorted(members & test_compounds)
        if len(train_members) < 2 or len(test_members) < 2:
            continue
        eligible_pathways[pathway_id] = {"train": train_members, "test": test_members}

    if not eligible_pathways:
        raise SystemExit("AraCyc default dataset generation produced no pathways with train/test support.")

    compound_to_pathways: dict[str, set[str]] = defaultdict(set)
    for pathway_id, split_members in eligible_pathways.items():
        for members in split_members.values():
            for compound_id in members:
                compound_to_pathways[compound_id].add(pathway_id)

    generated_samples: list[ResolvedSample] = []
    generated_counts = {"train": 0, "test": 0}
    estimated_pair_count = 0
    for pathway_id, split_members in sorted(eligible_pathways.items()):
        for split, members in split_members.items():
            subsets = _sample_subset_tuples(members, rng=rng, limit=max_samples_per_pathway)
            for subset_index, subset in enumerate(subsets, start=1):
                positive_ids = sorted(
                    set.intersection(*(compound_to_pathways[compound_id] for compound_id in subset))
                )
                if not positive_ids:
                    continue
                resolved_compounds = []
                for compound_id in subset:
                    row = metadata_by_id[compound_id]
                    resolved_compounds.append(
                        ResolvedCompound(
                            raw_token=row.get("display_name", "") or row.get("chebi_accession", "") or compound_id,
                            compound_id=compound_id,
                            chebi_accession=row.get("chebi_accession", ""),
                            chebi_name=row.get("display_name", "") or row.get("chebi_name", "") or compound_id,
                            match_type="direct",
                            match_score=1.0,
                        )
                    )
                positive_names = tuple(pathway_names[path_id] for path_id in positive_ids)
                generated_samples.append(
                    ResolvedSample(
                        sample_id=f"aracyc_{split}_{pathway_id}_{subset_index:03d}",
                        compounds_raw="; ".join(item.chebi_name for item in resolved_compounds),
                        pathway_ids_raw=";".join(positive_ids),
                        pathway_names_raw="; ".join(positive_names),
                        split=split,
                        weight=1.0,
                        source="aracyc_generated",
                        note=f"seed_pathway={pathway_id}",
                        resolved_compounds=tuple(resolved_compounds),
                        unresolved_tokens=(),
                        pathway_ids=tuple(positive_ids),
                        pathway_names=positive_names,
                    )
                )
                generated_counts[split] += 1
                estimated_pair_count += len(positive_ids) + DEFAULT_NEGATIVE_LIMIT

    if not generated_samples:
        raise SystemExit("AraCyc default dataset generation produced no samples.")

    generation_summary = {
        "eligible_pathways": len(eligible_pathways),
        "train_compounds": len(train_compounds),
        "test_compounds": len(test_compounds),
        "generated_train_samples": generated_counts["train"],
        "generated_test_samples": generated_counts["test"],
        "estimated_pair_count": estimated_pair_count,
        "max_samples_per_pathway": max_samples_per_pathway,
    }
    return generated_samples, [], generation_summary


def _ensure_fingerprint(record) -> Any:
    if record is None:
        return None
    if record.fingerprint is not None:
        return record.fingerprint
    if not record.smiles or DataStructs is None:
        return None
    from pathway_query_shared import _morgan_fingerprint_from_mol, _mol_from_smiles

    mol = _mol_from_smiles(record.smiles)
    if mol is None:
        return None
    record.fingerprint = _morgan_fingerprint_from_mol(mol)
    return record.fingerprint


def _pairwise_tanimoto(compound_ids: list[str], structure_lookup: dict[str, Any]) -> tuple[float, float, float]:
    if DataStructs is None or len(compound_ids) < 2:
        return 0.0, 0.0, 0.0
    values: list[float] = []
    for i in range(len(compound_ids)):
        left = structure_lookup.get(compound_ids[i])
        left_fp = _ensure_fingerprint(left)
        if left_fp is None:
            continue
        for j in range(i + 1, len(compound_ids)):
            right = structure_lookup.get(compound_ids[j])
            right_fp = _ensure_fingerprint(right)
            if right_fp is None:
                continue
            values.append(float(DataStructs.TanimotoSimilarity(left_fp, right_fp)))
    if not values:
        return 0.0, 0.0, 0.0
    return float(sum(values) / len(values)), float(max(values)), float(min(values))


def _query_level_features(
    sample: ResolvedSample,
    structure_lookup: dict[str, Any],
    parent_map: dict[str, str],
    root_cache: dict[str, str],
) -> dict[str, float]:
    compound_ids = [item.compound_id for item in sample.resolved_compounds]
    tan_mean, tan_max, tan_min = _pairwise_tanimoto(compound_ids, structure_lookup)

    formula_keys = {
        getattr(structure_lookup.get(compound_id), "formula_key", "")
        for compound_id in compound_ids
        if structure_lookup.get(compound_id) is not None and getattr(structure_lookup.get(compound_id), "formula_key", "")
    }
    charges = {
        getattr(structure_lookup.get(compound_id), "charge", "")
        for compound_id in compound_ids
        if structure_lookup.get(compound_id) is not None and getattr(structure_lookup.get(compound_id), "charge", "")
    }
    roots = {_root_ancestor(compound_id, parent_map, root_cache) for compound_id in compound_ids}

    size = len(compound_ids)
    return {
        "set_size": float(size),
        "pairwise_tanimoto_mean": tan_mean,
        "pairwise_tanimoto_max": tan_max,
        "pairwise_tanimoto_min": tan_min,
        "ontology_diversity": len(roots) / size if size else 0.0,
        "formula_diversity": len(formula_keys) / size if size else 0.0,
        "charge_diversity": len(charges) / size if size else 0.0,
    }


def _compound_records(sample: ResolvedSample) -> dict[str, CompoundTargetRecord]:
    records: dict[str, CompoundTargetRecord] = {}
    for item in sample.resolved_compounds:
        records[item.compound_id] = CompoundTargetRecord(
            compound_id=item.compound_id,
            chebi_name=item.chebi_name,
            match_type=item.match_type or "direct",
            match_score=item.match_score,
        )
    return records


def _single_pathway_row(
    pathway_id: str,
    *,
    pathway_names: dict[str, str],
    pathway_to_tokens: dict[str, set[str]],
    universe_tokens: set[str],
    target_tokens: set[str],
    compound_records: dict[str, CompoundTargetRecord],
) -> PathwayEnrichmentRow | None:
    members = pathway_to_tokens.get(pathway_id)
    if not members:
        return None

    N = len(universe_tokens)
    n = len(target_tokens)
    K_i = len(members)
    x_tokens = members & target_tokens
    x_i = len(x_tokens)
    if K_i == 0 or N == 0 or n == 0:
        return None

    target_rate = x_i / n
    background_rate = K_i / N
    expected_hits = n * K_i / N
    excess_hits = x_i - expected_hits
    enrichment_factor = (target_rate / background_rate) if background_rate > 0 else 0.0
    coverage = x_i / K_i

    from enrich_pathways import _hypergeom_p_value

    p_value = _hypergeom_p_value(N, K_i, n, x_i)
    # These placeholder values are overwritten after candidate collection, when
    # _build_enrichment_rows() applies a single BH correction across the full
    # candidate subset for the sample.
    fdr = 1.0
    neg_log10_p = -math.log10(max(p_value, 1e-300))
    neg_log10_fdr = 0.0

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

    ef_norm = 1.0 if enrichment_factor >= 1.0 else 0.0
    final_score = 0.55 * ef_norm + 0.30 * coverage + 0.15 * mapping_confidence_mean

    return PathwayEnrichmentRow(
        pathway_id=pathway_id,
        pathway_name=pathway_names.get(pathway_id, pathway_id),
        x_i=x_i,
        K_i=K_i,
        n=n,
        N=N,
        target_rate=target_rate,
        background_rate=background_rate,
        expected_hits=expected_hits,
        excess_hits=excess_hits,
        enrichment_factor=enrichment_factor,
        coverage=coverage,
        p_value=p_value,
        fdr=fdr,
        neg_log10_p=neg_log10_p,
        neg_log10_fdr=neg_log10_fdr,
        mapping_confidence_mean=mapping_confidence_mean,
        mapping_confidence_min=mapping_confidence_min,
        mapping_confidence_max=mapping_confidence_max,
        direct_hit_count=direct_hit_count,
        fuzzy_hit_count=fuzzy_hit_count,
        recovered_hit_count=recovered_hit_count,
        exact_structure_hit_count=exact_structure_hit_count,
        structural_neighbor_hit_count=structural_neighbor_hit_count,
        final_score=round(final_score, 4),
        confidence_level="pending_bh",
        hit_compound_ids=hit_compound_ids,
        hit_compound_names=tuple(record.chebi_name or f"CHEBI:{record.compound_id}" for record in hit_records),
    )


def _build_enrichment_rows(
    samples: list[ResolvedSample],
    *,
    split: str,
    pathway_names: dict[str, str],
    pathway_to_tokens: dict[str, set[str]],
    universe_tokens: set[str],
    structure_lookup: dict[str, Any],
    parent_map: dict[str, str],
    rng: random.Random,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    all_pathway_ids = sorted(pathway_to_tokens)
    root_cache: dict[str, str] = {}

    for sample in samples:
        target_tokens = {f"cmp:{item.compound_id}" for item in sample.resolved_compounds}
        compound_records = _compound_records(sample)
        baseline_rows = _compute_enrichment(
            pathway_names,
            pathway_to_tokens,
            universe_tokens,
            target_tokens,
            compound_records,
        )
        baseline_by_id = {row.pathway_id: row for row in baseline_rows}
        positive_ids = set(sample.pathway_ids)

        hard_negatives = [
            row.pathway_id for row in baseline_rows if row.pathway_id not in positive_ids
        ][:DEFAULT_HARD_NEGATIVE_LIMIT]
        negative_ids = list(hard_negatives)
        remaining_pool = [
            pathway_id
            for pathway_id in all_pathway_ids
            if pathway_id not in positive_ids and pathway_id not in negative_ids
        ]
        rng.shuffle(remaining_pool)
        needed = max(0, DEFAULT_NEGATIVE_LIMIT - len(negative_ids))
        negative_ids.extend(remaining_pool[:needed])

        candidate_ids = list(sorted(positive_ids)) + negative_ids
        query_features = _query_level_features(sample, structure_lookup, parent_map, root_cache)
        candidate_feature_rows: list[PathwayEnrichmentRow] = []
        for pathway_id in candidate_ids:
            feature_row = baseline_by_id.get(pathway_id)
            if feature_row is None:
                feature_row = _single_pathway_row(
                    pathway_id,
                    pathway_names=pathway_names,
                    pathway_to_tokens=pathway_to_tokens,
                    universe_tokens=universe_tokens,
                    target_tokens=target_tokens,
                    compound_records=compound_records,
                )
            if feature_row is None:
                continue
            candidate_feature_rows.append(feature_row)

        if not candidate_feature_rows:
            continue

        adjusted_fdrs = _benjamini_hochberg([row.p_value for row in candidate_feature_rows])
        for feature_row, adjusted_fdr in zip(candidate_feature_rows, adjusted_fdrs, strict=False):
            adjusted_neg_log10_fdr = -math.log10(max(adjusted_fdr, 1e-300))
            confidence_level = _confidence_level(feature_row.x_i, adjusted_fdr)

            rows.append(
                {
                    "sample_id": sample.sample_id,
                    "split": split,
                    "label": 1 if feature_row.pathway_id in positive_ids else 0,
                    "sample_weight": sample.weight,
                    "pathway_id": feature_row.pathway_id,
                    "pathway_name": feature_row.pathway_name,
                    "compounds_raw": sample.compounds_raw,
                    "compound_ids": ";".join(item.compound_id for item in sample.resolved_compounds),
                    "compound_names": "; ".join(item.chebi_name for item in sample.resolved_compounds),
                    "positive_pathway_ids": ";".join(sample.pathway_ids),
                    "positive_pathway_names": "; ".join(sample.pathway_names),
                    "baseline_score": feature_row.final_score,
                    "x_i": feature_row.x_i,
                    "K_i": feature_row.K_i,
                    "n": feature_row.n,
                    "N": feature_row.N,
                    "query_coverage": feature_row.target_rate,
                    "pathway_coverage": feature_row.coverage,
                    "background_rate": feature_row.background_rate,
                    "expected_hits": feature_row.expected_hits,
                    "excess_hits": feature_row.excess_hits,
                    "EF": feature_row.enrichment_factor,
                    "p_value": feature_row.p_value,
                    "FDR": adjusted_fdr,
                    "neg_log10_p": feature_row.neg_log10_p,
                    "neg_log10_fdr": adjusted_neg_log10_fdr,
                    "mapping_confidence_mean": feature_row.mapping_confidence_mean,
                    "mapping_confidence_min": feature_row.mapping_confidence_min,
                    "mapping_confidence_max": feature_row.mapping_confidence_max,
                    "direct_hit_count": feature_row.direct_hit_count,
                    "fuzzy_hit_count": feature_row.fuzzy_hit_count,
                    "recovered_hit_count": feature_row.recovered_hit_count,
                    "exact_structure_hit_count": feature_row.exact_structure_hit_count,
                    "structural_neighbor_hit_count": feature_row.structural_neighbor_hit_count,
                    "confidence_level": confidence_level,
                    **query_features,
                }
            )

    return rows


def _pair_rows_from_enrichment_rows(enrichment_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [dict(row) for row in enrichment_rows]


def _numeric_feature_columns(rows: list[dict[str, Any]]) -> list[str]:
    if not rows:
        return []
    excluded = {
        "sample_id",
        "split",
        "label",
        "sample_weight",
        "pathway_id",
        "pathway_name",
        "compounds_raw",
        "compound_ids",
        "compound_names",
        "positive_pathway_ids",
        "positive_pathway_names",
        # Exported for audit/readability only; not a numeric model feature.
        "confidence_level",
    }
    columns: list[str] = []
    for key, value in rows[0].items():
        if key in excluded:
            continue
        if isinstance(value, (int, float)):
            columns.append(key)
    return columns


def _train_model(
    train_rows: list[dict[str, Any]],
    feature_columns: list[str],
) -> tuple[str, Any]:
    X = np.asarray([[float(row[col]) for col in feature_columns] for row in train_rows], dtype=float)
    y = np.asarray([int(row["label"]) for row in train_rows], dtype=int)
    base_sample_weight = np.asarray([float(row["sample_weight"]) for row in train_rows], dtype=float)

    if lgb is not None:
        order = sorted(range(len(train_rows)), key=lambda idx: train_rows[idx]["sample_id"])
        X_rank = X[order]
        y_rank = y[order]
        weight_rank = base_sample_weight[order]
        groups: list[int] = []
        current_sample = None
        count = 0
        for idx in order:
            sample_id = train_rows[idx]["sample_id"]
            if sample_id != current_sample:
                if count:
                    groups.append(count)
                current_sample = sample_id
                count = 0
            count += 1
        if count:
            groups.append(count)

        model = lgb.LGBMRanker(
            objective="lambdarank",
            random_state=RANDOM_SEED,
            n_estimators=200,
            learning_rate=0.05,
            num_leaves=31,
            min_child_samples=10,
        )
        model.fit(X_rank, y_rank, group=groups, sample_weight=weight_rank)
        return "lightgbm_lambdarank", model

    positives = int(y.sum())
    negatives = len(y) - positives
    pos_scale = (negatives / positives) if positives and negatives else 1.0
    sample_weight = np.asarray([
        weight * (pos_scale if label == 1 else 1.0) for weight, label in zip(base_sample_weight, y, strict=False)
    ])
    model = HistGradientBoostingClassifier(
        random_state=RANDOM_SEED,
        max_depth=6,
        learning_rate=0.08,
        max_iter=200,
    )
    model.fit(X, y, sample_weight=sample_weight)
    return "sklearn_hist_gradient_boosting_fallback", model


def _predict_scores(model_kind: str, model: Any, rows: list[dict[str, Any]], feature_columns: list[str]) -> np.ndarray:
    X = np.asarray([[float(row[col]) for col in feature_columns] for row in rows], dtype=float)
    if model_kind == "lightgbm_lambdarank":
        return np.asarray(model.predict(X), dtype=float)
    probabilities = model.predict_proba(X)
    return np.asarray(probabilities[:, 1], dtype=float)


def _feature_importance(
    model_kind: str,
    model: Any,
    feature_columns: list[str],
    reference_rows: list[dict[str, Any]],
    rng: random.Random,
) -> list[dict[str, float | str]]:
    if not feature_columns or not reference_rows:
        return []

    if model_kind == "lightgbm_lambdarank" and hasattr(model, "feature_importances_"):
        values = np.asarray(model.feature_importances_, dtype=float)
    else:
        sample_rows = list(reference_rows)
        if len(sample_rows) > DEFAULT_PERMUTATION_IMPORTANCE_ROWS:
            sample_rows = rng.sample(sample_rows, DEFAULT_PERMUTATION_IMPORTANCE_ROWS)
        X = np.asarray([[float(row[col]) for col in feature_columns] for row in sample_rows], dtype=float)
        y = np.asarray([int(row["label"]) for row in sample_rows], dtype=int)
        result = permutation_importance(
            model,
            X,
            y,
            n_repeats=5,
            random_state=RANDOM_SEED,
            scoring="accuracy",
        )
        values = np.asarray(result.importances_mean, dtype=float)

    ranked = sorted(zip(feature_columns, values, strict=False), key=lambda item: float(item[1]), reverse=True)
    return [
        {"feature": feature, "importance": float(value), "rank": rank}
        for rank, (feature, value) in enumerate(ranked, start=1)
    ]


def _group_rows(rows: list[dict[str, Any]], scores_key: str) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["sample_id"]].append(row)
    for sample_rows in grouped.values():
        sample_rows.sort(key=lambda row: (float(row[scores_key]), float(row["baseline_score"])), reverse=True)
    return dict(grouped)


def _evaluate(
    rows: list[dict[str, Any]],
    *,
    baseline_key: str,
    model_key: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    grouped_raw: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped_raw[row["sample_id"]].append(row)

    prediction_rows: list[dict[str, Any]] = []
    baseline_ndcgs: list[float] = []
    model_ndcgs: list[float] = []
    baseline_mrrs: list[float] = []
    model_mrrs: list[float] = []
    baseline_recall = {1: [], 3: [], 5: []}
    model_recall = {1: [], 3: [], 5: []}

    for sample_id, sample_rows in grouped_raw.items():
        labels = np.asarray([int(row["label"]) for row in sample_rows], dtype=float)
        baseline_scores = np.asarray([float(row[baseline_key]) for row in sample_rows], dtype=float)
        model_scores = np.asarray([float(row[model_key]) for row in sample_rows], dtype=float)

        if labels.sum() <= 0:
            continue

        baseline_ndcgs.append(float(ndcg_score(labels.reshape(1, -1), baseline_scores.reshape(1, -1), k=5)))
        model_ndcgs.append(float(ndcg_score(labels.reshape(1, -1), model_scores.reshape(1, -1), k=5)))

        baseline_order = np.argsort(-baseline_scores)
        model_order = np.argsort(-model_scores)

        baseline_rank_map = {int(idx): rank + 1 for rank, idx in enumerate(baseline_order)}
        model_rank_map = {int(idx): rank + 1 for rank, idx in enumerate(model_order)}

        baseline_first = next((rank for rank, idx in enumerate(baseline_order, start=1) if labels[idx] > 0), len(sample_rows) + 1)
        model_first = next((rank for rank, idx in enumerate(model_order, start=1) if labels[idx] > 0), len(sample_rows) + 1)
        baseline_mrrs.append(0.0 if baseline_first > len(sample_rows) else 1.0 / baseline_first)
        model_mrrs.append(0.0 if model_first > len(sample_rows) else 1.0 / model_first)

        positive_total = int(labels.sum())
        for k in (1, 3, 5):
            baseline_hits = sum(int(labels[idx]) for idx in baseline_order[:k])
            model_hits = sum(int(labels[idx]) for idx in model_order[:k])
            baseline_recall[k].append(baseline_hits / positive_total)
            model_recall[k].append(model_hits / positive_total)

        for idx, row in enumerate(sample_rows):
            prediction_rows.append(
                {
                    **row,
                    "baseline_rank": baseline_rank_map[idx],
                    "ml_rank": model_rank_map[idx],
                }
            )

    def _mean(values: list[float]) -> float:
        return float(sum(values) / len(values)) if values else 0.0

    metrics = {
        "baseline": {
            "ndcg_at_5": _mean(baseline_ndcgs),
            "mrr": _mean(baseline_mrrs),
            "recall_at_1": _mean(baseline_recall[1]),
            "recall_at_3": _mean(baseline_recall[3]),
            "recall_at_5": _mean(baseline_recall[5]),
        },
        "ml": {
            "ndcg_at_5": _mean(model_ndcgs),
            "mrr": _mean(model_mrrs),
            "recall_at_1": _mean(model_recall[1]),
            "recall_at_3": _mean(model_recall[3]),
            "recall_at_5": _mean(model_recall[5]),
        },
        "samples_evaluated": len({row["sample_id"] for row in prediction_rows}),
    }
    prediction_rows.sort(key=lambda row: (row["sample_id"], row["ml_rank"], row["baseline_rank"]))
    return metrics, prediction_rows


def _shared_overlap(samples: list[ResolvedSample]) -> dict[str, float]:
    train_samples = [sample for sample in samples if sample.split == "train"]
    test_samples = [sample for sample in samples if sample.split == "test"]

    train_compounds = {item.compound_id for sample in train_samples for item in sample.resolved_compounds}
    test_compounds = {item.compound_id for sample in test_samples for item in sample.resolved_compounds}
    train_pathways = {pathway_id for sample in train_samples for pathway_id in sample.pathway_ids}
    test_pathways = {pathway_id for sample in test_samples for pathway_id in sample.pathway_ids}

    shared_compounds = train_compounds & test_compounds
    shared_pathways = train_pathways & test_pathways

    return {
        "train_compounds": len(train_compounds),
        "test_compounds": len(test_compounds),
        "shared_compounds": len(shared_compounds),
        "shared_compound_ratio": (len(shared_compounds) / len(test_compounds)) if test_compounds else 0.0,
        "train_pathways": len(train_pathways),
        "test_pathways": len(test_pathways),
        "shared_pathways": len(shared_pathways),
        "shared_pathway_ratio": (len(shared_pathways) / len(test_pathways)) if test_pathways else 0.0,
    }


def _write_tsv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _resolved_rows(samples: list[ResolvedSample]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sample in samples:
        rows.append(
            {
                "sample_id": sample.sample_id,
                "compounds_raw": sample.compounds_raw,
                "compound_ids": ";".join(item.compound_id for item in sample.resolved_compounds),
                "chebi_accessions": ";".join(item.chebi_accession for item in sample.resolved_compounds),
                "chebi_names": "; ".join(item.chebi_name for item in sample.resolved_compounds),
                "unresolved_count": len(sample.unresolved_tokens),
                "unresolved_tokens": " | ".join(sample.unresolved_tokens),
                "pathway_ids": ";".join(sample.pathway_ids),
                "pathway_names": "; ".join(sample.pathway_names),
                "split": sample.split,
                "weight": f"{sample.weight:.6g}",
                "source": sample.source,
                "note": sample.note,
            }
        )
    return rows


def _audit_rows(rows: list[AuditRow]) -> list[dict[str, Any]]:
    return [asdict(row) for row in rows]


def _enrichment_fieldnames() -> list[str]:
    return [
        "sample_id",
        "split",
        "label",
        "sample_weight",
        "pathway_id",
        "pathway_name",
        "compounds_raw",
        "compound_ids",
        "compound_names",
        "positive_pathway_ids",
        "positive_pathway_names",
        "baseline_score",
        "x_i",
        "K_i",
        "n",
        "N",
        "query_coverage",
        "pathway_coverage",
        "background_rate",
        "expected_hits",
        "excess_hits",
        "EF",
        "p_value",
        "FDR",
        "neg_log10_p",
        "neg_log10_fdr",
        "confidence_level",
        "mapping_confidence_mean",
        "mapping_confidence_min",
        "mapping_confidence_max",
        "direct_hit_count",
        "fuzzy_hit_count",
        "recovered_hit_count",
        "exact_structure_hit_count",
        "structural_neighbor_hit_count",
        "set_size",
        "pairwise_tanimoto_mean",
        "pairwise_tanimoto_max",
        "pairwise_tanimoto_min",
        "ontology_diversity",
        "formula_diversity",
        "charge_diversity",
    ]


def main() -> None:
    args = parse_args()
    workdir = Path(args.workdir).resolve()
    config = _interactive_config(args, workdir)

    rng = random.Random(RANDOM_SEED)
    out_dir: Path = config["out_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading query/enrichment state...", flush=True)
    state = load_preprocessed_state(workdir, verbose=True)
    pathway_index_path = workdir / "outputs" / "preprocessed" / "aracyc_compound_pathway_index.tsv"
    pathway_names, pathway_to_tokens, universe_tokens = _load_pathway_membership(pathway_index_path)
    pathway_name_index = _load_pathway_name_index(pathway_names)
    metadata_by_id, compound_by_accession = _read_step1_metadata(workdir)
    parent_map = _load_parent_map(workdir)
    structure_lookup = ensure_chebi_structure_lookup(state, verbose=False)

    generation_summary: dict[str, Any] = {}
    if config["dataset_mode"] == "default_aracyc":
        resolved_samples, audit_rows, generation_summary = _generate_default_aracyc_dataset(
            pathway_names=pathway_names,
            pathway_to_tokens=pathway_to_tokens,
            metadata_by_id=metadata_by_id,
            max_samples_per_pathway=config["max_samples_per_pathway"],
            rng=rng,
        )
    else:
        resolved_samples, audit_rows = _resolve_dataset(
            config["dataset"],
            compound_format=config["compound_format"],
            use_split_column=config["use_split_column"],
            strict=config["strict"],
            state=state,
            metadata_by_id=metadata_by_id,
            compound_by_accession=compound_by_accession,
            pathway_names=pathway_names,
            pathway_name_index=pathway_name_index,
        )
    if not resolved_samples:
        raise SystemExit("No valid samples remained after resolution.")
    _assert_unique_sample_ids(resolved_samples)

    if config["dataset_mode"] == "custom_tsv" and config["auto_split"] and not config["use_split_column"]:
        resolved_samples = _assign_auto_split(resolved_samples, rng)

    train_samples = [sample for sample in resolved_samples if sample.split == "train"]
    test_samples = [sample for sample in resolved_samples if sample.split == "test"]
    if not train_samples or not test_samples:
        raise SystemExit("Need both train and test samples after split handling.")

    print("Computing truncated sample-level enrichment...", flush=True)
    train_enrichment_rows = _build_enrichment_rows(
        train_samples,
        split="train",
        pathway_names=pathway_names,
        pathway_to_tokens=pathway_to_tokens,
        universe_tokens=universe_tokens,
        structure_lookup=structure_lookup,
        parent_map=parent_map,
        rng=rng,
    )
    test_enrichment_rows = _build_enrichment_rows(
        test_samples,
        split="test",
        pathway_names=pathway_names,
        pathway_to_tokens=pathway_to_tokens,
        universe_tokens=universe_tokens,
        structure_lookup=structure_lookup,
        parent_map=parent_map,
        rng=rng,
    )
    if not train_enrichment_rows or not test_enrichment_rows:
        raise SystemExit("Training or test enrichment candidate generation produced no rows.")

    print("Deriving ranking pairs from enrichment candidates...", flush=True)
    train_rows = _pair_rows_from_enrichment_rows(train_enrichment_rows)
    test_rows = _pair_rows_from_enrichment_rows(test_enrichment_rows)

    feature_columns = _numeric_feature_columns(train_rows)
    model_kind, model = _train_model(train_rows, feature_columns)
    for row, score in zip(train_rows, _predict_scores(model_kind, model, train_rows, feature_columns), strict=False):
        row["ml_score"] = float(score)
    for row, score in zip(test_rows, _predict_scores(model_kind, model, test_rows, feature_columns), strict=False):
        row["ml_score"] = float(score)

    metrics, prediction_rows = _evaluate(
        test_rows,
        baseline_key="baseline_score",
        model_key="ml_score",
    )
    feature_importance = _feature_importance(model_kind, model, feature_columns, test_rows, rng)
    overlap = _shared_overlap(resolved_samples)
    evaluation_warning = (
        "WARNING: Default AraCyc mode evaluates on in-database compounds; metrics are optimistic."
        if config["dataset_mode"] == "default_aracyc"
        else ""
    )
    metrics["config"] = {
        "dataset_mode": config["dataset_mode"],
        "dataset": str(config["dataset"]) if config["dataset"] is not None else "",
        "compound_format": config["compound_format"],
        "use_split_column": config["use_split_column"],
        "auto_split": config["auto_split"],
        "strict": config["strict"],
        "train_ratio": DEFAULT_TRAIN_RATIO,
        "max_samples_per_pathway": config["max_samples_per_pathway"],
        "model_kind": model_kind,
        "fallback_used": model_kind != "lightgbm_lambdarank",
        "feature_columns": feature_columns,
    }
    metrics["overlap"] = overlap
    metrics["feature_importance"] = feature_importance
    metrics["pair_counts"] = {
        "train_pairs": len(train_rows),
        "test_pairs": len(test_rows),
    }
    metrics["evaluation_warning"] = evaluation_warning
    if generation_summary:
        metrics["generation_summary"] = generation_summary

    label_vocab = sorted({pathway_id for sample in resolved_samples for pathway_id in sample.pathway_ids})

    resolved_dataset_path = out_dir / "resolved_dataset.tsv"
    audit_path = out_dir / "dataset_audit.tsv"
    train_enrichment_path = out_dir / "train_enrichment.tsv"
    test_enrichment_path = out_dir / "test_enrichment.tsv"
    train_pairs_path = out_dir / "train_pairs.tsv"
    test_pairs_path = out_dir / "test_pairs.tsv"
    model_path = out_dir / "model.pkl"
    label_vocab_path = out_dir / "label_vocab.json"
    metrics_path = out_dir / "metrics.json"
    predictions_path = out_dir / "validation_predictions.tsv"

    _write_tsv(
        resolved_dataset_path,
        _resolved_rows(resolved_samples),
        [
            "sample_id",
            "compounds_raw",
            "compound_ids",
            "chebi_accessions",
            "chebi_names",
            "unresolved_count",
            "unresolved_tokens",
            "pathway_ids",
            "pathway_names",
            "split",
            "weight",
            "source",
            "note",
        ],
    )
    enrichment_fieldnames = _enrichment_fieldnames()
    _write_tsv(
        audit_path,
        _audit_rows(audit_rows),
        [
            "sample_id",
            "reason",
            "detail",
            "compounds_raw",
            "pathway_ids_raw",
            "pathway_names_raw",
            "split_raw",
        ],
    )
    pair_fieldnames = enrichment_fieldnames + ["ml_score"]
    _write_tsv(train_enrichment_path, train_enrichment_rows, enrichment_fieldnames)
    _write_tsv(test_enrichment_path, test_enrichment_rows, enrichment_fieldnames)
    _write_tsv(train_pairs_path, train_rows, pair_fieldnames)
    _write_tsv(test_pairs_path, test_rows, pair_fieldnames)
    _write_tsv(
        predictions_path,
        prediction_rows,
        pair_fieldnames + ["baseline_rank", "ml_rank"],
    )
    with model_path.open("wb") as handle:
        pickle.dump(
            {
                "model_kind": model_kind,
                "feature_columns": feature_columns,
                "model": model,
                "label_vocab": label_vocab,
            },
            handle,
        )
    label_vocab_path.write_text(json.dumps(label_vocab, indent=2), encoding="utf-8")
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"Original samples: {len(resolved_samples) + len(audit_rows)}", flush=True)
    print(f"Resolved samples kept: {len(resolved_samples)}", flush=True)
    reason_counts: dict[str, int] = defaultdict(int)
    for row in audit_rows:
        reason_counts[row.reason] += 1
    filtered_summary = ", ".join(f"{reason}={count}" for reason, count in sorted(reason_counts.items())) or "none"
    print(f"Filtered samples: {filtered_summary}", flush=True)
    split_mode = (
        "compound-disjoint automatic 80/20 (AraCyc default)"
        if config["dataset_mode"] == "default_aracyc"
        else ("file split" if config["use_split_column"] else "automatic 80/20")
    )
    print(f"Dataset mode: {config['dataset_mode']}", flush=True)
    print(f"Split mode: {split_mode}", flush=True)
    if evaluation_warning:
        print(evaluation_warning, flush=True)
    print(f"Train samples: {len(train_samples)}, test samples: {len(test_samples)}", flush=True)
    if generation_summary:
        print(
            f"Eligible AraCyc pathways: {generation_summary['eligible_pathways']}, "
            f"train compounds: {generation_summary['train_compounds']}, "
            f"test compounds: {generation_summary['test_compounds']}, "
            f"estimated pairs: {generation_summary['estimated_pair_count']}, "
            f"max_samples_per_pathway: {generation_summary['max_samples_per_pathway']}",
            flush=True,
        )
    print(
        f"Shared compounds: {overlap['shared_compounds']} "
        f"({overlap['shared_compound_ratio']:.4f} of test compounds)",
        flush=True,
    )
    print(
        f"Shared pathways: {overlap['shared_pathways']} "
        f"({overlap['shared_pathway_ratio']:.4f} of test pathways)",
        flush=True,
    )
    print(
        "Baseline metrics: "
        f"NDCG@5={metrics['baseline']['ndcg_at_5']:.4f}, "
        f"MRR={metrics['baseline']['mrr']:.4f}, "
        f"R@1={metrics['baseline']['recall_at_1']:.4f}, "
        f"R@3={metrics['baseline']['recall_at_3']:.4f}, "
        f"R@5={metrics['baseline']['recall_at_5']:.4f}",
        flush=True,
    )
    print(
        "ML metrics: "
        f"NDCG@5={metrics['ml']['ndcg_at_5']:.4f}, "
        f"MRR={metrics['ml']['mrr']:.4f}, "
        f"R@1={metrics['ml']['recall_at_1']:.4f}, "
        f"R@3={metrics['ml']['recall_at_3']:.4f}, "
        f"R@5={metrics['ml']['recall_at_5']:.4f}",
        flush=True,
    )
    print(f"Model kind: {model_kind}", flush=True)
    print(f"Fallback used: {model_kind != 'lightgbm_lambdarank'}", flush=True)
    print(f"Resolved dataset written: {resolved_dataset_path}", flush=True)
    print(f"Dataset audit written: {audit_path}", flush=True)
    print(f"Train enrichment written: {train_enrichment_path}", flush=True)
    print(f"Test enrichment written: {test_enrichment_path}", flush=True)
    print(f"Train pairs written: {train_pairs_path}", flush=True)
    print(f"Test pairs written: {test_pairs_path}", flush=True)
    print(f"Model written: {model_path}", flush=True)
    print(f"Label vocab written: {label_vocab_path}", flush=True)
    print(f"Metrics written: {metrics_path}", flush=True)
    print(f"Validation predictions written: {predictions_path}", flush=True)


if __name__ == "__main__":
    main()
