#!/usr/bin/env python3
"""Interactive TSV-driven training for compound-set -> pathway reranking.

This script covers the full model lifecycle for the compound-set pathway
reranker:

1. resolve or generate labeled compound-set samples
2. run sample-level baseline enrichment to build candidate pathways
3. train a LightGBM ranking model on those candidate rows
4. evaluate with single-split or cross-validation metrics
5. optionally compute SHAP explanations
6. optionally publish a deployable online model bundle

The online query path intentionally reuses the feature construction logic from
this script via `compound_set_rerank_shared.py` so the deployed model is scored
with the same columns and semantics used at training time.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import pickle
import random
import statistics
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import ndcg_score

try:
    import lightgbm as lgb
    LIGHTGBM_IMPORT_ERROR = ""
except Exception as exc:  # pragma: no cover - optional dependency
    lgb = None
    LIGHTGBM_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"

try:
    import shap
    SHAP_IMPORT_ERROR = ""
except Exception as exc:  # pragma: no cover - optional dependency
    shap = None
    SHAP_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"

from enrich_pathways import (
    CompoundTargetRecord,
    PathwayEnrichmentRow,
    _benjamini_hochberg,
    _confidence_level,
    _compute_enrichment,
    _load_pathway_membership,
)
from compound_set_rerank_shared import (
    ONLINE_CANDIDATE_POLICY,
    TRAINING_CANDIDATE_POLICY,
    feature_frame as shared_feature_frame,
    online_model_dir,
    query_level_features_from_compound_ids,
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
DEFAULT_CV_FOLDS = 5
DEFAULT_NEGATIVE_LIMIT = 30
DEFAULT_HARD_NEGATIVE_LIMIT = 20
DEFAULT_MAX_SUBSETS_PER_PATHWAY = 50
DEFAULT_PERMUTATION_IMPORTANCE_ROWS = 2000
DEFAULT_SHAP_TOP_FEATURES = 5


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
    """Parse CLI arguments for training, evaluation, and model publication."""

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
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=DEFAULT_CV_FOLDS,
        help="Number of cross-validation folds.",
    )
    parser.add_argument(
        "--no-cross-validation",
        action="store_true",
        help="Disable cross-validation and run a single train/test split.",
    )
    parser.add_argument(
        "--export-fold-pairs",
        action="store_true",
        help="In CV mode, also write train/test pair TSVs for each fold.",
    )
    parser.add_argument(
        "--export-fold-models",
        action="store_true",
        help="In CV mode, also write model.pkl and label_vocab.json for each fold.",
    )
    parser.add_argument(
        "--publish-online-model",
        action="store_true",
        help="After training completes, publish a deployable online rerank model trained on all available data.",
    )
    parser.add_argument(
        "--allow-sklearn-fallback",
        action="store_true",
        help="Allow sklearn debug fallback when LightGBM cannot be imported.",
    )
    parser.add_argument(
        "--no-shap",
        action="store_true",
        help="Disable SHAP explanations.",
    )
    parser.add_argument(
        "--export-shap-long",
        action="store_true",
        help="Also export per-feature SHAP long tables.",
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
    """Resolve the final run configuration from flags plus interactive prompts."""

    dataset_mode = "custom_tsv" if args.dataset else "default_aracyc"
    dataset: Path | None = Path(args.dataset).resolve() if args.dataset else None
    cv_root = workdir / "outputs" / "crossValidation"
    cross_validation = not args.no_cross_validation
    cv_disabled_reason = ""

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

        if use_split_column:
            cross_validation = False
            cv_disabled_reason = "TSV split column provided; using single file split."
        if not use_split_column and not cross_validation:
            auto_split = args.auto_split
            if not auto_split:
                auto_split = _ask_bool("Use automatic 80/20 train/test split", default=True)
            if not auto_split:
                raise SystemExit("No split strategy selected.")

    out_dir = None
    if args.out_dir:
        candidate = Path(args.out_dir).expanduser()
        out_dir = (cv_root / candidate).resolve() if not candidate.is_absolute() else candidate.resolve()
        if out_dir != cv_root and cv_root not in out_dir.parents:
            raise SystemExit(f"Output directory must be inside {cv_root}")
    if out_dir is None:
        default_name = f"{dataset.stem}_cv" if dataset is not None else "aracyc_default_cv"
        default_out = cv_root / default_name
        out_dir = Path(_ask_text("Output directory", default=str(default_out))).expanduser().resolve()
        if out_dir != cv_root and cv_root not in out_dir.parents:
            raise SystemExit(f"Output directory must be inside {cv_root}")

    return {
        "dataset_mode": dataset_mode,
        "dataset": dataset,
        "compound_format": compound_format,
        "use_split_column": use_split_column,
        "auto_split": auto_split,
        "strict": strict,
        "allow_sklearn_fallback": args.allow_sklearn_fallback,
        "shap_enabled": not args.no_shap,
        "export_shap_long": args.export_shap_long,
        "max_samples_per_pathway": max(1, int(args.max_samples_per_pathway)),
        "cv_folds": max(2, int(args.cv_folds)),
        "cross_validation": cross_validation,
        "cv_disabled_reason": cv_disabled_reason,
        "export_fold_pairs": args.export_fold_pairs,
        "export_fold_models": args.export_fold_models,
        "publish_online_model": args.publish_online_model,
        "out_dir": out_dir,
        "cv_root": cv_root,
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


def _lightgbm_requirement_message(import_error: str) -> str:
    lines = [
        "lightgbm is required for the default LGBMRanker training path but is not available.",
        f"Import error: {import_error}" if import_error else "Import error: unknown",
        "Install or repair it before running this script.",
        "Suggested command: pip install lightgbm",
        "If you only need a debug-only smoke test, rerun with --allow-sklearn-fallback.",
    ]
    if "libomp.dylib" in import_error:
        lines.append("macOS note: LightGBM also needs the OpenMP runtime; install libomp and ensure it is on the loader path.")
    return "\n".join(lines)


def _resolve_model_backend(*, allow_sklearn_fallback: bool) -> dict[str, Any]:
    """Choose the training backend and document any fallback behavior."""

    if lgb is not None:
        return {
            "model_kind": "lightgbm_lambdarank",
            "fallback_used": False,
            "comparable_to_lightgbm": True,
            "lightgbm_available": True,
            "lightgbm_import_error": "",
            "lightgbm_version": getattr(lgb, "__version__", ""),
            "warning": "",
        }

    import_error = LIGHTGBM_IMPORT_ERROR
    if allow_sklearn_fallback:
        warning = "WARNING: Using sklearn fallback. Results are not comparable to LGBMRanker."
        return {
            "model_kind": "sklearn_debug_fallback",
            "fallback_used": True,
            "comparable_to_lightgbm": False,
            "lightgbm_available": False,
            "lightgbm_import_error": import_error,
            "lightgbm_version": "",
            "warning": warning,
        }

    raise SystemExit(_lightgbm_requirement_message(import_error))


def _resolve_shap_backend(*, shap_enabled: bool, model_backend: dict[str, Any]) -> dict[str, Any]:
    """Validate whether SHAP can run for the selected backend and flags."""

    if not shap_enabled:
        return {
            "shap_enabled": False,
            "shap_available": shap is not None,
            "shap_version": getattr(shap, "__version__", "") if shap is not None else "",
            "shap_scope": "test_only",
            "shap_long_exported": False,
            "shap_reason": "disabled_by_flag",
        }

    if model_backend["model_kind"] != "lightgbm_lambdarank":
        return {
            "shap_enabled": False,
            "shap_available": shap is not None,
            "shap_version": getattr(shap, "__version__", "") if shap is not None else "",
            "shap_scope": "test_only",
            "shap_long_exported": False,
            "shap_reason": "unsupported_model_kind",
        }

    if shap is None:
        lines = [
            "SHAP explanations are enabled but the `shap` package is not available.",
            f"Import error: {SHAP_IMPORT_ERROR}" if SHAP_IMPORT_ERROR else "Import error: unknown",
            "Install it before running this script.",
            "Suggested command: pip install shap",
        ]
        raise SystemExit("\n".join(lines))

    return {
        "shap_enabled": True,
        "shap_available": True,
        "shap_version": getattr(shap, "__version__", ""),
        "shap_scope": "test_only",
        "shap_long_exported": False,
        "shap_reason": "",
    }


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
    """Resolve a user TSV into normalized, labeled training samples plus audit rows."""

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


def _clone_sample_with_split(sample: ResolvedSample, split: str) -> ResolvedSample:
    return ResolvedSample(
        sample_id=sample.sample_id,
        compounds_raw=sample.compounds_raw,
        pathway_ids_raw=sample.pathway_ids_raw,
        pathway_names_raw=sample.pathway_names_raw,
        split=split,
        weight=sample.weight,
        source=sample.source,
        note=sample.note,
        resolved_compounds=sample.resolved_compounds,
        unresolved_tokens=sample.unresolved_tokens,
        pathway_ids=sample.pathway_ids,
        pathway_names=sample.pathway_names,
    )


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


def _partition_units(units: list[str], fold_count: int, rng: random.Random) -> list[list[str]]:
    if len(units) < fold_count:
        raise SystemExit(f"Need at least {fold_count} units for {fold_count}-fold cross-validation; got {len(units)}.")
    shuffled = list(units)
    rng.shuffle(shuffled)
    folds: list[list[str]] = [[] for _ in range(fold_count)]
    for index, unit in enumerate(shuffled):
        folds[index % fold_count].append(unit)
    return folds


def _build_samples_from_test_ids(samples: list[ResolvedSample], test_ids: set[str]) -> list[ResolvedSample]:
    fold_samples: list[ResolvedSample] = []
    for sample in samples:
        fold_samples.append(_clone_sample_with_split(sample, "test" if sample.sample_id in test_ids else "train"))
    return fold_samples


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


def _generate_default_aracyc_dataset_for_compound_split(
    *,
    pathway_names: dict[str, str],
    pathway_to_tokens: dict[str, set[str]],
    metadata_by_id: dict[str, dict[str, str]],
    train_compounds: set[str],
    test_compounds: set[str],
    max_samples_per_pathway: int,
    rng: random.Random,
) -> tuple[list[ResolvedSample], dict[str, Any]]:
    """Generate train/test samples for one compound-disjoint AraCyc split."""

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
    return generated_samples, generation_summary


def _all_default_aracyc_compounds(
    *,
    pathway_to_tokens: dict[str, set[str]],
    metadata_by_id: dict[str, dict[str, str]],
) -> list[str]:
    return sorted(
        {
            token.split(":", 1)[1]
            for members in pathway_to_tokens.values()
            for token in members
            if token.startswith("cmp:") and token.split(":", 1)[1] in metadata_by_id
        }
    )


def _generate_default_aracyc_dataset(
    *,
    pathway_names: dict[str, str],
    pathway_to_tokens: dict[str, set[str]],
    metadata_by_id: dict[str, dict[str, str]],
    max_samples_per_pathway: int,
    rng: random.Random,
) -> tuple[list[ResolvedSample], list[AuditRow], dict[str, Any]]:
    all_compounds = _all_default_aracyc_compounds(
        pathway_to_tokens=pathway_to_tokens,
        metadata_by_id=metadata_by_id,
    )
    if len(all_compounds) < 2:
        raise SystemExit("AraCyc default dataset cannot be generated: fewer than 2 compounds are available.")

    shuffled_compounds = list(all_compounds)
    rng.shuffle(shuffled_compounds)
    train_count = _choose_train_count(len(shuffled_compounds))
    train_compounds = set(shuffled_compounds[:train_count])
    test_compounds = set(shuffled_compounds[train_count:])
    generated_samples, generation_summary = _generate_default_aracyc_dataset_for_compound_split(
        pathway_names=pathway_names,
        pathway_to_tokens=pathway_to_tokens,
        metadata_by_id=metadata_by_id,
        train_compounds=train_compounds,
        test_compounds=test_compounds,
        max_samples_per_pathway=max_samples_per_pathway,
        rng=rng,
    )
    return generated_samples, [], generation_summary


def _generate_default_aracyc_deploy_samples(
    *,
    pathway_names: dict[str, str],
    pathway_to_tokens: dict[str, set[str]],
    metadata_by_id: dict[str, dict[str, str]],
    max_samples_per_pathway: int,
    rng: random.Random,
) -> tuple[list[ResolvedSample], dict[str, Any]]:
    pathway_to_compounds: dict[str, set[str]] = {
        pathway_id: {
            token.split(":", 1)[1]
            for token in members
            if token.startswith("cmp:") and token.split(":", 1)[1] in metadata_by_id
        }
        for pathway_id, members in pathway_to_tokens.items()
    }
    eligible_pathways = {
        pathway_id: sorted(compounds)
        for pathway_id, compounds in pathway_to_compounds.items()
        if len(compounds) >= 2
    }
    if not eligible_pathways:
        raise SystemExit("AraCyc deploy dataset generation produced no eligible pathways.")

    compound_to_pathways: dict[str, set[str]] = defaultdict(set)
    for pathway_id, members in eligible_pathways.items():
        for compound_id in members:
            compound_to_pathways[compound_id].add(pathway_id)

    generated_samples: list[ResolvedSample] = []
    estimated_pair_count = 0
    for pathway_id, members in sorted(eligible_pathways.items()):
        subsets = _sample_subset_tuples(members, rng=rng, limit=max_samples_per_pathway)
        for subset_index, subset in enumerate(subsets, start=1):
            positive_ids = sorted(set.intersection(*(compound_to_pathways[compound_id] for compound_id in subset)))
            if not positive_ids:
                continue
            resolved_compounds: list[ResolvedCompound] = []
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
                    sample_id=f"aracyc_deploy_{pathway_id}_{subset_index:03d}",
                    compounds_raw="; ".join(item.chebi_name for item in resolved_compounds),
                    pathway_ids_raw=";".join(positive_ids),
                    pathway_names_raw="; ".join(positive_names),
                    split="train",
                    weight=1.0,
                    source="aracyc_generated_deploy",
                    note=f"seed_pathway={pathway_id}",
                    resolved_compounds=tuple(resolved_compounds),
                    unresolved_tokens=(),
                    pathway_ids=tuple(positive_ids),
                    pathway_names=positive_names,
                )
            )
            estimated_pair_count += len(positive_ids) + DEFAULT_NEGATIVE_LIMIT

    if not generated_samples:
        raise SystemExit("AraCyc deploy dataset generation produced no samples.")

    generation_summary = {
        "eligible_pathways": len(eligible_pathways),
        "generated_train_samples": len(generated_samples),
        "estimated_pair_count": estimated_pair_count,
        "max_samples_per_pathway": max_samples_per_pathway,
        "training_data_scope": "all_data",
    }
    return generated_samples, generation_summary


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
    return query_level_features_from_compound_ids(
        [item.compound_id for item in sample.resolved_compounds],
        structure_lookup=structure_lookup,
        parent_map=parent_map,
        root_cache=root_cache,
    )


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
    """Build the truncated sample-level enrichment table consumed by model training.

    Each row represents one `(sample_id, candidate_pathway)` pair with:

    - baseline enrichment statistics
    - mapping-quality features
    - query-level chemistry features

    The candidate policy deliberately mirrors the earlier ranking setup:
    positive labels + top hard negatives + random negatives up to a fixed cap.
    """

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

        # Keep the training candidate policy explicit and stable so offline
        # metrics and published metadata can describe exactly what the model saw.
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


def _feature_frame(rows: list[dict[str, Any]], feature_columns: list[str]) -> pd.DataFrame:
    return shared_feature_frame(rows, feature_columns)


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
    *,
    model_kind: str,
) -> Any:
    """Fit the ranking model (or explicit debug fallback) on pair rows."""

    X = _feature_frame(train_rows, feature_columns)
    y = np.asarray([int(row["label"]) for row in train_rows], dtype=int)
    base_sample_weight = np.asarray([float(row["sample_weight"]) for row in train_rows], dtype=float)

    if model_kind == "lightgbm_lambdarank":
        order = sorted(range(len(train_rows)), key=lambda idx: train_rows[idx]["sample_id"])
        X_rank = X.iloc[order]
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
        return model

    positives = int(y.sum())
    negatives = len(y) - positives
    pos_scale = (negatives / positives) if positives and negatives else 1.0
    sample_weight = np.asarray([
        weight * (pos_scale if label == 1 else 1.0) for weight, label in zip(base_sample_weight, y, strict=False)
    ])
    from sklearn.ensemble import HistGradientBoostingClassifier

    model = HistGradientBoostingClassifier(
        random_state=RANDOM_SEED,
        max_depth=6,
        learning_rate=0.08,
        max_iter=200,
    )
    model.fit(X, y, sample_weight=sample_weight)
    return model


def _predict_scores(model_kind: str, model: Any, rows: list[dict[str, Any]], feature_columns: list[str]) -> np.ndarray:
    X = _feature_frame(rows, feature_columns)
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
        from sklearn.inspection import permutation_importance

        sample_rows = list(reference_rows)
        if len(sample_rows) > DEFAULT_PERMUTATION_IMPORTANCE_ROWS:
            sample_rows = rng.sample(sample_rows, DEFAULT_PERMUTATION_IMPORTANCE_ROWS)
        X = _feature_frame(sample_rows, feature_columns)
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


def _normalize_expected_value(value: Any) -> float:
    if isinstance(value, (list, tuple, np.ndarray)):
        arr = np.asarray(value, dtype=float).reshape(-1)
        return float(arr[0]) if len(arr) else 0.0
    return float(value)


def _json_feature_contributions(
    feature_columns: list[str],
    feature_values: np.ndarray,
    shap_values: np.ndarray,
    *,
    positive: bool,
    limit: int = DEFAULT_SHAP_TOP_FEATURES,
) -> str:
    pairs: list[dict[str, Any]] = []
    for feature, feature_value, shap_value in zip(feature_columns, feature_values, shap_values, strict=False):
        if positive and shap_value <= 0:
            continue
        if not positive and shap_value >= 0:
            continue
        pairs.append(
            {
                "feature": feature,
                "feature_value": float(feature_value),
                "shap_value": float(shap_value),
            }
        )
    pairs.sort(key=lambda item: item["shap_value"], reverse=positive)
    return json.dumps(pairs[:limit], ensure_ascii=True)


def _compute_shap_outputs(
    *,
    model_kind: str,
    model: Any,
    rows: list[dict[str, Any]],
    feature_columns: list[str],
    export_long: bool,
) -> dict[str, Any]:
    if not rows:
        return {
            "summary_rows": [],
            "row_outputs": [],
            "long_rows": [],
            "metadata": {
                "shap_enabled": False,
                "shap_available": shap is not None,
                "shap_version": getattr(shap, "__version__", "") if shap is not None else "",
                "shap_scope": "test_only",
                "shap_long_exported": export_long,
                "shap_rows": 0,
                "shap_reason": "no_rows",
            },
        }

    if model_kind != "lightgbm_lambdarank":
        return {
            "summary_rows": [],
            "row_outputs": [],
            "long_rows": [],
            "metadata": {
                "shap_enabled": False,
                "shap_available": shap is not None,
                "shap_version": getattr(shap, "__version__", "") if shap is not None else "",
                "shap_scope": "test_only",
                "shap_long_exported": export_long,
                "shap_rows": 0,
                "shap_reason": "unsupported_model_kind",
            },
        }

    X = _feature_frame(rows, feature_columns)
    tree_model = model.booster_ if hasattr(model, "booster_") else model
    explainer = shap.TreeExplainer(tree_model, model_output="raw")
    shap_values_raw = explainer.shap_values(X, check_additivity=False)
    if isinstance(shap_values_raw, list):
        shap_matrix = np.asarray(shap_values_raw[0], dtype=float)
    else:
        shap_matrix = np.asarray(shap_values_raw, dtype=float)
    expected_value = _normalize_expected_value(explainer.expected_value)
    feature_values = X.to_numpy(dtype=float)

    mean_abs = np.mean(np.abs(shap_matrix), axis=0)
    mean_signed = np.mean(shap_matrix, axis=0)
    positive_fraction = np.mean(shap_matrix > 0, axis=0)
    negative_fraction = np.mean(shap_matrix < 0, axis=0)
    ranked_summary = sorted(
        zip(feature_columns, mean_abs, mean_signed, positive_fraction, negative_fraction, strict=False),
        key=lambda item: float(item[1]),
        reverse=True,
    )
    summary_rows = [
        {
            "feature": feature,
            "mean_abs_shap": float(mean_abs_value),
            "mean_shap": float(mean_shap_value),
            "positive_fraction": float(pos_fraction),
            "negative_fraction": float(neg_fraction),
            "rank": rank,
        }
        for rank, (feature, mean_abs_value, mean_shap_value, pos_fraction, neg_fraction) in enumerate(
            ranked_summary,
            start=1,
        )
    ]

    row_outputs: list[dict[str, Any]] = []
    long_rows: list[dict[str, Any]] = []
    for row, row_feature_values, row_shap_values in zip(rows, feature_values, shap_matrix, strict=False):
        shap_sum = float(np.sum(row_shap_values))
        reconstructed_score = float(expected_value + shap_sum)
        ml_score = float(row["ml_score"])
        row_outputs.append(
            {
                "sample_id": row["sample_id"],
                "pathway_id": row["pathway_id"],
                "pathway_name": row["pathway_name"],
                "label": row["label"],
                "baseline_score": row["baseline_score"],
                "ml_score": ml_score,
                "baseline_rank": row["baseline_rank"],
                "ml_rank": row["ml_rank"],
                "expected_value": expected_value,
                "shap_sum": shap_sum,
                "reconstructed_score": reconstructed_score,
                "reconstruction_error": abs(reconstructed_score - ml_score),
                "top_positive_features_json": _json_feature_contributions(
                    feature_columns,
                    row_feature_values,
                    row_shap_values,
                    positive=True,
                ),
                "top_negative_features_json": _json_feature_contributions(
                    feature_columns,
                    row_feature_values,
                    row_shap_values,
                    positive=False,
                ),
            }
        )
        if export_long:
            for feature, feature_value, shap_value in zip(feature_columns, row_feature_values, row_shap_values, strict=False):
                long_rows.append(
                    {
                        "sample_id": row["sample_id"],
                        "pathway_id": row["pathway_id"],
                        "pathway_name": row["pathway_name"],
                        "feature": feature,
                        "feature_value": float(feature_value),
                        "shap_value": float(shap_value),
                        "abs_shap": abs(float(shap_value)),
                        "direction": "positive" if shap_value > 0 else ("negative" if shap_value < 0 else "neutral"),
                    }
                )

    return {
        "summary_rows": summary_rows,
        "row_outputs": row_outputs,
        "long_rows": long_rows,
        "metadata": {
            "shap_enabled": True,
            "shap_available": True,
            "shap_version": getattr(shap, "__version__", ""),
            "shap_scope": "test_only",
            "shap_long_exported": export_long,
            "shap_rows": len(row_outputs),
            "shap_reason": "",
        },
    }


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
    """Evaluate baseline and ML ranking quality on held-out candidate rows."""

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


def _directory_size_bytes(path: Path) -> int:
    total = 0
    for file_path in path.rglob("*"):
        if file_path.is_file():
            total += file_path.stat().st_size
    return total


def _flatten_fold_metrics(
    *,
    fold_id: str,
    metrics: dict[str, Any],
    overlap: dict[str, float],
    sample_counts: dict[str, int],
    enrichment_counts: dict[str, int],
    pair_counts: dict[str, int],
    generation_summary: dict[str, Any],
    output_dir: Path,
    model_backend: dict[str, Any],
    shap_metadata: dict[str, Any],
) -> dict[str, Any]:
    row = {
        "fold_id": fold_id,
        "baseline_ndcg_at_5": metrics["baseline"]["ndcg_at_5"],
        "baseline_mrr": metrics["baseline"]["mrr"],
        "baseline_recall_at_1": metrics["baseline"]["recall_at_1"],
        "baseline_recall_at_3": metrics["baseline"]["recall_at_3"],
        "baseline_recall_at_5": metrics["baseline"]["recall_at_5"],
        "ml_ndcg_at_5": metrics["ml"]["ndcg_at_5"],
        "ml_mrr": metrics["ml"]["mrr"],
        "ml_recall_at_1": metrics["ml"]["recall_at_1"],
        "ml_recall_at_3": metrics["ml"]["recall_at_3"],
        "ml_recall_at_5": metrics["ml"]["recall_at_5"],
        "ndcg_at_5_delta": metrics["ml"]["ndcg_at_5"] - metrics["baseline"]["ndcg_at_5"],
        "mrr_delta": metrics["ml"]["mrr"] - metrics["baseline"]["mrr"],
        "recall_at_1_delta": metrics["ml"]["recall_at_1"] - metrics["baseline"]["recall_at_1"],
        "recall_at_3_delta": metrics["ml"]["recall_at_3"] - metrics["baseline"]["recall_at_3"],
        "recall_at_5_delta": metrics["ml"]["recall_at_5"] - metrics["baseline"]["recall_at_5"],
        "samples_evaluated": metrics["samples_evaluated"],
        "train_samples": sample_counts["train"],
        "test_samples": sample_counts["test"],
        "train_enrichment_rows": enrichment_counts["train"],
        "test_enrichment_rows": enrichment_counts["test"],
        "train_pairs": pair_counts["train_pairs"],
        "test_pairs": pair_counts["test_pairs"],
        "train_compounds": overlap["train_compounds"],
        "test_compounds": overlap["test_compounds"],
        "shared_compounds": overlap["shared_compounds"],
        "shared_compound_ratio": overlap["shared_compound_ratio"],
        "train_pathways": overlap["train_pathways"],
        "test_pathways": overlap["test_pathways"],
        "shared_pathways": overlap["shared_pathways"],
        "shared_pathway_ratio": overlap["shared_pathway_ratio"],
        "model_kind": model_backend["model_kind"],
        "fallback_used": model_backend["fallback_used"],
        "comparable_to_lightgbm": model_backend["comparable_to_lightgbm"],
        "lightgbm_available": model_backend["lightgbm_available"],
        "lightgbm_version": model_backend["lightgbm_version"],
        "lightgbm_import_error": model_backend["lightgbm_import_error"],
        "shap_enabled": shap_metadata["shap_enabled"],
        "shap_available": shap_metadata["shap_available"],
        "shap_version": shap_metadata["shap_version"],
        "shap_scope": shap_metadata["shap_scope"],
        "shap_long_exported": shap_metadata["shap_long_exported"],
        "shap_rows": shap_metadata["shap_rows"],
        "shap_reason": shap_metadata["shap_reason"],
        "output_dir": str(output_dir),
    }
    for key, value in generation_summary.items():
        row[key] = value
    return row


def _cv_summary_rows(fold_metric_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metric_names = [
        "baseline_ndcg_at_5",
        "baseline_mrr",
        "baseline_recall_at_1",
        "baseline_recall_at_3",
        "baseline_recall_at_5",
        "ml_ndcg_at_5",
        "ml_mrr",
        "ml_recall_at_1",
        "ml_recall_at_3",
        "ml_recall_at_5",
        "ndcg_at_5_delta",
        "mrr_delta",
        "recall_at_1_delta",
        "recall_at_3_delta",
        "recall_at_5_delta",
    ]
    rows: list[dict[str, Any]] = []
    for metric_name in metric_names:
        values = [float(row[metric_name]) for row in fold_metric_rows]
        rows.append(
            {
                "metric": metric_name,
                "mean": float(statistics.mean(values)),
                "std": float(statistics.stdev(values)) if len(values) > 1 else 0.0,
                "min": float(min(values)),
                "max": float(max(values)),
            }
        )
    return rows


def _run_training_split(
    *,
    samples: list[ResolvedSample],
    out_dir: Path,
    dataset_mode: str,
    state: dict[str, Any],
    pathway_names: dict[str, str],
    pathway_to_tokens: dict[str, set[str]],
    universe_tokens: set[str],
    structure_lookup: dict[str, Any],
    parent_map: dict[str, str],
    rng: random.Random,
    generation_summary: dict[str, Any],
    model_backend: dict[str, Any],
    shap_backend: dict[str, Any],
    write_pairs: bool,
    write_model: bool,
    fold_id: str,
) -> dict[str, Any]:
    """Run one complete train/evaluate/export cycle for a split or one CV fold."""

    train_samples = [sample for sample in samples if sample.split == "train"]
    test_samples = [sample for sample in samples if sample.split == "test"]
    if not train_samples or not test_samples:
        raise SystemExit("Need both train and test samples after split handling.")

    print(f"[{fold_id}] Computing truncated sample-level enrichment...", flush=True)
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

    # The pair tables are derived directly from the persisted enrichment rows so
    # training never silently recomputes a different candidate set.
    print(f"[{fold_id}] Deriving ranking pairs from enrichment candidates...", flush=True)
    train_rows = _pair_rows_from_enrichment_rows(train_enrichment_rows)
    test_rows = _pair_rows_from_enrichment_rows(test_enrichment_rows)

    feature_columns = _numeric_feature_columns(train_rows)
    model_kind = str(model_backend["model_kind"])
    model = _train_model(train_rows, feature_columns, model_kind=model_kind)
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
    shap_outputs = _compute_shap_outputs(
        model_kind=model_kind,
        model=model,
        rows=prediction_rows,
        feature_columns=feature_columns,
        export_long=bool(shap_backend["shap_long_exported"]),
    ) if shap_backend["shap_enabled"] else {
        "summary_rows": [],
        "row_outputs": [],
        "long_rows": [],
        "metadata": {
            "shap_enabled": False,
            "shap_available": shap_backend["shap_available"],
            "shap_version": shap_backend["shap_version"],
            "shap_scope": shap_backend["shap_scope"],
            "shap_long_exported": shap_backend["shap_long_exported"],
            "shap_rows": 0,
            "shap_reason": shap_backend["shap_reason"],
        },
    }
    shap_metadata = shap_outputs["metadata"]
    overlap = _shared_overlap(samples)
    evaluation_warning = (
        "WARNING: Default AraCyc mode evaluates on in-database compounds; metrics are optimistic."
        if dataset_mode == "default_aracyc"
        else ""
    )
    pair_counts = {
        "train_pairs": len(train_rows),
        "test_pairs": len(test_rows),
    }
    sample_counts = {
        "train": len(train_samples),
        "test": len(test_samples),
    }
    enrichment_counts = {
        "train": len(train_enrichment_rows),
        "test": len(test_enrichment_rows),
    }
    metrics["config"] = {
        "dataset_mode": dataset_mode,
        "train_ratio": DEFAULT_TRAIN_RATIO,
        "model_kind": model_kind,
        "fallback_used": model_backend["fallback_used"],
        "comparable_to_lightgbm": model_backend["comparable_to_lightgbm"],
        "lightgbm_available": model_backend["lightgbm_available"],
        "lightgbm_version": model_backend["lightgbm_version"],
        "feature_columns": feature_columns,
        "fold_id": fold_id,
    }
    if model_backend["lightgbm_import_error"]:
        metrics["config"]["lightgbm_import_error"] = model_backend["lightgbm_import_error"]
    metrics["overlap"] = overlap
    metrics["feature_importance"] = feature_importance
    metrics["pair_counts"] = pair_counts
    metrics["sample_counts"] = sample_counts
    metrics["enrichment_counts"] = enrichment_counts
    metrics["evaluation_warning"] = evaluation_warning
    metrics["model_kind"] = model_kind
    metrics["fallback_used"] = model_backend["fallback_used"]
    metrics["comparable_to_lightgbm"] = model_backend["comparable_to_lightgbm"]
    metrics["lightgbm_available"] = model_backend["lightgbm_available"]
    metrics["lightgbm_version"] = model_backend["lightgbm_version"]
    if model_backend["lightgbm_import_error"]:
        metrics["lightgbm_import_error"] = model_backend["lightgbm_import_error"]
    metrics["shap_enabled"] = shap_metadata["shap_enabled"]
    metrics["shap_available"] = shap_metadata["shap_available"]
    metrics["shap_version"] = shap_metadata["shap_version"]
    metrics["shap_scope"] = shap_metadata["shap_scope"]
    metrics["shap_long_exported"] = shap_metadata["shap_long_exported"]
    metrics["shap_rows"] = shap_metadata["shap_rows"]
    metrics["shap_reason"] = shap_metadata["shap_reason"]
    if generation_summary:
        metrics["generation_summary"] = generation_summary

    label_vocab = sorted({pathway_id for sample in samples for pathway_id in sample.pathway_ids})
    out_dir.mkdir(parents=True, exist_ok=True)
    resolved_dataset_path = out_dir / "resolved_dataset.tsv"
    train_enrichment_path = out_dir / "train_enrichment.tsv"
    test_enrichment_path = out_dir / "test_enrichment.tsv"
    predictions_path = out_dir / "validation_predictions.tsv"
    shap_summary_path = out_dir / "shap_feature_importance.tsv"
    shap_rows_path = out_dir / "test_shap_rows.tsv"
    shap_long_path = out_dir / "test_shap_long.tsv"
    metrics_path = out_dir / "metrics.json"

    _write_tsv(
        resolved_dataset_path,
        _resolved_rows(samples),
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
    _write_tsv(train_enrichment_path, train_enrichment_rows, enrichment_fieldnames)
    _write_tsv(test_enrichment_path, test_enrichment_rows, enrichment_fieldnames)

    pair_fieldnames = enrichment_fieldnames + ["ml_score"]
    if write_pairs:
        _write_tsv(out_dir / "train_pairs.tsv", train_rows, pair_fieldnames)
        _write_tsv(out_dir / "test_pairs.tsv", test_rows, pair_fieldnames)
    _write_tsv(
        predictions_path,
        prediction_rows,
        pair_fieldnames + ["baseline_rank", "ml_rank"],
    )
    if shap_metadata["shap_enabled"]:
        _write_tsv(
            shap_summary_path,
            shap_outputs["summary_rows"],
            ["feature", "mean_abs_shap", "mean_shap", "positive_fraction", "negative_fraction", "rank"],
        )
        _write_tsv(
            shap_rows_path,
            shap_outputs["row_outputs"],
            [
                "sample_id",
                "pathway_id",
                "pathway_name",
                "label",
                "baseline_score",
                "ml_score",
                "baseline_rank",
                "ml_rank",
                "expected_value",
                "shap_sum",
                "reconstructed_score",
                "reconstruction_error",
                "top_positive_features_json",
                "top_negative_features_json",
            ],
        )
        if shap_metadata["shap_long_exported"]:
            _write_tsv(
                shap_long_path,
                shap_outputs["long_rows"],
                [
                    "sample_id",
                    "pathway_id",
                    "pathway_name",
                    "feature",
                    "feature_value",
                    "shap_value",
                    "abs_shap",
                    "direction",
                ],
            )
    if write_model:
        with (out_dir / "model.pkl").open("wb") as handle:
            pickle.dump(
                {
                    "model_kind": model_kind,
                    "feature_columns": feature_columns,
                    "model": model,
                    "label_vocab": label_vocab,
                },
                handle,
            )
        (out_dir / "label_vocab.json").write_text(json.dumps(label_vocab, indent=2), encoding="utf-8")
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    return {
        "metrics": metrics,
        "prediction_rows": prediction_rows,
        "test_enrichment_rows": test_enrichment_rows,
        "resolved_rows": _resolved_rows(samples),
        "fold_metric_row": _flatten_fold_metrics(
            fold_id=fold_id,
            metrics=metrics,
            overlap=overlap,
            sample_counts=sample_counts,
            enrichment_counts=enrichment_counts,
            pair_counts=pair_counts,
            generation_summary=generation_summary,
            output_dir=out_dir,
            model_backend=model_backend,
            shap_metadata=shap_metadata,
        ),
        "evaluation_warning": evaluation_warning,
        "out_dir": out_dir,
        "shap_summary_rows": shap_outputs["summary_rows"],
        "shap_row_outputs": shap_outputs["row_outputs"],
        "shap_long_rows": shap_outputs["long_rows"],
    }


def _build_default_cv_samples(
    *,
    pathway_names: dict[str, str],
    pathway_to_tokens: dict[str, set[str]],
    metadata_by_id: dict[str, dict[str, str]],
    max_samples_per_pathway: int,
    cv_folds: int,
    rng: random.Random,
) -> list[tuple[str, list[ResolvedSample], dict[str, Any]]]:
    all_compounds = _all_default_aracyc_compounds(
        pathway_to_tokens=pathway_to_tokens,
        metadata_by_id=metadata_by_id,
    )
    if len(all_compounds) < 2:
        raise SystemExit("AraCyc default dataset cannot be generated: fewer than 2 compounds are available.")
    compound_folds = _partition_units(all_compounds, cv_folds, rng)

    results: list[tuple[str, list[ResolvedSample], dict[str, Any]]] = []
    for fold_index, test_compounds in enumerate(compound_folds, start=1):
        test_compound_set = set(test_compounds)
        train_compound_set = set(all_compounds) - test_compound_set
        fold_samples, generation_summary = _generate_default_aracyc_dataset_for_compound_split(
            pathway_names=pathway_names,
            pathway_to_tokens=pathway_to_tokens,
            metadata_by_id=metadata_by_id,
            train_compounds=train_compound_set,
            test_compounds=test_compound_set,
            max_samples_per_pathway=max_samples_per_pathway,
            rng=rng,
        )
        if not fold_samples:
            raise SystemExit(f"Fold {fold_index:02d} produced no samples.")
        _assert_unique_sample_ids(fold_samples)
        results.append((f"fold_{fold_index:02d}", fold_samples, generation_summary))
    return results


def _build_custom_cv_samples(
    samples: list[ResolvedSample],
    *,
    cv_folds: int,
    rng: random.Random,
) -> list[tuple[str, list[ResolvedSample], dict[str, Any]]]:
    sample_ids = [sample.sample_id for sample in samples]
    sample_folds = _partition_units(sample_ids, cv_folds, rng)
    results: list[tuple[str, list[ResolvedSample], dict[str, Any]]] = []
    for fold_index, test_ids in enumerate(sample_folds, start=1):
        fold_samples = _build_samples_from_test_ids(samples, set(test_ids))
        _assert_unique_sample_ids(fold_samples)
        results.append((f"fold_{fold_index:02d}", fold_samples, {}))
    return results


def _publish_online_model(
    *,
    workdir: Path,
    run_dir: Path,
    dataset_mode: str,
    resolved_samples: list[ResolvedSample],
    pathway_names: dict[str, str],
    pathway_to_tokens: dict[str, set[str]],
    universe_tokens: set[str],
    structure_lookup: dict[str, Any],
    parent_map: dict[str, str],
    metadata_by_id: dict[str, dict[str, str]],
    max_samples_per_pathway: int,
    model_backend: dict[str, Any],
) -> Path:
    """Train and publish the deployable online rerank model.

    This intentionally retrains on all available data. The resulting deploy
    model is therefore not identical to any single CV fold model, so the
    published metadata records that CV metrics are reference-only.
    """

    if model_backend["model_kind"] != "lightgbm_lambdarank":
        raise SystemExit("Cannot publish online model from sklearn_debug_fallback; publish requires lightgbm_lambdarank.")

    publish_rng = random.Random(RANDOM_SEED)
    if dataset_mode == "default_aracyc":
        deploy_samples, publish_generation_summary = _generate_default_aracyc_deploy_samples(
            pathway_names=pathway_names,
            pathway_to_tokens=pathway_to_tokens,
            metadata_by_id=metadata_by_id,
            max_samples_per_pathway=max_samples_per_pathway,
            rng=publish_rng,
        )
    else:
        deploy_samples = [_clone_sample_with_split(sample, "train") for sample in resolved_samples]
        publish_generation_summary = {
            "generated_train_samples": len(deploy_samples),
            "training_data_scope": "all_data",
        }

    deploy_enrichment_rows = _build_enrichment_rows(
        deploy_samples,
        split="train",
        pathway_names=pathway_names,
        pathway_to_tokens=pathway_to_tokens,
        universe_tokens=universe_tokens,
        structure_lookup=structure_lookup,
        parent_map=parent_map,
        rng=publish_rng,
    )
    if not deploy_enrichment_rows:
        raise SystemExit("Deploy model publication failed: no training enrichment rows were generated.")

    deploy_rows = _pair_rows_from_enrichment_rows(deploy_enrichment_rows)
    feature_columns = _numeric_feature_columns(deploy_rows)
    deploy_model = _train_model(deploy_rows, feature_columns, model_kind=model_backend["model_kind"])
    label_vocab = sorted({pathway_id for sample in deploy_samples for pathway_id in sample.pathway_ids})

    model_dir = online_model_dir(workdir)
    model_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "model_kind": model_backend["model_kind"],
        "feature_columns": feature_columns,
        "dataset_mode": dataset_mode,
        "lightgbm_version": model_backend["lightgbm_version"],
        "candidate_policy": TRAINING_CANDIDATE_POLICY,
        "online_candidate_policy": ONLINE_CANDIDATE_POLICY,
        "published_from_run": str(run_dir),
        "training_data_scope": "all_data",
        "evaluation_note": "deploy_model_trained_on_all_data_cv_metrics_reference_only",
        "max_samples_per_pathway": max_samples_per_pathway,
        "samples_published": len(deploy_samples),
        "enrichment_rows_published": len(deploy_enrichment_rows),
        "training_rows_published": len(deploy_rows),
        "generation_summary": publish_generation_summary,
    }

    with (model_dir / "model.pkl").open("wb") as handle:
        pickle.dump(
            {
                "model_kind": model_backend["model_kind"],
                "feature_columns": feature_columns,
                "model": deploy_model,
                "label_vocab": label_vocab,
            },
            handle,
        )
    (model_dir / "label_vocab.json").write_text(json.dumps(label_vocab, indent=2), encoding="utf-8")
    (model_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("Published deploy model (trained on full data). CV metrics are reference only.", flush=True)
    print(f"Online model directory: {model_dir}", flush=True)
    print(f"Online candidate policy: {ONLINE_CANDIDATE_POLICY}", flush=True)
    return model_dir


def main() -> None:
    """Entry point for training, evaluation, CV aggregation, and publication."""

    args = parse_args()
    workdir = Path(args.workdir).resolve()
    config = _interactive_config(args, workdir)
    model_backend = _resolve_model_backend(allow_sklearn_fallback=config["allow_sklearn_fallback"])
    shap_backend = _resolve_shap_backend(shap_enabled=config["shap_enabled"], model_backend=model_backend)
    if config["publish_online_model"] and model_backend["model_kind"] != "lightgbm_lambdarank":
        raise SystemExit("Cannot publish online model from sklearn_debug_fallback; publish requires lightgbm_lambdarank.")

    rng = random.Random(RANDOM_SEED)
    run_dir: Path = config["out_dir"]
    run_dir.mkdir(parents=True, exist_ok=True)

    if model_backend["warning"]:
        print(model_backend["warning"], flush=True)
    if model_backend["lightgbm_available"]:
        print(
            f"Using LightGBM LGBMRanker (version {model_backend['lightgbm_version']}).",
            flush=True,
        )
    if shap_backend["shap_enabled"]:
        print(f"Using SHAP explanations (version {shap_backend['shap_version']}).", flush=True)
    elif shap_backend["shap_reason"] and shap_backend["shap_reason"] != "disabled_by_flag":
        print(f"SHAP disabled: {shap_backend['shap_reason']}", flush=True)

    print("Loading query/enrichment state...", flush=True)
    state = load_preprocessed_state(workdir, verbose=True)
    pathway_index_path = workdir / "outputs" / "preprocessed" / "aracyc_compound_pathway_index.tsv"
    pathway_names, pathway_to_tokens, universe_tokens = _load_pathway_membership(pathway_index_path)
    pathway_name_index = _load_pathway_name_index(pathway_names)
    metadata_by_id, compound_by_accession = _read_step1_metadata(workdir)
    parent_map = _load_parent_map(workdir)
    structure_lookup = ensure_chebi_structure_lookup(state, verbose=False)

    generation_summary: dict[str, Any] = {}
    audit_rows: list[AuditRow] = []
    resolved_samples: list[ResolvedSample] = []
    if config["dataset_mode"] == "default_aracyc":
        if not config["cross_validation"]:
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
    if config["dataset_mode"] == "custom_tsv" and not resolved_samples:
        raise SystemExit("No valid samples remained after resolution.")
    if config["dataset_mode"] == "custom_tsv":
        _assert_unique_sample_ids(resolved_samples)

    # Single-split mode is used for explicit train/test runs, smoke tests, and
    # the small training jobs that precede online model publication.
    if not config["cross_validation"]:
        if config["dataset_mode"] == "custom_tsv" and config["auto_split"] and not config["use_split_column"]:
            resolved_samples = _assign_auto_split(resolved_samples, rng)
        if not resolved_samples:
            raise SystemExit("No valid samples remained after resolution.")

        single_split_dir = run_dir / "single_split"
        result = _run_training_split(
            samples=resolved_samples,
            out_dir=single_split_dir,
            dataset_mode=config["dataset_mode"],
            state=state,
            pathway_names=pathway_names,
            pathway_to_tokens=pathway_to_tokens,
            universe_tokens=universe_tokens,
            structure_lookup=structure_lookup,
            parent_map=parent_map,
            rng=rng,
            generation_summary=generation_summary,
            model_backend=model_backend,
            shap_backend={**shap_backend, "shap_long_exported": config["export_shap_long"]},
            write_pairs=True,
            write_model=True,
            fold_id="single_split",
        )
        audit_path = run_dir / "dataset_audit.tsv"
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
        print(f"Dataset mode: {config['dataset_mode']}", flush=True)
        if config["cv_disabled_reason"]:
            print(config["cv_disabled_reason"], flush=True)
        print(f"Split mode: {'file split' if config['use_split_column'] else 'automatic 80/20'}", flush=True)
        print(f"Model kind: {result['metrics']['model_kind']}", flush=True)
        print(f"Comparable to LGBMRanker: {result['metrics']['comparable_to_lightgbm']}", flush=True)
        if result["metrics"].get("lightgbm_version"):
            print(f"LightGBM version: {result['metrics']['lightgbm_version']}", flush=True)
        if result["metrics"].get("lightgbm_import_error"):
            print(f"LightGBM import error: {result['metrics']['lightgbm_import_error']}", flush=True)
        print(f"SHAP enabled: {result['metrics']['shap_enabled']}", flush=True)
        if result["metrics"].get("shap_version"):
            print(f"SHAP version: {result['metrics']['shap_version']}", flush=True)
        if result["metrics"].get("shap_reason"):
            print(f"SHAP reason: {result['metrics']['shap_reason']}", flush=True)
        if result["evaluation_warning"]:
            print(result["evaluation_warning"], flush=True)
        print(f"Run output directory: {run_dir}", flush=True)
        print(f"Single split directory: {single_split_dir}", flush=True)
        print(
            "Baseline metrics: "
            f"NDCG@5={result['metrics']['baseline']['ndcg_at_5']:.4f}, "
            f"MRR={result['metrics']['baseline']['mrr']:.4f}, "
            f"R@1={result['metrics']['baseline']['recall_at_1']:.4f}, "
            f"R@3={result['metrics']['baseline']['recall_at_3']:.4f}, "
            f"R@5={result['metrics']['baseline']['recall_at_5']:.4f}",
            flush=True,
        )
        print(
            "ML metrics: "
            f"NDCG@5={result['metrics']['ml']['ndcg_at_5']:.4f}, "
            f"MRR={result['metrics']['ml']['mrr']:.4f}, "
            f"R@1={result['metrics']['ml']['recall_at_1']:.4f}, "
            f"R@3={result['metrics']['ml']['recall_at_3']:.4f}, "
            f"R@5={result['metrics']['ml']['recall_at_5']:.4f}",
            flush=True,
        )
        if config["publish_online_model"]:
            _publish_online_model(
                workdir=workdir,
                run_dir=run_dir,
                dataset_mode=config["dataset_mode"],
                resolved_samples=resolved_samples,
                pathway_names=pathway_names,
                pathway_to_tokens=pathway_to_tokens,
                universe_tokens=universe_tokens,
                structure_lookup=structure_lookup,
                parent_map=parent_map,
                metadata_by_id=metadata_by_id,
                max_samples_per_pathway=config["max_samples_per_pathway"],
                model_backend=model_backend,
            )
        print(f"Total output size: {_directory_size_bytes(run_dir) / (1024 * 1024):.2f} MB", flush=True)
        return

    fold_payloads: list[tuple[str, list[ResolvedSample], dict[str, Any]]]
    split_strategy = "compound-disjoint"
    # CV is evaluation-only. When publication is requested later, the deploy
    # model is retrained separately on all available data.
    if config["dataset_mode"] == "default_aracyc":
        fold_payloads = _build_default_cv_samples(
            pathway_names=pathway_names,
            pathway_to_tokens=pathway_to_tokens,
            metadata_by_id=metadata_by_id,
            max_samples_per_pathway=config["max_samples_per_pathway"],
            cv_folds=config["cv_folds"],
            rng=rng,
        )
    else:
        split_strategy = "sample-random"
        fold_payloads = _build_custom_cv_samples(
            resolved_samples,
            cv_folds=config["cv_folds"],
            rng=rng,
        )

    audit_path = run_dir / "dataset_audit.tsv"
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

    cv_sample_rows: list[dict[str, Any]] = []
    cv_test_enrichment_rows: list[dict[str, Any]] = []
    cv_prediction_rows: list[dict[str, Any]] = []
    cv_shap_feature_rows: list[dict[str, Any]] = []
    cv_shap_row_outputs: list[dict[str, Any]] = []
    cv_shap_long_rows: list[dict[str, Any]] = []
    fold_metric_rows: list[dict[str, Any]] = []
    root_metrics: dict[str, Any] = {
        "config": {
            "dataset_mode": config["dataset_mode"],
            "dataset": str(config["dataset"]) if config["dataset"] is not None else "",
            "compound_format": config["compound_format"],
            "use_split_column": config["use_split_column"],
            "strict": config["strict"],
            "cv_folds": config["cv_folds"],
            "split_strategy": split_strategy,
            "max_samples_per_pathway": config["max_samples_per_pathway"],
            "export_fold_pairs": config["export_fold_pairs"],
            "export_fold_models": config["export_fold_models"],
            "publish_online_model": config["publish_online_model"],
            "allow_sklearn_fallback": config["allow_sklearn_fallback"],
            "model_kind": model_backend["model_kind"],
            "fallback_used": model_backend["fallback_used"],
            "comparable_to_lightgbm": model_backend["comparable_to_lightgbm"],
            "lightgbm_available": model_backend["lightgbm_available"],
            "lightgbm_version": model_backend["lightgbm_version"],
            "shap_enabled": shap_backend["shap_enabled"],
            "shap_available": shap_backend["shap_available"],
            "shap_version": shap_backend["shap_version"],
            "shap_scope": shap_backend["shap_scope"],
            "export_shap_long": config["export_shap_long"],
        },
        "fold_metrics": {},
        "evaluation_warning": (
            "WARNING: Default AraCyc mode evaluates on in-database compounds; metrics are optimistic."
            if config["dataset_mode"] == "default_aracyc"
            else ""
        ),
        "model_kind": model_backend["model_kind"],
        "fallback_used": model_backend["fallback_used"],
        "comparable_to_lightgbm": model_backend["comparable_to_lightgbm"],
        "lightgbm_available": model_backend["lightgbm_available"],
        "lightgbm_version": model_backend["lightgbm_version"],
        "shap_enabled": shap_backend["shap_enabled"],
        "shap_available": shap_backend["shap_available"],
        "shap_version": shap_backend["shap_version"],
        "shap_scope": shap_backend["shap_scope"],
        "shap_long_exported": config["export_shap_long"],
        "shap_reason": shap_backend["shap_reason"],
    }
    if model_backend["lightgbm_import_error"]:
        root_metrics["config"]["lightgbm_import_error"] = model_backend["lightgbm_import_error"]
        root_metrics["lightgbm_import_error"] = model_backend["lightgbm_import_error"]
    if shap_backend["shap_reason"]:
        root_metrics["config"]["shap_reason"] = shap_backend["shap_reason"]

    for fold_id, fold_samples, fold_generation_summary in fold_payloads:
        fold_dir = run_dir / "folds" / fold_id
        result = _run_training_split(
            samples=fold_samples,
            out_dir=fold_dir,
            dataset_mode=config["dataset_mode"],
            state=state,
            pathway_names=pathway_names,
            pathway_to_tokens=pathway_to_tokens,
            universe_tokens=universe_tokens,
            structure_lookup=structure_lookup,
            parent_map=parent_map,
            rng=rng,
            generation_summary=fold_generation_summary,
            model_backend=model_backend,
            shap_backend={**shap_backend, "shap_long_exported": config["export_shap_long"]},
            write_pairs=config["export_fold_pairs"],
            write_model=config["export_fold_models"],
            fold_id=fold_id,
        )
        root_metrics["fold_metrics"][fold_id] = result["metrics"]
        fold_metric_rows.append(result["fold_metric_row"])
        for row in result["resolved_rows"]:
            cv_sample_rows.append({"fold_id": fold_id, **row})
        for row in result["test_enrichment_rows"]:
            cv_test_enrichment_rows.append({"fold_id": fold_id, **row})
        for row in result["prediction_rows"]:
            cv_prediction_rows.append({"fold_id": fold_id, **row})
        for row in result["shap_summary_rows"]:
            cv_shap_feature_rows.append({"fold_id": fold_id, **row})
        for row in result["shap_row_outputs"]:
            cv_shap_row_outputs.append({"fold_id": fold_id, **row})
        for row in result["shap_long_rows"]:
            cv_shap_long_rows.append({"fold_id": fold_id, **row})

    cv_summary_rows = _cv_summary_rows(fold_metric_rows)
    root_metrics["cv_summary"] = {row["metric"]: row for row in cv_summary_rows}
    root_metrics["shap_rows"] = len(cv_shap_row_outputs)

    _write_tsv(
        run_dir / "cv_samples.tsv",
        cv_sample_rows,
        ["fold_id"] + [
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
    _write_tsv(
        run_dir / "cv_fold_metrics.tsv",
        fold_metric_rows,
        list(fold_metric_rows[0].keys()),
    )
    _write_tsv(
        run_dir / "cv_fold_summary.tsv",
        cv_summary_rows,
        ["metric", "mean", "std", "min", "max"],
    )
    _write_tsv(
        run_dir / "cv_all_test_enrichment.tsv",
        cv_test_enrichment_rows,
        ["fold_id"] + _enrichment_fieldnames(),
    )
    _write_tsv(
        run_dir / "cv_all_predictions.tsv",
        cv_prediction_rows,
        ["fold_id"] + _enrichment_fieldnames() + ["ml_score", "baseline_rank", "ml_rank"],
    )
    if cv_shap_row_outputs:
        per_fold_shap_rows = {row["fold_id"]: int(row["shap_rows"]) for row in fold_metric_rows}
        shap_feature_accumulator: dict[str, dict[str, float]] = defaultdict(
            lambda: {
                "weight": 0.0,
                "mean_abs_shap": 0.0,
                "mean_shap": 0.0,
                "positive_fraction": 0.0,
                "negative_fraction": 0.0,
            }
        )
        for row in cv_shap_feature_rows:
            feature = str(row["feature"])
            weight = float(per_fold_shap_rows.get(str(row["fold_id"]), 0))
            accumulator = shap_feature_accumulator[feature]
            accumulator["weight"] += weight
            accumulator["mean_abs_shap"] += float(row["mean_abs_shap"]) * weight
            accumulator["mean_shap"] += float(row["mean_shap"]) * weight
            accumulator["positive_fraction"] += float(row["positive_fraction"]) * weight
            accumulator["negative_fraction"] += float(row["negative_fraction"]) * weight

        aggregated_shap_summary: list[dict[str, Any]] = []
        for feature, accumulator in shap_feature_accumulator.items():
            weight = accumulator["weight"] or 1.0
            aggregated_shap_summary.append(
                {
                    "feature": feature,
                    "mean_abs_shap": accumulator["mean_abs_shap"] / weight,
                    "mean_shap": accumulator["mean_shap"] / weight,
                    "positive_fraction": accumulator["positive_fraction"] / weight,
                    "negative_fraction": accumulator["negative_fraction"] / weight,
                }
            )
        aggregated_shap_summary.sort(key=lambda row: float(row["mean_abs_shap"]), reverse=True)
        aggregated_shap_summary = [{**row, "rank": rank} for rank, row in enumerate(aggregated_shap_summary, start=1)]
        _write_tsv(
            run_dir / "cv_shap_feature_summary.tsv",
            aggregated_shap_summary,
            ["feature", "mean_abs_shap", "mean_shap", "positive_fraction", "negative_fraction", "rank"],
        )
        _write_tsv(
            run_dir / "cv_all_test_shap_rows.tsv",
            cv_shap_row_outputs,
            [
                "fold_id",
                "sample_id",
                "pathway_id",
                "pathway_name",
                "label",
                "baseline_score",
                "ml_score",
                "baseline_rank",
                "ml_rank",
                "expected_value",
                "shap_sum",
                "reconstructed_score",
                "reconstruction_error",
                "top_positive_features_json",
                "top_negative_features_json",
            ],
        )
        if config["export_shap_long"]:
            _write_tsv(
                run_dir / "cv_all_test_shap_long.tsv",
                cv_shap_long_rows,
                [
                    "fold_id",
                    "sample_id",
                    "pathway_id",
                    "pathway_name",
                    "feature",
                    "feature_value",
                    "shap_value",
                    "abs_shap",
                    "direction",
                ],
            )
    (run_dir / "metrics.json").write_text(json.dumps(root_metrics, indent=2), encoding="utf-8")

    reason_counts: dict[str, int] = defaultdict(int)
    for row in audit_rows:
        reason_counts[row.reason] += 1
    filtered_summary = ", ".join(f"{reason}={count}" for reason, count in sorted(reason_counts.items())) or "none"

    print(f"Dataset mode: {config['dataset_mode']}", flush=True)
    print(f"CV mode: {config['cv_folds']}-fold {split_strategy}", flush=True)
    print(f"Model kind: {model_backend['model_kind']}", flush=True)
    print(f"Comparable to LGBMRanker: {model_backend['comparable_to_lightgbm']}", flush=True)
    if model_backend["lightgbm_version"]:
        print(f"LightGBM version: {model_backend['lightgbm_version']}", flush=True)
    if model_backend["lightgbm_import_error"]:
        print(f"LightGBM import error: {model_backend['lightgbm_import_error']}", flush=True)
    print(f"SHAP enabled: {shap_backend['shap_enabled']}", flush=True)
    if shap_backend["shap_version"]:
        print(f"SHAP version: {shap_backend['shap_version']}", flush=True)
    if shap_backend["shap_reason"]:
        print(f"SHAP reason: {shap_backend['shap_reason']}", flush=True)
    if root_metrics["evaluation_warning"]:
        print(root_metrics["evaluation_warning"], flush=True)
    print(f"Original samples: {len(resolved_samples) + len(audit_rows) if resolved_samples else 'fold-generated'}", flush=True)
    print(f"Filtered samples: {filtered_summary}", flush=True)
    print(f"Run output directory: {run_dir}", flush=True)
    for fold_id, _, _ in fold_payloads:
        print(f"{fold_id} directory: {run_dir / 'folds' / fold_id}", flush=True)
    for row in fold_metric_rows:
        print(
            f"{row['fold_id']}: train_samples={row['train_samples']}, test_samples={row['test_samples']}, "
            f"train_pairs={row['train_pairs']}, test_pairs={row['test_pairs']}",
            flush=True,
        )
    summary_by_metric = {row["metric"]: row for row in cv_summary_rows}
    print(
        "Baseline CV mean: "
        f"NDCG@5={summary_by_metric['baseline_ndcg_at_5']['mean']:.4f}±{summary_by_metric['baseline_ndcg_at_5']['std']:.4f}, "
        f"MRR={summary_by_metric['baseline_mrr']['mean']:.4f}±{summary_by_metric['baseline_mrr']['std']:.4f}, "
        f"R@1={summary_by_metric['baseline_recall_at_1']['mean']:.4f}±{summary_by_metric['baseline_recall_at_1']['std']:.4f}, "
        f"R@3={summary_by_metric['baseline_recall_at_3']['mean']:.4f}±{summary_by_metric['baseline_recall_at_3']['std']:.4f}, "
        f"R@5={summary_by_metric['baseline_recall_at_5']['mean']:.4f}±{summary_by_metric['baseline_recall_at_5']['std']:.4f}",
        flush=True,
    )
    print(
        "ML CV mean: "
        f"NDCG@5={summary_by_metric['ml_ndcg_at_5']['mean']:.4f}±{summary_by_metric['ml_ndcg_at_5']['std']:.4f}, "
        f"MRR={summary_by_metric['ml_mrr']['mean']:.4f}±{summary_by_metric['ml_mrr']['std']:.4f}, "
        f"R@1={summary_by_metric['ml_recall_at_1']['mean']:.4f}±{summary_by_metric['ml_recall_at_1']['std']:.4f}, "
        f"R@3={summary_by_metric['ml_recall_at_3']['mean']:.4f}±{summary_by_metric['ml_recall_at_3']['std']:.4f}, "
        f"R@5={summary_by_metric['ml_recall_at_5']['mean']:.4f}±{summary_by_metric['ml_recall_at_5']['std']:.4f}",
        flush=True,
    )
    print(
        "ML vs baseline delta mean: "
        f"NDCG@5={summary_by_metric['ndcg_at_5_delta']['mean']:.4f}, "
        f"MRR={summary_by_metric['mrr_delta']['mean']:.4f}, "
        f"R@1={summary_by_metric['recall_at_1_delta']['mean']:.4f}, "
        f"R@3={summary_by_metric['recall_at_3_delta']['mean']:.4f}, "
        f"R@5={summary_by_metric['recall_at_5_delta']['mean']:.4f}",
        flush=True,
    )
    if config["publish_online_model"]:
        _publish_online_model(
            workdir=workdir,
            run_dir=run_dir,
            dataset_mode=config["dataset_mode"],
            resolved_samples=resolved_samples,
            pathway_names=pathway_names,
            pathway_to_tokens=pathway_to_tokens,
            universe_tokens=universe_tokens,
            structure_lookup=structure_lookup,
            parent_map=parent_map,
            metadata_by_id=metadata_by_id,
            max_samples_per_pathway=config["max_samples_per_pathway"],
            model_backend=model_backend,
        )
    print(f"Total output size: {_directory_size_bytes(run_dir) / (1024 * 1024):.2f} MB", flush=True)


if __name__ == "__main__":
    main()
