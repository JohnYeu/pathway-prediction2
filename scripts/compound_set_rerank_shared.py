#!/usr/bin/env python3
"""Shared helpers for compound-set -> pathway reranking.

This module exists to keep training-time and online-time feature construction
identical. The training script and the online enrichment CLI both need:

- the same query-level chemistry features
- the same per-(query, pathway) enrichment feature layout
- the same feature column ordering when calling LightGBM
- a stable on-disk location for the published online model bundle

Keeping those helpers here prevents the online CLI from silently drifting away
from the model that was actually trained and published.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from pathway_query_shared import DataStructs, _mol_from_smiles, _morgan_fingerprint_from_mol

if TYPE_CHECKING:
    from enrich_pathways import CompoundTargetRecord, PathwayEnrichmentRow


TRAINING_CANDIDATE_POLICY = "positive_paths_plus_top20_hard_negatives_plus_random_to_30"
ONLINE_CANDIDATE_POLICY = "baseline_enrichment_rows_x_i_ge_2_only"


def online_model_dir(workdir: Path) -> Path:
    """Return the fixed deployment directory for the online rerank model."""

    return workdir / "outputs" / "onlineModels" / "compound_set_pathway_rerank"


def load_parent_map(workdir: Path) -> dict[str, str]:
    """Load the ChEBI parent map used for ontology diversity features."""

    parent_map: dict[str, str] = {}
    with (workdir / "compounds.tsv").open(newline="", encoding="utf-8") as handle:
        import csv

        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            compound_id = row["id"]
            parent_id = (row.get("parent_id") or "").strip()
            parent_map[compound_id] = parent_id
    return parent_map


def root_ancestor(compound_id: str, parent_map: dict[str, str], cache: dict[str, str]) -> str:
    """Return the root ancestor for a compound in the ChEBI parent tree."""

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


def ensure_fingerprint(record: Any) -> Any:
    """Populate and return an RDKit fingerprint for a structure record."""

    if record is None:
        return None
    if getattr(record, "fingerprint", None) is not None:
        return record.fingerprint
    if not getattr(record, "smiles", "") or DataStructs is None:
        return None
    mol = _mol_from_smiles(record.smiles)
    if mol is None:
        return None
    record.fingerprint = _morgan_fingerprint_from_mol(mol)
    return record.fingerprint


def pairwise_tanimoto(compound_ids: list[str], structure_lookup: dict[str, Any]) -> tuple[float, float, float]:
    """Summarize pairwise structural similarity inside one compound set."""

    if DataStructs is None or len(compound_ids) < 2:
        return 0.0, 0.0, 0.0
    values: list[float] = []
    for i in range(len(compound_ids)):
        left = structure_lookup.get(compound_ids[i])
        left_fp = ensure_fingerprint(left)
        if left_fp is None:
            continue
        for j in range(i + 1, len(compound_ids)):
            right = structure_lookup.get(compound_ids[j])
            right_fp = ensure_fingerprint(right)
            if right_fp is None:
                continue
            values.append(float(DataStructs.TanimotoSimilarity(left_fp, right_fp)))
    if not values:
        return 0.0, 0.0, 0.0
    return float(sum(values) / len(values)), float(max(values)), float(min(values))


def query_level_features_from_compound_ids(
    compound_ids: list[str],
    *,
    structure_lookup: dict[str, Any],
    parent_map: dict[str, str],
    root_cache: dict[str, str],
) -> dict[str, float]:
    """Compute the query-level features shared by training and online rerank."""

    tan_mean, tan_max, tan_min = pairwise_tanimoto(compound_ids, structure_lookup)
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
    roots = {root_ancestor(compound_id, parent_map, root_cache) for compound_id in compound_ids}
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


def compound_records_from_items(items: list[Any]) -> dict[str, Any]:
    """Project resolved compounds into the minimal enrichment input schema."""

    from enrich_pathways import CompoundTargetRecord

    records: dict[str, Any] = {}
    for item in items:
        records[item.compound_id] = CompoundTargetRecord(
            compound_id=item.compound_id,
            chebi_name=item.chebi_name,
            match_type=getattr(item, "match_type", "") or "direct",
            match_score=float(getattr(item, "match_score", 1.0)),
        )
    return records


def rerank_feature_values(
    pathway_row: Any,
    query_features: dict[str, float],
) -> dict[str, float]:
    """Convert one enrichment row plus query features into model features."""

    return {
        "baseline_score": float(pathway_row.final_score),
        "x_i": float(pathway_row.x_i),
        "K_i": float(pathway_row.K_i),
        "n": float(pathway_row.n),
        "N": float(pathway_row.N),
        "query_coverage": float(pathway_row.target_rate),
        "pathway_coverage": float(pathway_row.coverage),
        "background_rate": float(pathway_row.background_rate),
        "expected_hits": float(pathway_row.expected_hits),
        "excess_hits": float(pathway_row.excess_hits),
        "EF": float(pathway_row.enrichment_factor),
        "p_value": float(pathway_row.p_value),
        "FDR": float(pathway_row.fdr),
        "neg_log10_p": float(pathway_row.neg_log10_p),
        "neg_log10_fdr": float(pathway_row.neg_log10_fdr),
        "mapping_confidence_mean": float(pathway_row.mapping_confidence_mean),
        "mapping_confidence_min": float(pathway_row.mapping_confidence_min),
        "mapping_confidence_max": float(pathway_row.mapping_confidence_max),
        "direct_hit_count": float(pathway_row.direct_hit_count),
        "fuzzy_hit_count": float(pathway_row.fuzzy_hit_count),
        "recovered_hit_count": float(pathway_row.recovered_hit_count),
        "exact_structure_hit_count": float(pathway_row.exact_structure_hit_count),
        "structural_neighbor_hit_count": float(pathway_row.structural_neighbor_hit_count),
        **query_features,
    }


def feature_frame(rows: list[dict[str, Any]], feature_columns: list[str]) -> pd.DataFrame:
    """Build a feature matrix with a stable, explicit column order."""

    if not rows:
        return pd.DataFrame(columns=feature_columns, dtype=float)
    return pd.DataFrame(
        [{column: float(row[column]) for column in feature_columns} for row in rows],
        columns=feature_columns,
        dtype=float,
    )


def load_online_model_bundle(workdir: Path) -> tuple[dict[str, Any] | None, str]:
    """Load the published online model bundle or return a structured failure reason."""

    model_dir = online_model_dir(workdir)
    model_path = model_dir / "model.pkl"
    metadata_path = model_dir / "metadata.json"
    label_vocab_path = model_dir / "label_vocab.json"

    if not model_dir.exists() or not model_path.exists() or not metadata_path.exists() or not label_vocab_path.exists():
        return None, "online_model_missing"

    try:
        bundle = pickle.load(model_path.open("rb"))
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        label_vocab = json.loads(label_vocab_path.read_text(encoding="utf-8"))
    except Exception:
        return None, "online_model_unreadable"

    if not isinstance(bundle, dict):
        return None, "online_model_unreadable"
    model_kind = str(bundle.get("model_kind") or metadata.get("model_kind") or "")
    feature_columns = bundle.get("feature_columns")
    model = bundle.get("model")
    if model_kind != "lightgbm_lambdarank" or not isinstance(feature_columns, list) or not feature_columns or model is None:
        return None, "online_model_incompatible"

    return {
        "model_kind": model_kind,
        "feature_columns": [str(column) for column in feature_columns],
        "model": model,
        "metadata": metadata,
        "label_vocab": label_vocab,
        "model_dir": model_dir,
    }, "applied"


def predict_rerank_scores(model_bundle: dict[str, Any], feature_rows: list[dict[str, Any]]) -> np.ndarray:
    """Score rerank feature rows with the published online LightGBM model."""

    X = feature_frame(feature_rows, model_bundle["feature_columns"])
    return np.asarray(model_bundle["model"].predict(X), dtype=float)
