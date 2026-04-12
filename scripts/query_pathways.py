#!/usr/bin/env python3
"""Query-time pathway lookup for the active AraCyc-first pipeline.

Usage:
    python3 query_pathways.py --workdir ..
    python3 query_pathways.py --workdir .. --query "glucose"
    python3 query_pathways.py --workdir .. --query "glutamte"
    python3 query_pathways.py --workdir .. --query "block:WQZGKKKJIJFFOK"
    python3 query_pathways.py --workdir .. --query "smiles:OC(=O)CCC(N)C(=O)O"
"""

from __future__ import annotations

import argparse
import csv
import gzip
import pickle
import re
import sys
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from process_chebi_to_pathways_v2 import build_variants, formula_key, normalize_name

try:
    from rdkit import Chem, DataStructs, RDLogger
    from rdkit.Chem import AllChem
except Exception:  # pragma: no cover - graceful degradation when RDKit is absent
    Chem = None
    DataStructs = None
    AllChem = None
    RDLogger = None


FULL_INCHIKEY_RE = re.compile(r"^[A-Z]{14}-[A-Z]{10}-[A-Z]$")
BLOCK_QUERY_RE = re.compile(r"^block:([A-Z]{14})$")
SMILES_QUERY_RE = re.compile(r"^smiles:(.+)$", re.IGNORECASE)
CACHE_SCHEMA_VERSION = 1
VARIANT_ORDER = ("exact", "compact", "singular", "stereo_stripped")
FUZZY_MIN_RATIO = 0.85
FUZZY_AUTO_RATIO = 0.92
FUZZY_MIN_GAP = 0.05
FUZZY_MIN_NAME_LENGTH = 6
SMILES_NEIGHBOR_MIN = 0.60
RECOVERED_NEIGHBOR_MIN = 0.85
BLOCK_SIMILARITY = 0.90
MAX_FUZZY_SUGGESTIONS = 5
MAX_STRUCTURE_NEIGHBORS = 20


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
class QueryPathway:
    """A pathway prepared for display after query-time aggregation."""

    compound_id: str
    chebi_name: str
    pathway_name: str
    pathway_id: str
    pathway_category: str
    score: float
    confidence_level: str
    match_method: str
    aracyc_compound_id: str
    source_db: str
    gene_count: int
    ec_numbers: str
    annotation_confidence: str
    reason: str
    support_similarity: float = 1.0
    recovered_chebi_id: str = ""


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


@dataclass(slots=True)
class ChEBIExactCandidate:
    """A strict exact-lookup ChEBI entity recovered from names.tsv/compounds.tsv."""

    compound_id: str
    chebi_accession: str
    name: str
    ascii_name: str
    stars: int


@dataclass(slots=True)
class StructureRecord:
    """Structure and chemistry fields used for query-time structure recovery."""

    compound_id: str
    chebi_name: str
    smiles: str
    standard_inchi: str
    standard_inchi_key: str
    formula: str
    formula_key: str
    charge: str
    exact_mass: float | None
    fingerprint: Any = field(default=None, repr=False)

    @property
    def inchikey_block(self) -> str:
        return _inchikey_block(self.standard_inchi_key)


@dataclass(slots=True)
class NeighborHit:
    """One structure-derived Arabidopsis compound candidate."""

    compound_id: str
    similarity: float
    recovered_chebi_id: str = ""


@dataclass(slots=True)
class FuzzyMatchOutcome:
    """Result of Arabidopsis fuzzy recovery."""

    records: list[AraCycMappingRecord]
    did_you_mean: tuple[str, ...]


@dataclass(slots=True)
class QueryResult:
    """Full result bundle returned by run_query()."""

    query: str
    match_type: str
    resolution_path: str
    mappings: list[AraCycMappingRecord]
    pathways: list[QueryPathway]
    expanded: list[ExpandedPathway]
    matched_compound: str = ""
    neighbor_similarity: float = 0.0
    did_you_mean: tuple[str, ...] = ()
    recovered_chebi_id: str = ""
    projection_type: str = ""
    note: str = ""


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def _open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", newline="")
    return path.open("r", encoding="utf-8", newline="")


def _safe_float(value: str) -> float | None:
    text = (value or "").strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _normalize_charge(value: str) -> str:
    text = (value or "").strip()
    if not text:
        return ""
    try:
        return str(int(float(text)))
    except ValueError:
        return text


def _inchikey_block(value: str) -> str:
    text = (value or "").strip().upper()
    if not text:
        return ""
    return text.split("-", 1)[0]


def _mol_from_smiles(text: str):
    if Chem is None:
        return None
    try:
        if RDLogger is not None:
            RDLogger.DisableLog("rdApp.warning")
            RDLogger.DisableLog("rdApp.error")
        return Chem.MolFromSmiles(text)
    except Exception:
        return None
    finally:
        if RDLogger is not None:
            RDLogger.EnableLog("rdApp.warning")
            RDLogger.EnableLog("rdApp.error")


def _mol_to_inchikey(mol) -> str:
    if Chem is None or mol is None:
        return ""
    try:
        return Chem.MolToInchiKey(mol)
    except Exception:
        return ""


def _morgan_fingerprint_from_mol(mol):
    if AllChem is None or mol is None:
        return None
    try:
        if RDLogger is not None:
            RDLogger.DisableLog("rdApp.warning")
            RDLogger.DisableLog("rdApp.error")
        return AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
    except Exception:
        return None
    finally:
        if RDLogger is not None:
            RDLogger.EnableLog("rdApp.warning")
            RDLogger.EnableLog("rdApp.error")


def _length_prefilter(query_text: str, candidate_text: str) -> bool:
    qlen = len(query_text)
    clen = len(candidate_text)
    if not qlen or not clen:
        return False
    return abs(qlen - clen) / max(qlen, clen) <= 0.40


def _record_display_name(record: AraCycMappingRecord) -> str:
    primary = record.chebi_name or record.aracyc_common_name
    secondary = record.aracyc_common_name if record.aracyc_common_name and record.aracyc_common_name != primary else ""
    if secondary:
        return f"{primary} / {secondary}"
    return primary


def _dedupe_mappings(records: list[AraCycMappingRecord]) -> list[AraCycMappingRecord]:
    seen: set[str] = set()
    result: list[AraCycMappingRecord] = []
    for record in sorted(records, key=lambda r: r.match_score, reverse=True):
        if record.compound_id in seen:
            continue
        seen.add(record.compound_id)
        result.append(record)
    return result


# ---------------------------------------------------------------------------
# Index loading
# ---------------------------------------------------------------------------


def load_name_to_aracyc(path: Path) -> tuple[dict[str, list[AraCycMappingRecord]], dict[str, list[AraCycMappingRecord]]]:
    """Load name_to_aracyc_index.tsv into name and compound lookups."""
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
            for name in (row["chebi_name"], row["aracyc_common_name"]):
                for variant_text in build_variants(name).values():
                    if variant_text:
                        by_name.setdefault(variant_text, []).append(record)

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
# Lazy query caches
# ---------------------------------------------------------------------------


def _source_mtimes(sources: list[Path]) -> dict[str, float]:
    return {str(path): path.stat().st_mtime for path in sources if path.exists()}


def _load_or_build_pickle(
    cache_path: Path,
    builder,
    label: str,
    verbose: bool,
    sources: list[Path] | None = None,
) -> Any:
    sources = sources or []
    current_mtimes = _source_mtimes(sources)
    if cache_path.exists():
        try:
            with cache_path.open("rb") as handle:
                wrapper = pickle.load(handle)
            if (
                isinstance(wrapper, dict)
                and wrapper.get("schema_version") == CACHE_SCHEMA_VERSION
                and wrapper.get("source_mtimes") == current_mtimes
            ):
                return wrapper["payload"]
            if verbose:
                print(f"  Rebuilding {label} (cache stale) ...", flush=True)
        except Exception:
            if verbose:
                print(f"  Rebuilding {label} (cache unreadable) ...", flush=True)
    elif verbose:
        print(f"  Building {label} ...", flush=True)
    payload = builder()
    wrapper = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "source_mtimes": current_mtimes,
        "payload": payload,
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("wb") as handle:
        pickle.dump(wrapper, handle)
    return payload


def ensure_chebi_name_lookup(state: dict[str, Any], verbose: bool = False) -> dict[str, Any]:
    if state.get("chebi_name_lookup") is not None:
        return state["chebi_name_lookup"]

    workdir = state["workdir"]
    cache_path = workdir / "outputs" / "preprocessed" / "query_chebi_name_lookup.pkl"
    compounds_path = workdir / "compounds.tsv"
    names_path = workdir / "refs" / "names.tsv.gz"

    def builder() -> dict[str, Any]:
        records: dict[str, dict[str, Any]] = {}
        variant_indexes = {variant: {} for variant in VARIANT_ORDER}

        def add_name(compound_id: str, raw_name: str) -> None:
            text = (raw_name or "").strip()
            if not text:
                return
            for variant, value in build_variants(text).items():
                if not value:
                    continue
                variant_indexes[variant].setdefault(value, set()).add(compound_id)

        with compounds_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            for row in reader:
                compound_id = row["id"]
                records[compound_id] = {
                    "compound_id": compound_id,
                    "chebi_accession": row["chebi_accession"],
                    "name": row["name"],
                    "ascii_name": row.get("ascii_name", ""),
                    "stars": int(row.get("stars") or 0),
                }
                add_name(compound_id, row["name"])
                add_name(compound_id, row.get("ascii_name", ""))

        with _open_text(names_path) as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            for row in reader:
                if row.get("language_code", "") not in {"", "en"}:
                    continue
                compound_id = row["compound_id"]
                if compound_id not in records:
                    continue
                add_name(compound_id, row.get("name", ""))
                add_name(compound_id, row.get("ascii_name", ""))

        compact_indexes = {
            variant: {key: tuple(sorted(value_set, key=int)) for key, value_set in index.items()}
            for variant, index in variant_indexes.items()
        }
        return {"records": records, "variant_indexes": compact_indexes}

    lookup = _load_or_build_pickle(
        cache_path,
        builder,
        "query_chebi_name_lookup.pkl",
        verbose,
        sources=[compounds_path, names_path],
    )
    state["chebi_name_lookup"] = lookup
    return lookup


def ensure_chebi_structure_lookup(state: dict[str, Any], verbose: bool = False) -> dict[str, StructureRecord]:
    if state.get("chebi_structure_lookup") is not None:
        return state["chebi_structure_lookup"]

    workdir = state["workdir"]
    cache_path = workdir / "outputs" / "preprocessed" / "query_chebi_structure_lookup.pkl"
    structures_path = workdir / "refs" / "structures.tsv.gz"
    chemical_data_path = workdir / "refs" / "chemical_data.tsv.gz"
    compounds_path = workdir / "compounds.tsv"

    def builder() -> dict[str, StructureRecord]:
        compound_names: dict[str, str] = {}
        with compounds_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            for row in reader:
                compound_names[row["id"]] = row["name"]

        chemical_lookup: dict[str, dict[str, Any]] = {}
        chemical_preference: dict[str, int] = {}
        with _open_text(chemical_data_path) as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            for row in reader:
                compound_id = row["compound_id"]
                status_rank = 0 if row.get("status_id", "") == "1" else 1
                if compound_id in chemical_lookup and status_rank >= chemical_preference[compound_id]:
                    continue
                exact_mass = _safe_float(row.get("monoisotopic_mass", "")) or _safe_float(row.get("mass", ""))
                chemical_lookup[compound_id] = {
                    "formula": row.get("formula", "") or "",
                    "formula_key": formula_key(row.get("formula", "") or ""),
                    "charge": _normalize_charge(row.get("charge", "")),
                    "exact_mass": exact_mass,
                }
                chemical_preference[compound_id] = status_rank

        structure_lookup: dict[str, StructureRecord] = {}
        structure_preference: dict[str, tuple[int, int]] = {}
        with _open_text(structures_path) as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            for row in reader:
                compound_id = row["compound_id"]
                default_rank = 0 if row.get("default_structure", "").lower() == "true" else 1
                status_rank = 0 if row.get("status_id", "") == "1" else 1
                current_rank = (default_rank, status_rank)
                if compound_id in structure_lookup and current_rank >= structure_preference[compound_id]:
                    continue
                chem = chemical_lookup.get(compound_id, {})
                structure_lookup[compound_id] = StructureRecord(
                    compound_id=compound_id,
                    chebi_name=compound_names.get(compound_id, ""),
                    smiles=row.get("smiles", "") or "",
                    standard_inchi=row.get("standard_inchi", "") or "",
                    standard_inchi_key=(row.get("standard_inchi_key", "") or "").upper(),
                    formula=chem.get("formula", ""),
                    formula_key=chem.get("formula_key", ""),
                    charge=chem.get("charge", ""),
                    exact_mass=chem.get("exact_mass"),
                )
                structure_preference[compound_id] = current_rank
        return structure_lookup

    lookup = _load_or_build_pickle(
        cache_path,
        builder,
        "query_chebi_structure_lookup.pkl",
        verbose,
        sources=[compounds_path, structures_path, chemical_data_path],
    )
    state["chebi_structure_lookup"] = lookup
    return lookup


def ensure_arabidopsis_structure_index(state: dict[str, Any], verbose: bool = False) -> dict[str, Any]:
    if state.get("arabidopsis_structure_index") is not None:
        return state["arabidopsis_structure_index"]

    workdir = state["workdir"]
    cache_path = workdir / "outputs" / "preprocessed" / "query_arabidopsis_structure_index.pkl"
    pathway_output_path = workdir / "outputs" / "chebi_pathways_aracyc_refactored.tsv"
    structures_path = workdir / "refs" / "structures.tsv.gz"
    chebi_structure_lookup = ensure_chebi_structure_lookup(state, verbose=verbose)
    pathway_index: dict[str, list[AraCycRankedPathway]] = state["pathway_index"]

    def builder() -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for compound_id in sorted(pathway_index, key=int):
            structure = chebi_structure_lookup.get(compound_id)
            if structure is None:
                continue
            if not structure.standard_inchi_key and not structure.smiles:
                continue
            rows.append({
                "compound_id": structure.compound_id,
                "chebi_name": structure.chebi_name,
                "smiles": structure.smiles,
                "standard_inchi": structure.standard_inchi,
                "standard_inchi_key": structure.standard_inchi_key,
                "formula": structure.formula,
                "formula_key": structure.formula_key,
                "charge": structure.charge,
                "exact_mass": structure.exact_mass,
            })
        return rows

    raw_rows = _load_or_build_pickle(
        cache_path,
        builder,
        "query_arabidopsis_structure_index.pkl",
        verbose,
        sources=[pathway_output_path, structures_path],
    )
    records: list[StructureRecord] = []
    by_full_inchikey: dict[str, list[StructureRecord]] = {}
    by_block: dict[str, list[StructureRecord]] = {}
    for row in raw_rows:
        record = StructureRecord(
            compound_id=row["compound_id"],
            chebi_name=row.get("chebi_name", ""),
            smiles=row.get("smiles", ""),
            standard_inchi=row.get("standard_inchi", ""),
            standard_inchi_key=(row.get("standard_inchi_key", "") or "").upper(),
            formula=row.get("formula", ""),
            formula_key=row.get("formula_key", ""),
            charge=row.get("charge", ""),
            exact_mass=row.get("exact_mass"),
        )
        if record.smiles:
            mol = _mol_from_smiles(record.smiles)
            if mol is not None:
                record.fingerprint = _morgan_fingerprint_from_mol(mol)
        records.append(record)
        if record.standard_inchi_key:
            by_full_inchikey.setdefault(record.standard_inchi_key, []).append(record)
        if record.inchikey_block:
            by_block.setdefault(record.inchikey_block, []).append(record)

    index = {"records": records, "by_full_inchikey": by_full_inchikey, "by_block": by_block}
    state["arabidopsis_structure_index"] = index
    return index


# ---------------------------------------------------------------------------
# Query resolution helpers
# ---------------------------------------------------------------------------


def resolve_query(query: str, name_index: dict[str, list[AraCycMappingRecord]]) -> list[AraCycMappingRecord]:
    """Resolve a plain-text query to AraCyc mapping records."""
    variants = build_variants(query)
    seen_compound_ids: set[str] = set()
    results: list[AraCycMappingRecord] = []

    for variant in VARIANT_ORDER:
        variant_text = variants.get(variant, "")
        if not variant_text:
            continue
        for record in name_index.get(variant_text, []):
            if record.compound_id in seen_compound_ids:
                continue
            seen_compound_ids.add(record.compound_id)
            results.append(record)

    return sorted(results, key=lambda r: r.match_score, reverse=True)


def fuzzy_match_arabidopsis(query: str, state: dict[str, Any]) -> FuzzyMatchOutcome:
    """Try typo correction only within the Arabidopsis indexed name space."""
    query_variants = build_variants(query)
    query_length = len(query.strip())
    if query_length < FUZZY_MIN_NAME_LENGTH:
        return FuzzyMatchOutcome(records=[], did_you_mean=())

    scores: list[tuple[float, str]] = []
    for candidate_text in state["arabidopsis_name_texts"]:
        if not _length_prefilter(query_variants["exact"], candidate_text):
            continue
        ratio = max(
            SequenceMatcher(None, query_variants[variant], candidate_text).ratio()
            for variant in VARIANT_ORDER
            if query_variants.get(variant)
        )
        if ratio >= FUZZY_MIN_RATIO:
            scores.append((ratio, candidate_text))

    if not scores:
        return FuzzyMatchOutcome(records=[], did_you_mean=())

    scores.sort(key=lambda item: (-item[0], item[1]))
    top_scores = scores[:MAX_FUZZY_SUGGESTIONS]
    suggestions: list[str] = []
    for _, candidate_text in top_scores:
        for record in state["name_index"].get(candidate_text, []):
            display_name = _record_display_name(record)
            if display_name not in suggestions:
                suggestions.append(display_name)
            if len(suggestions) >= MAX_FUZZY_SUGGESTIONS:
                break
        if len(suggestions) >= MAX_FUZZY_SUGGESTIONS:
            break

    best_ratio, best_text = top_scores[0]
    second_ratio = top_scores[1][0] if len(top_scores) > 1 else 0.0
    if best_ratio >= FUZZY_AUTO_RATIO and (best_ratio - second_ratio) >= FUZZY_MIN_GAP:
        records = _dedupe_mappings(state["name_index"].get(best_text, []))
        return FuzzyMatchOutcome(records=records, did_you_mean=tuple(suggestions))
    return FuzzyMatchOutcome(records=[], did_you_mean=tuple(suggestions))


def recover_chebi_exact_candidates(query: str, state: dict[str, Any], verbose: bool = False) -> list[ChEBIExactCandidate]:
    """Recover full ChEBI entities using exact hash lookups only."""
    lookup = ensure_chebi_name_lookup(state, verbose=verbose)
    query_variants = build_variants(query)
    candidate_ids: list[str] = []

    for variant in VARIANT_ORDER:
        variant_text = query_variants.get(variant, "")
        if not variant_text:
            continue
        ids = lookup["variant_indexes"].get(variant, {}).get(variant_text, ())
        if ids:
            candidate_ids = list(ids)
            break

    records = lookup["records"]
    candidates = [
        ChEBIExactCandidate(
            compound_id=compound_id,
            chebi_accession=records[compound_id]["chebi_accession"],
            name=records[compound_id]["name"],
            ascii_name=records[compound_id]["ascii_name"],
            stars=int(records[compound_id]["stars"]),
        )
        for compound_id in candidate_ids
        if compound_id in records
    ]
    candidates.sort(key=lambda record: (-record.stars, int(record.compound_id)))
    return candidates


def _is_mass_compatible(query_record: StructureRecord, target_record: StructureRecord) -> bool:
    if query_record.exact_mass is None or target_record.exact_mass is None:
        return True
    return abs(query_record.exact_mass - target_record.exact_mass) <= 0.5


def _is_charge_compatible(query_record: StructureRecord, target_record: StructureRecord) -> bool:
    if not query_record.charge or not target_record.charge:
        return True
    return query_record.charge == target_record.charge


def aggregate_primary_pathways(
    neighbors: list[NeighborHit],
    state: dict[str, Any],
    top_k: int,
) -> list[QueryPathway]:
    """Aggregate primary pathways from supporting Arabidopsis compounds."""
    pathway_index: dict[str, list[AraCycRankedPathway]] = state["pathway_index"]
    aggregated: dict[str, QueryPathway] = {}

    for neighbor in neighbors:
        for pathway in pathway_index.get(neighbor.compound_id, []):
            predicted_score = round(pathway.score * neighbor.similarity, 4)
            key = pathway.pathway_id or pathway.pathway_name
            existing = aggregated.get(key)
            if existing is not None and predicted_score <= existing.score:
                continue
            aggregated[key] = QueryPathway(
                compound_id=pathway.compound_id,
                chebi_name=pathway.chebi_name,
                pathway_name=pathway.pathway_name,
                pathway_id=pathway.pathway_id,
                pathway_category=pathway.pathway_category,
                score=predicted_score,
                confidence_level=pathway.confidence_level,
                match_method=pathway.match_method,
                aracyc_compound_id=pathway.aracyc_compound_id,
                source_db=pathway.source_db,
                gene_count=pathway.gene_count,
                ec_numbers=pathway.ec_numbers,
                annotation_confidence=pathway.annotation_confidence,
                reason=(
                    pathway.reason
                    if neighbor.similarity >= 0.999
                    else f"{pathway.reason}; structural_similarity={neighbor.similarity:.3f}"
                ),
                support_similarity=neighbor.similarity,
                recovered_chebi_id=neighbor.recovered_chebi_id,
            )

    results = sorted(
        aggregated.values(),
        key=lambda pathway: (pathway.score, pathway.support_similarity),
        reverse=True,
    )
    return results[:top_k]


def collect_support_mappings(compound_ids: list[str], state: dict[str, Any]) -> list[AraCycMappingRecord]:
    compound_index: dict[str, list[AraCycMappingRecord]] = state["compound_index"]
    records: list[AraCycMappingRecord] = []
    for compound_id in compound_ids:
        records.extend(compound_index.get(compound_id, []))
    return _dedupe_mappings(records)


def collect_expanded_predictions(compound_ids: list[str], state: dict[str, Any], top_k: int) -> list[ExpandedPathway]:
    expanded_index: dict[str, list[ExpandedPathway]] = state["expanded_index"]
    expanded: list[ExpandedPathway] = []
    for compound_id in compound_ids:
        expanded.extend(expanded_index.get(compound_id, []))
    expanded.sort(key=lambda item: item.ml_score, reverse=True)
    return expanded[:top_k]


def build_query_result(
    *,
    query: str,
    match_type: str,
    resolution_path: str,
    neighbors: list[NeighborHit],
    state: dict[str, Any],
    top_k: int,
    include_expanded: bool = False,
    did_you_mean: tuple[str, ...] = (),
    base_mappings: list[AraCycMappingRecord] | None = None,
    expanded_compound_ids: list[str] | None = None,
    projection_type: str = "",
) -> QueryResult:
    pathways = aggregate_primary_pathways(neighbors, state, top_k=top_k)
    support_compound_ids = [pathway.compound_id for pathway in pathways]
    mappings = _dedupe_mappings(base_mappings) if base_mappings is not None else collect_support_mappings(support_compound_ids, state)
    expanded_source_ids = expanded_compound_ids if expanded_compound_ids is not None else support_compound_ids
    expanded = collect_expanded_predictions(expanded_source_ids, state, top_k) if include_expanded else []
    best_pathway = pathways[0] if pathways else None
    matched_compound = best_pathway.chebi_name if best_pathway else (mappings[0].chebi_name if mappings else "")
    recovered_chebi_id = best_pathway.recovered_chebi_id if best_pathway else ""
    neighbor_similarity = max((pathway.support_similarity for pathway in pathways), default=0.0)
    return QueryResult(
        query=query,
        match_type=match_type,
        resolution_path=resolution_path,
        mappings=mappings,
        pathways=pathways,
        expanded=expanded,
        matched_compound=matched_compound,
        neighbor_similarity=neighbor_similarity,
        did_you_mean=did_you_mean,
        recovered_chebi_id=recovered_chebi_id,
        projection_type=projection_type,
    )


def resolve_full_inchikey_query(query: str, state: dict[str, Any], top_k: int, resolution_path: str) -> QueryResult:
    index = ensure_arabidopsis_structure_index(state, verbose=False)
    records = index["by_full_inchikey"].get(query.upper(), [])
    neighbors = [NeighborHit(compound_id=record.compound_id, similarity=1.0) for record in records]
    if not neighbors:
        return QueryResult(
            query=query,
            match_type="",
            resolution_path=resolution_path,
            mappings=[],
            pathways=[],
            expanded=[],
            note="No Arabidopsis compound has this exact InChIKey.",
        )
    return build_query_result(
        query=query,
        match_type="exact_structure",
        resolution_path=resolution_path,
        neighbors=neighbors,
        state=state,
        top_k=top_k,
    )


def resolve_block_query(query: str, state: dict[str, Any], top_k: int) -> QueryResult:
    match = BLOCK_QUERY_RE.fullmatch(query.strip())
    if match is None:
        return QueryResult(
            query=query,
            match_type="",
            resolution_path="block_query",
            mappings=[],
            pathways=[],
            expanded=[],
            note="Invalid block query. Use block:XXXXXXXXXXXXXX",
        )
    block = match.group(1)
    index = ensure_arabidopsis_structure_index(state, verbose=False)
    records = index["by_block"].get(block, [])
    neighbors = [NeighborHit(compound_id=record.compound_id, similarity=BLOCK_SIMILARITY) for record in records]
    if not neighbors:
        return QueryResult(
            query=query,
            match_type="",
            resolution_path="block -> structural_neighbor",
            mappings=[],
            pathways=[],
            expanded=[],
            note="No Arabidopsis compound shares this InChIKey connectivity block.",
        )
    return build_query_result(
        query=query,
        match_type="structural_neighbor",
        resolution_path="block -> structural_neighbor",
        neighbors=neighbors,
        state=state,
        top_k=top_k,
    )


def resolve_smiles_query(query: str, state: dict[str, Any], top_k: int) -> QueryResult:
    mol = _mol_from_smiles(query)
    if mol is None or Chem is None or DataStructs is None:
        return QueryResult(
            query=query,
            match_type="",
            resolution_path="smiles",
            mappings=[],
            pathways=[],
            expanded=[],
            note="Input is not a valid SMILES string in the current environment.",
        )

    inchikey = _mol_to_inchikey(mol).upper()
    if inchikey:
        exact_result = resolve_full_inchikey_query(query=inchikey, state=state, top_k=top_k, resolution_path="smiles -> exact_structure")
        if exact_result.pathways:
            exact_result.query = query
            return exact_result

    index = ensure_arabidopsis_structure_index(state, verbose=False)
    query_fp = _morgan_fingerprint_from_mol(mol)
    if query_fp is None:
        return QueryResult(
            query=query,
            match_type="",
            resolution_path="smiles",
            mappings=[],
            pathways=[],
            expanded=[],
            note="Could not derive a Morgan fingerprint from the input SMILES.",
        )

    scored: list[NeighborHit] = []
    for record in index["records"]:
        if record.fingerprint is None:
            continue
        similarity = DataStructs.TanimotoSimilarity(query_fp, record.fingerprint)
        if similarity >= SMILES_NEIGHBOR_MIN:
            scored.append(NeighborHit(compound_id=record.compound_id, similarity=similarity))

    scored.sort(key=lambda item: item.similarity, reverse=True)
    neighbors = scored[:MAX_STRUCTURE_NEIGHBORS]
    if not neighbors:
        return QueryResult(
            query=query,
            match_type="",
            resolution_path="smiles -> structural_neighbor",
            mappings=[],
            pathways=[],
            expanded=[],
            note=f"No Arabidopsis structural neighbors met the Tanimoto >= {SMILES_NEIGHBOR_MIN:.2f} threshold.",
        )
    return build_query_result(
        query=query,
        match_type="structural_neighbor",
        resolution_path="smiles -> structural_neighbor",
        neighbors=neighbors,
        state=state,
        top_k=top_k,
    )


def resolve_recovered_chebi_candidates(
    query: str,
    candidates: list[ChEBIExactCandidate],
    state: dict[str, Any],
    top_k: int,
) -> QueryResult:
    if Chem is None or DataStructs is None:
        return QueryResult(
            query=query,
            match_type="",
            resolution_path="plain_text -> chebi_exact",
            mappings=[],
            pathways=[],
            expanded=[],
            note="RDKit is unavailable, so recovered ChEBI entities cannot be projected back to Arabidopsis structures.",
        )

    chebi_structures = ensure_chebi_structure_lookup(state, verbose=False)
    arab_index = ensure_arabidopsis_structure_index(state, verbose=False)
    exact_neighbors: list[NeighborHit] = []
    structural_neighbors: list[NeighborHit] = []

    for candidate in candidates:
        query_record = chebi_structures.get(candidate.compound_id)
        if query_record is None:
            continue

        if query_record.standard_inchi_key:
            exact_records = arab_index["by_full_inchikey"].get(query_record.standard_inchi_key, [])
            for record in exact_records:
                exact_neighbors.append(
                    NeighborHit(
                        compound_id=record.compound_id,
                        similarity=1.0,
                        recovered_chebi_id=candidate.chebi_accession,
                    )
                )
        if exact_neighbors:
            continue

        block = query_record.inchikey_block
        if block and query_record.formula_key:
            for record in arab_index["by_block"].get(block, []):
                if record.formula_key and record.formula_key == query_record.formula_key:
                    structural_neighbors.append(
                        NeighborHit(
                            compound_id=record.compound_id,
                            similarity=BLOCK_SIMILARITY,
                            recovered_chebi_id=candidate.chebi_accession,
                        )
                    )

        if query_record.smiles:
            mol = _mol_from_smiles(query_record.smiles)
            query_fp = _morgan_fingerprint_from_mol(mol)
            if query_fp is None:
                continue
            for record in arab_index["records"]:
                if record.fingerprint is None:
                    continue
                similarity = DataStructs.TanimotoSimilarity(query_fp, record.fingerprint)
                if similarity < RECOVERED_NEIGHBOR_MIN:
                    continue
                if not query_record.formula_key or not record.formula_key or query_record.formula_key != record.formula_key:
                    continue
                if not _is_charge_compatible(query_record, record):
                    continue
                if not _is_mass_compatible(query_record, record):
                    continue
                structural_neighbors.append(
                    NeighborHit(
                        compound_id=record.compound_id,
                        similarity=similarity,
                        recovered_chebi_id=candidate.chebi_accession,
                    )
                )

    if exact_neighbors:
        return build_query_result(
            query=query,
            match_type="recovered_from_chebi",
            resolution_path="plain_text -> chebi_exact -> exact_structure",
            neighbors=exact_neighbors,
            state=state,
            top_k=top_k,
            projection_type="exact_structure",
        )

    structural_neighbors.sort(key=lambda item: item.similarity, reverse=True)
    if structural_neighbors:
        return build_query_result(
            query=query,
            match_type="recovered_from_chebi",
            resolution_path="plain_text -> chebi_exact -> structural_neighbor",
            neighbors=structural_neighbors[:MAX_STRUCTURE_NEIGHBORS],
            state=state,
            top_k=top_k,
            projection_type="structural_neighbor",
        )

    return QueryResult(
        query=query,
        match_type="",
        resolution_path="plain_text -> chebi_exact",
        mappings=[],
        pathways=[],
        expanded=[],
        recovered_chebi_id=candidates[0].chebi_accession if candidates else "",
        note=(
            "Recovered a strict ChEBI entity, but it did not meet the high-confidence "
            "Arabidopsis structure projection thresholds."
        ),
    )


# ---------------------------------------------------------------------------
# Main query entrypoint
# ---------------------------------------------------------------------------


def run_query(
    query: str,
    state: dict[str, Any],
    top_k: int = 10,
    no_fuzzy: bool = False,
) -> QueryResult:
    """Run a full query: resolve name/structure → find mappings → rank pathways."""
    text = query.strip()
    if not text:
        return QueryResult(
            query=query,
            match_type="",
            resolution_path="empty",
            mappings=[],
            pathways=[],
            expanded=[],
            note="Input query is empty.",
        )

    if FULL_INCHIKEY_RE.fullmatch(text):
        return resolve_full_inchikey_query(text, state, top_k=top_k, resolution_path="inchikey -> exact_structure")

    if BLOCK_QUERY_RE.fullmatch(text):
        return resolve_block_query(text, state, top_k=top_k)

    smiles_match = SMILES_QUERY_RE.fullmatch(text)
    if smiles_match is not None:
        return resolve_smiles_query(smiles_match.group(1).strip(), state, top_k=top_k)

    mappings = resolve_query(text, state["name_index"])
    if mappings:
        direct_mappings = _dedupe_mappings(mappings[:5])
        neighbors = [NeighborHit(compound_id=record.compound_id, similarity=1.0) for record in mappings[:5]]
        return build_query_result(
            query=text,
            match_type="direct",
            resolution_path="plain_text -> arabidopsis_direct",
            neighbors=neighbors,
            state=state,
            top_k=top_k,
            include_expanded=True,
            base_mappings=direct_mappings,
            expanded_compound_ids=[record.compound_id for record in direct_mappings],
        )

    did_you_mean: tuple[str, ...] = ()
    if not no_fuzzy:
        fuzzy = fuzzy_match_arabidopsis(text, state)
        did_you_mean = fuzzy.did_you_mean
        if fuzzy.records:
            fuzzy_mappings = _dedupe_mappings(fuzzy.records[:5])
            neighbors = [NeighborHit(compound_id=record.compound_id, similarity=1.0) for record in fuzzy.records[:5]]
            return build_query_result(
                query=text,
                match_type="fuzzy",
                resolution_path="plain_text -> arabidopsis_fuzzy",
                neighbors=neighbors,
                state=state,
                top_k=top_k,
                include_expanded=True,
                did_you_mean=fuzzy.did_you_mean,
                base_mappings=fuzzy_mappings,
                expanded_compound_ids=[record.compound_id for record in fuzzy_mappings],
            )

    chebi_candidates = recover_chebi_exact_candidates(text, state, verbose=False)
    if chebi_candidates:
        result = resolve_recovered_chebi_candidates(text, chebi_candidates, state, top_k=top_k)
        if did_you_mean and not result.did_you_mean:
            result.did_you_mean = did_you_mean
        return result

    note = "No AraCyc/PlantCyc mapping found."
    if did_you_mean:
        note = "No exact Arabidopsis or strict ChEBI recovery succeeded."
    return QueryResult(
        query=text,
        match_type="",
        resolution_path="plain_text",
        mappings=[],
        pathways=[],
        expanded=[],
        did_you_mean=did_you_mean,
        note=note,
    )


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------


def print_result(result: QueryResult, top_k: int = 10) -> None:
    """Pretty-print query results."""

    print(f"\n{'='*70}")
    print(f"Query: {result.query}")
    print(f"{'='*70}")

    if result.match_type:
        print(f"match_type: {result.match_type}")
        if result.projection_type:
            print(f"projection_type: {result.projection_type}")
        print(f"resolution_path: {result.resolution_path}")
    elif result.resolution_path:
        print(f"resolution_path: {result.resolution_path}")

    if result.matched_compound:
        print(f"matched_compound: {result.matched_compound}")
    if result.recovered_chebi_id:
        print(f"recovered_chebi_id: {result.recovered_chebi_id}")
    if result.neighbor_similarity and result.match_type in {"exact_structure", "structural_neighbor"}:
        print(f"neighbor_similarity: {result.neighbor_similarity:.3f}")

    if result.note:
        print(f"note: {result.note}")

    if result.did_you_mean:
        print("did_you_mean:")
        for suggestion in result.did_you_mean:
            print(f"  - {suggestion}")

    if not result.mappings and not result.pathways and not result.expanded:
        return

    if result.mappings:
        print(f"\nAraCyc/PlantCyc Mappings ({len(result.mappings)} shown):")
        for index, mapping in enumerate(result.mappings[:5], 1):
            xref = " [ChEBI xref]" if mapping.chebi_xref_direct else ""
            structure = " [structure]" if mapping.structure_validated else ""
            print(f"  {index}. {mapping.chebi_accession} ({mapping.chebi_name})")
            print(f"     -> {mapping.aracyc_compound_id} ({mapping.aracyc_common_name}) [{mapping.source_db}]")
            print(f"     method={mapping.match_method}, score={mapping.match_score:.3f}{xref}{structure}")
            print(f"     pathways: {mapping.pathway_count}")

    if result.pathways:
        print(f"\nRanked Pathways (top {min(top_k, len(result.pathways))}):")
        for index, pathway in enumerate(result.pathways[:top_k], 1):
            genes = f", genes={pathway.gene_count}" if pathway.gene_count else ""
            ec = f", EC={pathway.ec_numbers}" if pathway.ec_numbers else ""
            similarity = ""
            if pathway.support_similarity < 0.999:
                similarity = f", similarity={pathway.support_similarity:.3f}"
            print(f"  #{index} [{pathway.confidence_level}] {pathway.pathway_name}")
            print(f"     score={pathway.score:.3f}, category={pathway.pathway_category}{genes}{ec}{similarity}")
            print(f"     via CHEBI:{pathway.compound_id} -> {pathway.aracyc_compound_id} [{pathway.source_db}, {pathway.match_method}]")
    else:
        print("\n  No pathways found in primary chain.")

    if result.expanded:
        print(f"\nExpanded Pathways [experimental] ({len(result.expanded)}):")
        for expanded in result.expanded[:5]:
            print(f"  - {expanded.pathway_name} (ml_score={expanded.ml_score:.3f}, {expanded.ml_confidence})")


# ---------------------------------------------------------------------------
# State loading
# ---------------------------------------------------------------------------


def load_preprocessed_state(workdir: Path, verbose: bool = True) -> dict[str, Any]:
    """Load the active preprocessed indexes for query-time use."""
    preprocessed_dir = workdir / "outputs" / "preprocessed"
    outputs_dir = workdir / "outputs"

    name_to_aracyc_path = preprocessed_dir / "name_to_aracyc_index.tsv"
    pathway_output_path = outputs_dir / "chebi_pathways_aracyc_refactored.tsv"
    expanded_path = preprocessed_dir / "ml_pathway_predictions.tsv"

    if verbose:
        print(f"Loading AraCyc indexes from {preprocessed_dir}...", flush=True)

    if not name_to_aracyc_path.exists():
        print(f"  Warning: {name_to_aracyc_path} not found. Run preprocess_all.py first.", file=sys.stderr)
        return {
            "workdir": workdir,
            "name_index": {},
            "compound_index": {},
            "pathway_index": {},
            "expanded_index": {},
            "arabidopsis_name_texts": (),
            "chebi_name_lookup": None,
            "chebi_structure_lookup": None,
            "arabidopsis_structure_index": None,
        }

    name_index, compound_index = load_name_to_aracyc(name_to_aracyc_path)
    pathway_index = load_pathway_output(pathway_output_path)
    expanded_index = load_expanded_predictions(expanded_path)

    if verbose:
        print(f"  Name index: {len(name_index)} entries", flush=True)
        print(f"  Compound index: {len(compound_index)} compounds", flush=True)
        print(f"  Pathway index: {len(pathway_index)} compounds with pathways", flush=True)
        print(f"  Expanded index: {len(expanded_index)} compounds", flush=True)

    return {
        "workdir": workdir,
        "name_index": name_index,
        "compound_index": compound_index,
        "pathway_index": pathway_index,
        "expanded_index": expanded_index,
        "arabidopsis_name_texts": tuple(sorted(name_index)),
        "chebi_name_lookup": None,
        "chebi_structure_lookup": None,
        "arabidopsis_structure_index": None,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def repl(state: dict[str, Any], top_k: int, no_fuzzy: bool) -> None:
    """Interactive REPL for pathway queries."""
    print("\nAraCyc-first pathway query REPL. Type a compound name or 'quit' to exit.")
    while True:
        try:
            query = input("\nquery> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not query or query.lower() in {"quit", "exit", "q"}:
            break
        result = run_query(query, state, top_k=top_k, no_fuzzy=no_fuzzy)
        print_result(result, top_k)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Query AraCyc-first pathway mappings.")
    parser.add_argument("--workdir", default=".", help="Workspace root directory.")
    parser.add_argument("--query", "-q", default="", help="Single query (non-interactive).")
    parser.add_argument("--top-k", type=int, default=10, help="Max pathways to show.")
    parser.add_argument("--no-fuzzy", action="store_true", help="Disable typo correction for plain-text queries.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    workdir = Path(args.workdir).resolve()
    state = load_preprocessed_state(workdir)

    if args.query:
        result = run_query(args.query, state, top_k=args.top_k, no_fuzzy=args.no_fuzzy)
        print_result(result, args.top_k)
    else:
        repl(state, args.top_k, args.no_fuzzy)


if __name__ == "__main__":
    main()
