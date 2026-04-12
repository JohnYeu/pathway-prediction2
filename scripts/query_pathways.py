#!/usr/bin/env python3
"""Interactive and one-shot query interface for the preprocessed name-first flow."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from process_chebi_to_pathways_v2 import (
    ALIAS_SOURCE_WEIGHTS,
    CHAR_EDIT2_BASE_SCORE,
    GENERIC_KEGG_MAPS,
    MIN_CHAR_FUZZY_COMPACT_LEN,
    build_variants,
    char_edit_distance_at_most_two,
    formula_sets_compatible,
    normalize_name,
)


PREPROCESSED_DIRNAME = "preprocessed"
VARIANT_ORDER = ("exact", "compact", "singular", "stereo_stripped")
VARIANT_PRIORITY = {
    "exact": 1.00,
    "compact": 0.97,
    "singular": 0.95,
    "stereo_stripped": 0.93,
}
STEREO_TOKENS = {
    "d",
    "l",
    "dl",
    "ld",
    "r",
    "s",
    "rs",
    "sr",
    "cis",
    "trans",
    "alpha",
    "beta",
    "gamma",
    "delta",
}
DERIVATIVE_TOKENS = {
    "phosphate",
    "diphosphate",
    "triphosphate",
    "sulfate",
    "sulfonate",
    "glucoside",
    "glycoside",
    "acetyl",
    "methyl",
    "ethyl",
    "ester",
    "amide",
    "glucuronide",
    "palmitoyl",
    "oleoyl",
}
SALT_TOKENS = {
    "sodium",
    "disodium",
    "trisodium",
    "potassium",
    "dipotassium",
    "tripotassium",
    "calcium",
    "magnesium",
    "hydrochloride",
    "hydrobromide",
    "hydroiodide",
}
HYDRATE_TOKENS = {
    "hydrate",
    "monohydrate",
    "dihydrate",
    "trihydrate",
    "hemihydrate",
    "sesquihydrate",
}
GENERIC_CLASS_TOKENS = {
    "lipid",
    "lipids",
    "compound",
    "compounds",
    "flavonoid",
    "flavonoids",
    "glucosinolate",
    "glucosinolates",
    "alkaloid",
    "alkaloids",
    "sterol",
    "sterols",
}
LOCANT_RE = re.compile(r"\b\d+\b")


@dataclass(slots=True)
class StructureRecord:
    formula_keys: frozenset[str]
    inchi_key_prefixes: frozenset[str]
    inchi_key_fulls: frozenset[str]
    evidence_sources: tuple[str, ...]


@dataclass(slots=True)
class NameSemantics:
    stereo_tokens: frozenset[str]
    locants: frozenset[str]
    derivative_tokens: frozenset[str]
    salt_tokens: frozenset[str]
    hydrate_tokens: frozenset[str]
    generic_class_tokens: frozenset[str]
    acid_base_signature: str

    @property
    def has_sensitive_tokens(self) -> bool:
        return bool(
            self.stereo_tokens
            or self.locants
            or self.derivative_tokens
            or self.salt_tokens
            or self.hydrate_tokens
            or self.generic_class_tokens
            or self.acid_base_signature
        )


@dataclass(slots=True)
class MatchValidation:
    compatible: bool
    formula_validated: bool = False
    inchi_prefix_validated: bool = False
    inchi_full_validated: bool = False
    semantic_validated: bool = False
    block_reason: str = ""


@dataclass(slots=True)
class NameRecord:
    compound_id: str
    chebi_accession: str
    canonical_name: str
    alias: str
    source_type: str
    is_primary_name: bool
    exact_name: str
    compact_name: str
    singular_name: str
    stereo_stripped_name: str


@dataclass(slots=True)
class MappingRecord:
    compound_id: str
    chebi_accession: str
    canonical_name: str
    canonical_exact_name: str
    canonical_compact_name: str
    canonical_singular_name: str
    canonical_stereo_stripped_name: str
    kegg_compound_id: str
    kegg_primary_name: str
    mapping_score: float
    mapping_confidence_level: str
    mapping_method: str
    direct_kegg_xref: bool
    has_structure_evidence: bool
    used_pubchem_synonym: bool
    best_alias: str
    best_alias_source: str
    best_variant: str
    evidence_count: int
    external_sources: tuple[str, ...]
    bridge_method: str
    bridge_confidence: str
    bridge_source_db: str
    arabidopsis_supported: bool
    mapping_reason: str


@dataclass(slots=True)
class PathwayAnnotation:
    pathway_id: str
    pathway_target_type: str
    map_pathway_id: str
    ath_pathway_id: str
    pathway_name: str
    map_id: str
    kegg_name: str
    brite_l1: str
    brite_l2: str
    brite_l3: str
    kegg_vstamp: str
    pathway_group: str
    pathway_category: str
    map_pathway_compound_count: int
    ath_gene_count: int
    go_top_terms_json: str
    go_best_term: str
    go_best_fdr: float
    go_vstamp: str
    plant_context_tags: tuple[str, ...]
    plant_evidence_sources: tuple[str, ...]
    aracyc_evidence_score: float
    reactome_matches: str
    plant_reactome_matches_json: str
    plant_reactome_best_id: str
    plant_reactome_best_name: str
    plant_reactome_best_category: str
    plant_reactome_best_description: str
    plant_reactome_alignment_score: float
    plant_reactome_alignment_confidence: str
    plant_reactome_tags: tuple[str, ...]
    plant_reactome_vstamp: str
    annotation_confidence: str


@dataclass(slots=True)
class PlantEvidence:
    plant_support_source: str
    plant_pathway_ids: tuple[str, ...]
    plant_support_examples: str
    plant_support_bonus: float
    plant_support_gene_count: int


@dataclass(slots=True)
class PathwayRoleEvidence:
    relation_vstamp: str
    direct_link: bool
    support_reaction_count: int
    support_rids: tuple[str, ...]
    has_substrate_role: bool
    has_product_role: bool
    has_both_role: bool
    cofactor_like: bool
    role_summary: str
    reaction_role_score: float


@dataclass(slots=True)
class ResolvedName:
    compound_id: str
    chebi_accession: str
    canonical_name: str
    matched_alias: str
    alias_source: str
    match_stage: str
    matched_variant: str
    resolution_score: float
    formula_validated: bool
    inchi_prefix_validated: bool
    inchi_full_validated: bool
    semantic_validated: bool
    formula_blocked_candidates: int
    semantic_blocked_candidates: int
    structure_blocked_candidates: int


@dataclass(slots=True)
class RankedPathway:
    pathway_rank: int
    pathway_target_id: str
    pathway_target_type: str
    pathway_name: str
    score: float
    confidence_level: str
    support_kegg_compound_ids: tuple[str, ...]
    support_kegg_names: tuple[str, ...]
    relation_vstamp: str
    support_reaction_count: int
    role_summary: str
    brite_summary: str
    go_best_term: str
    plant_context_tags: tuple[str, ...]
    plant_reactome_best_category: str
    annotation_confidence: str
    reason: str


@dataclass(slots=True)
class ExpandedPathway:
    """One ML-scored expanded pathway candidate loaded at query time."""

    compound_id: str
    chebi_accession: str
    pathway_id: str
    pathway_name: str
    pathway_source: str
    candidate_origin: str
    ml_score: float
    ml_confidence: str
    reason: str


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for one-shot or interactive querying."""

    parser = argparse.ArgumentParser(description="Query pathways from preprocessed indexes without rescanning raw refs.")
    parser.add_argument("--workdir", default=".", help="Workspace containing outputs/preprocessed/.")
    parser.add_argument("--name", help="One-shot metabolite name query.")
    parser.add_argument("--top-k", type=int, default=3, help="Number of top pathways to print.")
    return parser.parse_args()


def ensure_exists(path: Path) -> None:
    """Raise a clear error when a required preprocessed file is missing."""

    if not path.exists():
        raise FileNotFoundError(f"Missing required preprocessed file: {path}")


def preprocessed_dir(workdir: Path) -> Path:
    """Return the preprocessed output directory."""

    return workdir / "outputs" / PREPROCESSED_DIRNAME


def load_name_indexes(path: Path):
    """Load standard-name and alias indexes from the preprocessed name table."""

    standard_indexes = {variant: defaultdict(dict) for variant in VARIANT_ORDER}
    alias_indexes = {variant: defaultdict(dict) for variant in VARIANT_ORDER}
    primary_records = {}

    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            record = NameRecord(
                compound_id=row["compound_id"],
                chebi_accession=row["chebi_accession"],
                canonical_name=row["canonical_name"],
                alias=row["alias"],
                source_type=row["source_type"],
                is_primary_name=row["is_primary_name"].lower() == "true",
                exact_name=row["exact_name"],
                compact_name=row["compact_name"],
                singular_name=row["singular_name"],
                stereo_stripped_name=row["stereo_stripped_name"],
            )
            target_indexes = standard_indexes if record.is_primary_name else alias_indexes
            for variant_name, field_name in (
                ("exact", "exact_name"),
                ("compact", "compact_name"),
                ("singular", "singular_name"),
                ("stereo_stripped", "stereo_stripped_name"),
            ):
                value = getattr(record, field_name)
                if not value:
                    continue
                existing = target_indexes[variant_name][value].get(record.compound_id)
                if existing is None or ALIAS_SOURCE_WEIGHTS.get(record.source_type, 0.0) > ALIAS_SOURCE_WEIGHTS.get(existing.source_type, 0.0):
                    target_indexes[variant_name][value][record.compound_id] = record
            if record.is_primary_name:
                primary_records[record.compound_id] = record
    return standard_indexes, alias_indexes, primary_records


def normalize_inchi_key_prefix(value: str) -> str:
    """Return the first connectivity block of an InChIKey."""

    text = (value or "").strip().upper()
    if not text:
        return ""
    return text.split("-", 1)[0]


def summarize_structure_record(
    formula_keys: set[str],
    inchi_key_prefixes: set[str],
    inchi_key_fulls: set[str],
    evidence_sources: set[str],
) -> StructureRecord:
    """Create an immutable structure summary for one normalized name."""

    return StructureRecord(
        formula_keys=frozenset(value for value in formula_keys if value),
        inchi_key_prefixes=frozenset(value for value in inchi_key_prefixes if value),
        inchi_key_fulls=frozenset(value for value in inchi_key_fulls if value),
        evidence_sources=tuple(sorted(value for value in evidence_sources if value)),
    )


def load_structure_indexes(path: Path):
    """Load exact-name and compact-name structure lookups."""

    exact_formula_keys = defaultdict(set)
    compact_formula_keys = defaultdict(set)
    exact_inchi_key_prefixes = defaultdict(set)
    compact_inchi_key_prefixes = defaultdict(set)
    exact_inchi_key_fulls = defaultdict(set)
    compact_inchi_key_fulls = defaultdict(set)
    exact_sources = defaultdict(set)
    compact_sources = defaultdict(set)
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            formula_keys = {value for value in row["formula_keys"].split(";") if value}
            inchi_key_prefixes = {value for value in row.get("inchi_key_prefixes", "").split(";") if value}
            inchi_key_fulls = {value for value in row.get("inchi_key_fulls", "").split(";") if value}
            evidence_sources = {value for value in row.get("evidence_sources", "").split(";") if value}
            if row["exact_name"]:
                exact_formula_keys[row["exact_name"]].update(formula_keys)
                exact_inchi_key_prefixes[row["exact_name"]].update(inchi_key_prefixes)
                exact_inchi_key_fulls[row["exact_name"]].update(inchi_key_fulls)
                exact_sources[row["exact_name"]].update(evidence_sources)
            if row["compact_name"]:
                compact_formula_keys[row["compact_name"]].update(formula_keys)
                compact_inchi_key_prefixes[row["compact_name"]].update(inchi_key_prefixes)
                compact_inchi_key_fulls[row["compact_name"]].update(inchi_key_fulls)
                compact_sources[row["compact_name"]].update(evidence_sources)
    exact_index = {
        key: summarize_structure_record(
            exact_formula_keys[key],
            exact_inchi_key_prefixes[key],
            exact_inchi_key_fulls[key],
            exact_sources[key],
        )
        for key in exact_formula_keys.keys() | exact_inchi_key_prefixes.keys() | exact_inchi_key_fulls.keys()
    }
    compact_index = {
        key: summarize_structure_record(
            compact_formula_keys[key],
            compact_inchi_key_prefixes[key],
            compact_inchi_key_fulls[key],
            compact_sources[key],
        )
        for key in compact_formula_keys.keys() | compact_inchi_key_prefixes.keys() | compact_inchi_key_fulls.keys()
    }
    return exact_index, compact_index


def infer_acid_base_signature(tokens: tuple[str, ...]) -> str:
    """Return a conservative acid/base signature from the normalized name."""

    if not tokens:
        return ""
    if "acid" in tokens:
        return "acid"
    last_token = tokens[-1]
    if last_token.endswith("ate") and last_token not in DERIVATIVE_TOKENS:
        return "anion_like"
    return ""


def extract_name_semantics(text: str) -> NameSemantics:
    """Extract chemistry-sensitive name semantics used to block false typo fixes."""

    normalized = normalize_name(text)
    tokens = tuple(token for token in normalized.split() if token)
    return NameSemantics(
        stereo_tokens=frozenset(token for token in tokens if token in STEREO_TOKENS),
        locants=frozenset(LOCANT_RE.findall(normalized)),
        derivative_tokens=frozenset(token for token in tokens if token in DERIVATIVE_TOKENS),
        salt_tokens=frozenset(token for token in tokens if token in SALT_TOKENS),
        hydrate_tokens=frozenset(token for token in tokens if token in HYDRATE_TOKENS),
        generic_class_tokens=frozenset(token for token in tokens if token in GENERIC_CLASS_TOKENS),
        acid_base_signature=infer_acid_base_signature(tokens),
    )


def semantics_compatible(query_semantics: NameSemantics, candidate_semantics: NameSemantics) -> MatchValidation:
    """Reject weak matches when chemistry-sensitive name semantics disagree."""

    for left, right, reason in (
        (query_semantics.stereo_tokens, candidate_semantics.stereo_tokens, "stereo_conflict"),
        (query_semantics.locants, candidate_semantics.locants, "locant_conflict"),
        (query_semantics.derivative_tokens, candidate_semantics.derivative_tokens, "derivative_conflict"),
        (query_semantics.salt_tokens, candidate_semantics.salt_tokens, "salt_conflict"),
        (query_semantics.hydrate_tokens, candidate_semantics.hydrate_tokens, "hydrate_conflict"),
        (query_semantics.generic_class_tokens, candidate_semantics.generic_class_tokens, "generic_class_conflict"),
    ):
        if left != right and (left or right):
            return MatchValidation(compatible=False, block_reason=reason)
    if query_semantics.acid_base_signature != candidate_semantics.acid_base_signature and (
        query_semantics.acid_base_signature or candidate_semantics.acid_base_signature
    ):
        return MatchValidation(compatible=False, block_reason="acid_base_conflict")
    return MatchValidation(compatible=True, semantic_validated=True)


def lookup_structure_record(variants: dict[str, str], exact_index, compact_index) -> StructureRecord | None:
    """Resolve the richest available structure summary for one name."""

    records = []
    if variants["exact"] in exact_index:
        records.append(exact_index[variants["exact"]])
    if variants["compact"] in compact_index:
        records.append(compact_index[variants["compact"]])
    if not records:
        return None
    formula_keys = set()
    inchi_key_prefixes = set()
    inchi_key_fulls = set()
    evidence_sources = set()
    for record in records:
        formula_keys.update(record.formula_keys)
        inchi_key_prefixes.update(record.inchi_key_prefixes)
        inchi_key_fulls.update(record.inchi_key_fulls)
        evidence_sources.update(record.evidence_sources)
    return summarize_structure_record(formula_keys, inchi_key_prefixes, inchi_key_fulls, evidence_sources)


def validate_match_strength(
    query: str,
    query_variants: dict[str, str],
    candidate_text: str,
    variant_name: str,
    structure_index_provider,
) -> MatchValidation:
    """Validate weak name matches with semantics, formula, and InChIKey gates."""

    query_semantics = extract_name_semantics(query)
    candidate_semantics = extract_name_semantics(candidate_text)
    semantic_result = semantics_compatible(query_semantics, candidate_semantics)
    if not semantic_result.compatible:
        return semantic_result

    requires_structure_gate = variant_name in {"stereo_stripped", "compact_formula_edit2"} or query_semantics.has_sensitive_tokens or candidate_semantics.has_sensitive_tokens
    if not requires_structure_gate:
        return semantic_result

    exact_index, compact_index = structure_index_provider()
    query_structure = lookup_structure_record(query_variants, exact_index, compact_index)
    candidate_structure = lookup_structure_record(build_variants(candidate_text), exact_index, compact_index)

    query_formula_keys = set(query_structure.formula_keys) if query_structure else set()
    candidate_formula_keys = set(candidate_structure.formula_keys) if candidate_structure else set()
    if not formula_sets_compatible(query_formula_keys, candidate_formula_keys):
        return MatchValidation(compatible=False, block_reason="formula_conflict")

    result = MatchValidation(
        compatible=True,
        semantic_validated=True,
        formula_validated=bool(query_formula_keys and candidate_formula_keys and query_formula_keys & candidate_formula_keys),
    )
    query_full = set(query_structure.inchi_key_fulls) if query_structure else set()
    candidate_full = set(candidate_structure.inchi_key_fulls) if candidate_structure else set()
    if query_full and candidate_full:
        if query_full & candidate_full:
            result.inchi_full_validated = True
            result.inchi_prefix_validated = True
            return result
        return MatchValidation(compatible=False, block_reason="inchi_full_conflict")

    query_prefixes = set(query_structure.inchi_key_prefixes) if query_structure else set()
    candidate_prefixes = set(candidate_structure.inchi_key_prefixes) if candidate_structure else set()
    if query_prefixes and candidate_prefixes:
        if query_prefixes & candidate_prefixes:
            result.inchi_prefix_validated = True
            return result
        return MatchValidation(compatible=False, block_reason="inchi_prefix_conflict")
    return result


def load_name_to_kegg(path: Path):
    """Load selected KEGG mappings indexed by canonical standard-name variants.

    The query flow is intentionally name-first. Step 1 resolves the user's text
    to a canonical standard name, and step 2 uses that canonical name to obtain
    KEGG compound IDs. compound_id is still preserved as provenance, but it is
    no longer the primary lookup key for the query flow.
    """

    mappings_by_variant = {variant: defaultdict(list) for variant in VARIANT_ORDER}
    best_mapping_score_by_compound = defaultdict(float)
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            canonical_variants = build_variants(row["canonical_name"])
            record = MappingRecord(
                compound_id=row["compound_id"],
                chebi_accession=row["chebi_accession"],
                canonical_name=row["canonical_name"],
                canonical_exact_name=row.get("canonical_exact_name") or canonical_variants["exact"],
                canonical_compact_name=row.get("canonical_compact_name") or canonical_variants["compact"],
                canonical_singular_name=row.get("canonical_singular_name") or canonical_variants["singular"],
                canonical_stereo_stripped_name=row.get("canonical_stereo_stripped_name") or canonical_variants["stereo_stripped"],
                kegg_compound_id=row["kegg_compound_id"],
                kegg_primary_name=row["kegg_primary_name"],
                mapping_score=float(row["mapping_score"]),
                mapping_confidence_level=row["mapping_confidence_level"],
                mapping_method=row["mapping_method"],
                direct_kegg_xref=row["direct_kegg_xref"].lower() == "true",
                has_structure_evidence=row["has_structure_evidence"].lower() == "true",
                used_pubchem_synonym=row["used_pubchem_synonym"].lower() == "true",
                best_alias=row["best_alias"],
                best_alias_source=row["best_alias_source"],
                best_variant=row["best_variant"],
                evidence_count=int(row["evidence_count"] or 0),
                external_sources=tuple(value for value in row["external_sources"].split(";") if value),
                bridge_method=row.get("bridge_method", ""),
                bridge_confidence=row.get("bridge_confidence", ""),
                bridge_source_db=row.get("bridge_source_db", ""),
                arabidopsis_supported=row.get("arabidopsis_supported", "").lower() == "true",
                mapping_reason=row["mapping_reason"],
            )
            for variant_name, value in (
                ("exact", record.canonical_exact_name),
                ("compact", record.canonical_compact_name),
                ("singular", record.canonical_singular_name),
                ("stereo_stripped", record.canonical_stereo_stripped_name),
            ):
                if value:
                    mappings_by_variant[variant_name][value].append(record)
            best_mapping_score_by_compound[record.compound_id] = max(
                best_mapping_score_by_compound[record.compound_id],
                record.mapping_score,
            )
    for variant_name in mappings_by_variant:
        for value in mappings_by_variant[variant_name]:
            mappings_by_variant[variant_name][value].sort(
                key=lambda item: (item.mapping_score, item.direct_kegg_xref, item.has_structure_evidence),
                reverse=True,
            )
    return (
        {variant_name: dict(index) for variant_name, index in mappings_by_variant.items()},
        dict(best_mapping_score_by_compound),
    )


def load_structure_to_kegg(path: Path):
    """Load exact InChIKey-driven KEGG mappings keyed by compound_id."""

    mappings = defaultdict(list)
    if not path.exists():
        return {}
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            mappings[row["compound_id"]].append(
                MappingRecord(
                    compound_id=row["compound_id"],
                    chebi_accession=row["chebi_accession"],
                    canonical_name=row["canonical_name"],
                    canonical_exact_name=row["canonical_exact_name"],
                    canonical_compact_name=row["canonical_compact_name"],
                    canonical_singular_name=row["canonical_singular_name"],
                    canonical_stereo_stripped_name=row["canonical_stereo_stripped_name"],
                    kegg_compound_id=row["kegg_compound_id"],
                    kegg_primary_name=row["kegg_primary_name"],
                    mapping_score=float(row["mapping_score"]),
                    mapping_confidence_level=row["mapping_confidence_level"],
                    mapping_method=row["mapping_method"],
                    direct_kegg_xref=row["direct_kegg_xref"].lower() == "true",
                    has_structure_evidence=row["has_structure_evidence"].lower() == "true",
                    used_pubchem_synonym=row["used_pubchem_synonym"].lower() == "true",
                    best_alias=row["best_alias"],
                    best_alias_source=row["best_alias_source"],
                    best_variant=row["best_variant"],
                    evidence_count=int(row["evidence_count"] or 0),
                    external_sources=tuple(value for value in row["external_sources"].split(";") if value),
                    bridge_method=row.get("bridge_method", ""),
                    bridge_confidence=row.get("bridge_confidence", ""),
                    bridge_source_db=row.get("bridge_source_db", ""),
                    arabidopsis_supported=row.get("arabidopsis_supported", "").lower() == "true",
                    mapping_reason=row["mapping_reason"],
                )
            )
    for compound_id in mappings:
        mappings[compound_id].sort(
            key=lambda item: (item.mapping_score, item.has_structure_evidence, item.kegg_compound_id),
            reverse=True,
        )
    return dict(mappings)


def load_plant_to_kegg_bridge(path: Path):
    """Load precomputed PMN -> KEGG bridge candidates keyed by compound_id."""

    mappings = defaultdict(list)
    if not path.exists():
        return {}, {}
    best_mapping_score_by_compound = defaultdict(float)
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            record = MappingRecord(
                compound_id=row["compound_id"],
                chebi_accession=row["chebi_accession"],
                canonical_name=row["canonical_name"],
                canonical_exact_name=build_variants(row["canonical_name"])["exact"],
                canonical_compact_name=build_variants(row["canonical_name"])["compact"],
                canonical_singular_name=build_variants(row["canonical_name"])["singular"],
                canonical_stereo_stripped_name=build_variants(row["canonical_name"])["stereo_stripped"],
                kegg_compound_id=row["kegg_cid"],
                kegg_primary_name=row.get("kegg_primary_name", ""),
                mapping_score=float(row["bridge_score"]),
                mapping_confidence_level=row["bridge_confidence"],
                mapping_method=row["bridge_method"],
                direct_kegg_xref=row["bridge_method"] == "plant_direct_kegg_xref",
                has_structure_evidence=row["has_structure_evidence"].lower() == "true",
                used_pubchem_synonym=False,
                best_alias="",
                best_alias_source="",
                best_variant="",
                evidence_count=max(1, len([value for value in row["supporting_ids"].split(";") if value])),
                external_sources=(row["plant_db"],),
                bridge_method=row["bridge_method"],
                bridge_confidence=row["bridge_confidence"],
                bridge_source_db=row["plant_db"],
                arabidopsis_supported=row["arabidopsis_supported"].lower() == "true",
                mapping_reason=row["bridge_reason"],
            )
            mappings[row["compound_id"]].append(record)
            best_mapping_score_by_compound[row["compound_id"]] = max(
                best_mapping_score_by_compound[row["compound_id"]],
                record.mapping_score,
            )
    for compound_id in mappings:
        mappings[compound_id].sort(
            key=lambda item: (
                item.mapping_score,
                item.arabidopsis_supported,
                item.has_structure_evidence,
                item.kegg_compound_id,
            ),
            reverse=True,
        )
    return dict(mappings), dict(best_mapping_score_by_compound)


def load_compound_to_pathway(path: Path):
    """Load KEGG compound -> map pathway links."""

    links = defaultdict(list)
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            links[row["kegg_compound_id"]].append(row["map_pathway_id"])
    for kegg_compound_id in links:
        links[kegg_compound_id] = sorted(set(links[kegg_compound_id]))
    return dict(links)


def load_compound_to_pathway_roles(path: Path):
    """Load the aggregated step-3 reaction-role evidence."""

    roles = defaultdict(dict)
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            roles[row["kegg_compound_id"]][row["map_pathway_id"]] = PathwayRoleEvidence(
                relation_vstamp=row["relation_vstamp"],
                direct_link=row["direct_link"].lower() == "true",
                support_reaction_count=int(row["support_reaction_count"] or 0),
                support_rids=tuple(value for value in row["support_rids"].split(";") if value),
                has_substrate_role=row["has_substrate_role"].lower() == "true",
                has_product_role=row["has_product_role"].lower() == "true",
                has_both_role=row["has_both_role"].lower() == "true",
                cofactor_like=row["cofactor_like"].lower() == "true",
                role_summary=row["role_summary"],
                reaction_role_score=float(row["reaction_role_score"] or 0.0),
            )
    return {cid: dict(items) for cid, items in roles.items()}


def load_map_to_ath(path: Path):
    """Load map pathway -> ath pathway conversion records."""

    table = {}
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            table[row["map_pathway_id"]] = row["ath_pathway_id"]
    return table


def load_pathway_annotations(path: Path):
    """Load unified pathway annotations keyed by the final target pathway id."""

    annotations = {}
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            annotations[row["pathway_id"]] = PathwayAnnotation(
                pathway_id=row["pathway_id"],
                pathway_target_type=row["pathway_target_type"],
                map_pathway_id=row["map_pathway_id"],
                ath_pathway_id=row["ath_pathway_id"],
                pathway_name=row["pathway_name"],
                map_id=row.get("map_id", row["map_pathway_id"]),
                kegg_name=row.get("kegg_name", row["pathway_name"]),
                brite_l1=row.get("brite_l1", row.get("pathway_group", "")),
                brite_l2=row.get("brite_l2", row.get("pathway_category", "")),
                brite_l3=row.get("brite_l3", row["pathway_name"]),
                kegg_vstamp=row.get("kegg_vstamp", ""),
                pathway_group=row["pathway_group"],
                pathway_category=row["pathway_category"],
                map_pathway_compound_count=int(row["map_pathway_compound_count"] or 0),
                ath_gene_count=int(row["ath_gene_count"] or 0),
                reactome_matches=row["reactome_matches"],
                go_top_terms_json=row.get("go_top_terms_json", ""),
                go_best_term=row.get("go_best_term", ""),
                go_best_fdr=float(row.get("go_best_fdr", "") or 0.0),
                go_vstamp=row.get("go_vstamp", ""),
                plant_context_tags=tuple(value for value in row.get("plant_context_tags", "").split(";") if value),
                plant_evidence_sources=tuple(value for value in row.get("plant_evidence_sources", "").split(";") if value),
                aracyc_evidence_score=float(row.get("aracyc_evidence_score", "") or 0.0),
                plant_reactome_matches_json=row.get("plant_reactome_matches_json", ""),
                plant_reactome_best_id=row.get("plant_reactome_best_id", ""),
                plant_reactome_best_name=row.get("plant_reactome_best_name", ""),
                plant_reactome_best_category=row.get("plant_reactome_best_category", ""),
                plant_reactome_best_description=row.get("plant_reactome_best_description", ""),
                plant_reactome_alignment_score=float(row.get("plant_reactome_alignment_score", "") or 0.0),
                plant_reactome_alignment_confidence=row.get("plant_reactome_alignment_confidence", ""),
                plant_reactome_tags=tuple(value for value in row.get("plant_reactome_tags", "").split(";") if value),
                plant_reactome_vstamp=row.get("plant_reactome_vstamp", ""),
                annotation_confidence=row.get("annotation_confidence", "low"),
            )
    return annotations


def load_plant_evidence(path: Path):
    """Load plant support rows keyed by compound_id and normalized pathway name."""

    evidence = defaultdict(dict)
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            item = PlantEvidence(
                plant_support_source=row["plant_support_source"],
                plant_pathway_ids=tuple(value for value in row.get("plant_pathway_ids", "").split(";") if value),
                plant_support_examples=row["plant_support_examples"],
                plant_support_bonus=float(row["plant_support_bonus"]),
                plant_support_gene_count=int(row["plant_support_gene_count"] or 0),
            )
            current = evidence[row["compound_id"]].get(row["pathway_name_normalized"])
            if current is None or item.plant_support_bonus > current.plant_support_bonus:
                evidence[row["compound_id"]][row["pathway_name_normalized"]] = item
    return {compound_id: dict(values) for compound_id, values in evidence.items()}


def load_expanded_predictions(path: Path) -> dict[str, list[ExpandedPathway]]:
    """Load ML-scored expanded pathway predictions, keyed by compound_id."""

    predictions: dict[str, list[ExpandedPathway]] = defaultdict(list)
    if not path.exists():
        return dict(predictions)
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            predictions[row["compound_id"]].append(ExpandedPathway(
                compound_id=row["compound_id"],
                chebi_accession=row["chebi_accession"],
                pathway_id=row["pathway_id"],
                pathway_name=row["pathway_name"],
                pathway_source=row["pathway_source"],
                candidate_origin=row["candidate_origin"],
                ml_score=float(row["ml_score"]),
                ml_confidence=row["ml_confidence"],
                reason=row["reason"],
            ))
    # Sort each compound's predictions by ml_score descending
    for compound_id in predictions:
        predictions[compound_id].sort(key=lambda x: x.ml_score, reverse=True)
    return dict(predictions)


def choose_best_hit(hits, variant_name, stage, best_mapping_score_by_compound, validations):
    """Pick the best compound when a standard-name or alias tier produced hits."""

    best = None
    for record in hits.values():
        validation = validations.get(record.compound_id, MatchValidation(compatible=True))
        mapping_score = best_mapping_score_by_compound.get(record.compound_id, 0.0)
        score = (
            VARIANT_PRIORITY[variant_name]
            + (0.16 if stage == "standard_name" else 0.08)
            + ALIAS_SOURCE_WEIGHTS.get(record.source_type, 0.0)
            + (mapping_score * 0.10)
            + (0.03 if validation.inchi_full_validated else 0.0)
            + (0.02 if validation.inchi_prefix_validated else 0.0)
            + (0.01 if validation.formula_validated else 0.0)
        )
        candidate = ResolvedName(
            compound_id=record.compound_id,
            chebi_accession=record.chebi_accession,
            canonical_name=record.canonical_name,
            matched_alias=record.alias,
            alias_source=record.source_type,
            match_stage=stage,
            matched_variant=variant_name,
            resolution_score=score,
            formula_validated=validation.formula_validated,
            inchi_prefix_validated=validation.inchi_prefix_validated,
            inchi_full_validated=validation.inchi_full_validated,
            semantic_validated=validation.semantic_validated,
            formula_blocked_candidates=0,
            semantic_blocked_candidates=0,
            structure_blocked_candidates=0,
        )
        if best is None or (candidate.resolution_score, candidate.compound_id) > (best.resolution_score, best.compound_id):
            best = candidate
    return best


def resolve_name(
    query: str,
    standard_indexes,
    alias_indexes,
    primary_records,
    structure_index_provider,
    best_mapping_score_by_compound,
) -> ResolvedName | None:
    """Run the table-style step-1 name resolution logic."""

    variants = build_variants(query)

    for variant_name in VARIANT_ORDER:
        value = variants[variant_name]
        if not value:
            continue
        hits = standard_indexes[variant_name].get(value)
        if hits:
            validations = {}
            semantic_blocked = 0
            structure_blocked = 0
            for compound_id, record in hits.items():
                validation = validate_match_strength(query, variants, record.alias, variant_name, structure_index_provider)
                if validation.compatible:
                    validations[compound_id] = validation
                elif validation.block_reason.endswith("_conflict") and validation.block_reason.startswith(("stereo", "locant", "derivative", "salt", "hydrate", "generic", "acid_base")):
                    semantic_blocked += 1
                else:
                    structure_blocked += 1
            if validations:
                best = choose_best_hit(hits, variant_name, "standard_name", best_mapping_score_by_compound, validations)
                if best:
                    best.semantic_blocked_candidates = semantic_blocked
                    best.structure_blocked_candidates = structure_blocked
                    return best

    for variant_name in VARIANT_ORDER:
        value = variants[variant_name]
        if not value:
            continue
        hits = alias_indexes[variant_name].get(value)
        if hits:
            validations = {}
            semantic_blocked = 0
            structure_blocked = 0
            for compound_id, record in hits.items():
                validation = validate_match_strength(query, variants, record.alias, variant_name, structure_index_provider)
                if validation.compatible:
                    validations[compound_id] = validation
                elif validation.block_reason.endswith("_conflict") and validation.block_reason.startswith(("stereo", "locant", "derivative", "salt", "hydrate", "generic", "acid_base")):
                    semantic_blocked += 1
                else:
                    structure_blocked += 1
            if validations:
                best = choose_best_hit(hits, variant_name, "alias_name", best_mapping_score_by_compound, validations)
                if best:
                    best.semantic_blocked_candidates = semantic_blocked
                    best.structure_blocked_candidates = structure_blocked
                    return best

    compact_query = variants["compact"]
    if not compact_query or len(compact_query) < MIN_CHAR_FUZZY_COMPACT_LEN:
        return None

    best = None
    formula_blocked = 0
    semantic_blocked = 0
    structure_blocked = 0
    for primary in primary_records.values():
        if not primary.compact_name:
            continue
        if abs(len(compact_query) - len(primary.compact_name)) > 2:
            continue
        if not char_edit_distance_at_most_two(compact_query, primary.compact_name):
            continue
        validation = validate_match_strength(query, variants, primary.alias, "compact_formula_edit2", structure_index_provider)
        if not validation.compatible:
            if validation.block_reason == "formula_conflict":
                formula_blocked += 1
            elif validation.block_reason.endswith("_conflict") and validation.block_reason.startswith(("stereo", "locant", "derivative", "salt", "hydrate", "generic", "acid_base")):
                semantic_blocked += 1
            else:
                structure_blocked += 1
            continue
        mapping_score = best_mapping_score_by_compound.get(primary.compound_id, 0.0)
        score = CHAR_EDIT2_BASE_SCORE + 0.16 + (mapping_score * 0.10)
        score += 0.02 if validation.formula_validated else 0.0
        score += 0.03 if validation.inchi_full_validated else 0.0
        score += 0.02 if validation.inchi_prefix_validated else 0.0
        candidate = ResolvedName(
            compound_id=primary.compound_id,
            chebi_accession=primary.chebi_accession,
            canonical_name=primary.canonical_name,
            matched_alias=primary.alias,
            alias_source=primary.source_type,
            match_stage="fuzzy_typo_correction",
            matched_variant="compact_formula_edit2",
            resolution_score=score,
            formula_validated=validation.formula_validated,
            inchi_prefix_validated=validation.inchi_prefix_validated,
            inchi_full_validated=validation.inchi_full_validated,
            semantic_validated=validation.semantic_validated,
            formula_blocked_candidates=formula_blocked,
            semantic_blocked_candidates=semantic_blocked,
            structure_blocked_candidates=structure_blocked,
        )
        if best is None or (candidate.resolution_score, candidate.compound_id) > (best.resolution_score, best.compound_id):
            best = candidate
    return best


def lookup_mappings_for_standard_name(canonical_name: str, mappings_by_name):
    """Resolve a canonical standard name to KEGG mapping rows from step 2."""

    variants = build_variants(canonical_name)
    selected = {}
    for variant_name in VARIANT_ORDER:
        value = variants[variant_name]
        if not value:
            continue
        for record in mappings_by_name[variant_name].get(value, []):
            key = (record.compound_id, record.kegg_compound_id)
            existing = selected.get(key)
            if existing is None or (
                record.mapping_score,
                record.direct_kegg_xref,
                record.has_structure_evidence,
            ) > (
                existing.mapping_score,
                existing.direct_kegg_xref,
                existing.has_structure_evidence,
            ):
                selected[key] = record
        if selected:
            break
    return sorted(
        selected.values(),
        key=lambda item: (item.mapping_score, item.direct_kegg_xref, item.has_structure_evidence),
        reverse=True,
    )


def rank_pathways(
    resolved: ResolvedName,
    mappings,
    compound_to_pathway,
    compound_to_pathway_roles,
    map_to_ath,
    pathway_annotations,
    plant_evidence,
    top_k: int,
) -> list[RankedPathway]:
    """Run query-side steps 2 through 7 using only preprocessed indexes."""

    if not mappings:
        return []

    max_map_count = max((item.map_pathway_compound_count for item in pathway_annotations.values()), default=1)
    max_gene_count = max((item.ath_gene_count for item in pathway_annotations.values()), default=1)
    aggregated = {}
    for mapping in mappings:
        evidence_by_name = plant_evidence.get(mapping.compound_id, {})
        for map_pathway_id in compound_to_pathway.get(mapping.kegg_compound_id, []):
            role = compound_to_pathway_roles.get(mapping.kegg_compound_id, {}).get(map_pathway_id)
            ath_pathway_id = map_to_ath.get(map_pathway_id, "")
            target_id = ath_pathway_id or map_pathway_id
            annotation = pathway_annotations.get(target_id)
            if annotation is None:
                continue

            specificity = 1 - math.log1p(annotation.map_pathway_compound_count) / math.log1p(max_map_count)
            gene_ratio = math.log1p(annotation.ath_gene_count) / math.log1p(max_gene_count) if annotation.ath_gene_count else 0.0
            plant = evidence_by_name.get(normalize_name(annotation.pathway_name))
            plant_bonus = plant.plant_support_bonus if plant else 0.0
            generic_penalty = -0.06 if annotation.map_pathway_id in GENERIC_KEGG_MAPS or annotation.pathway_category == "Global and overview maps" else 0.0
            contributions = {
                "mapping_confidence": round(0.45 * mapping.mapping_score, 3),
                "pathway_specificity": round(0.15 * specificity, 3),
                "ath_exists": 0.12 if annotation.ath_pathway_id else 0.0,
                "ath_gene_support": round(0.10 * gene_ratio, 3),
                "direct_kegg_xref": 0.05 if mapping.direct_kegg_xref else 0.0,
                "structure_support": 0.05 if mapping.has_structure_evidence else 0.0,
                "direct_link_bonus": 0.12 if role and role.direct_link else 0.0,
                "substrate_role_bonus": 0.12 if role and role.has_substrate_role else 0.0,
                "product_role_bonus": 0.06 if role and role.has_product_role else 0.0,
                "both_role_bonus": 0.03 if role and role.has_both_role else 0.0,
                "cofactor_penalty": -0.15 if role and role.cofactor_like else 0.0,
                "reaction_support_bonus": round(min(0.12, 0.05 * math.log1p(role.support_reaction_count)), 3) if role else 0.0,
                "plantcyc_support": round(plant_bonus, 3),
                "generic_pathway_penalty": generic_penalty,
                "map_fallback_penalty": -0.10 if not annotation.ath_pathway_id else 0.0,
                "annotation_quality_bonus": 0.03 if annotation.annotation_confidence == "high" else 0.01 if annotation.annotation_confidence == "medium" else 0.0,
                "plant_reactome_alignment_bonus": 0.02 if annotation.plant_reactome_alignment_confidence == "high" else 0.01 if annotation.plant_reactome_alignment_confidence == "medium" else 0.0,
            }
            score = max(0.0, min(sum(contributions.values()), 0.999))
            confidence = "high" if score >= 0.70 and annotation.ath_pathway_id and annotation.ath_gene_count > 0 else "medium" if score >= 0.40 else "low"
            if role and role.cofactor_like and not (role.has_substrate_role or role.has_product_role or role.has_both_role):
                confidence = "medium" if confidence == "high" else confidence
            positive_features = [f"{k}={v:.3f}" for k, v in sorted(contributions.items(), key=lambda item: item[1], reverse=True) if v > 0][:3]
            negative_features = [f"{k}={v:.3f}" for k, v in sorted(contributions.items(), key=lambda item: item[1]) if v < 0][:2]
            reason_parts = [
                f"{confidence} confidence",
                f"score {score:.3f}",
                f"mapping {mapping.kegg_compound_id} ({mapping.kegg_primary_name}) via {mapping.mapping_method}",
            ]
            if role and role.direct_link:
                reason_parts.append("direct KEGG compound-pathway link")
            if role and role.role_summary:
                reason_parts.append(f"reaction roles: {role.role_summary}")
            if role and role.cofactor_like:
                reason_parts.append("cofactor-like participation lowered the score")
            if role and role.relation_vstamp:
                reason_parts.append(f"relation_vstamp={role.relation_vstamp}")
            if annotation.pathway_target_type == "map_fallback":
                reason_parts.append("used map fallback because no ath-specific pathway was available")
            if annotation.ath_gene_count:
                reason_parts.append(f"ath gene support: {annotation.ath_gene_count}")
            if annotation.brite_l1 or annotation.brite_l2 or annotation.brite_l3:
                reason_parts.append(
                    "brite="
                    + " / ".join(value for value in (annotation.brite_l1, annotation.brite_l2, annotation.brite_l3) if value)
                )
            if annotation.go_best_term:
                if annotation.go_best_fdr > 0:
                    reason_parts.append(f"GO BP: {annotation.go_best_term} (FDR {annotation.go_best_fdr:.3g})")
                else:
                    reason_parts.append(f"GO BP: {annotation.go_best_term}")
            if annotation.plant_context_tags:
                reason_parts.append(f"plant tags: {','.join(annotation.plant_context_tags)}")
            if plant:
                reason_parts.append(f"{plant.plant_support_source} support: {plant.plant_support_examples}")
            if annotation.plant_reactome_best_category:
                reason_parts.append(
                    f"Plant Reactome: {annotation.plant_reactome_best_category}"
                    + (f" ({annotation.plant_reactome_alignment_confidence})" if annotation.plant_reactome_alignment_confidence else "")
                )
            if positive_features:
                reason_parts.append(f"top positive features: {';'.join(positive_features)}")
            if negative_features:
                reason_parts.append(f"top negative features: {';'.join(negative_features)}")
            reason = "; ".join(reason_parts)

            entry = aggregated.setdefault(
                target_id,
                {
                    "score": -1.0,
                    "confidence": confidence,
                    "reason": reason,
                    "annotation": annotation,
                    "support_ids": set(),
                    "support_names": set(),
                    "relation_vstamp": role.relation_vstamp if role else "",
                    "support_reaction_count": role.support_reaction_count if role else 0,
                    "role_summary": role.role_summary if role else "",
                    "brite_summary": " / ".join(value for value in (annotation.brite_l1, annotation.brite_l2, annotation.brite_l3) if value),
                },
            )
            if score > float(entry["score"]):
                entry["score"] = score
                entry["confidence"] = confidence
                entry["reason"] = reason
                entry["annotation"] = annotation
                entry["relation_vstamp"] = role.relation_vstamp if role else ""
                entry["support_reaction_count"] = role.support_reaction_count if role else 0
                entry["role_summary"] = role.role_summary if role else ""
                entry["brite_summary"] = " / ".join(value for value in (annotation.brite_l1, annotation.brite_l2, annotation.brite_l3) if value)
            entry["support_ids"].add(mapping.kegg_compound_id)
            entry["support_names"].add(mapping.kegg_primary_name)

    ranked = sorted(
        aggregated.values(),
        key=lambda item: (
            float(item["score"]),
            item["annotation"].ath_gene_count,
            -item["annotation"].map_pathway_compound_count,
            item["annotation"].pathway_id,
        ),
        reverse=True,
    )

    rows = []
    for rank, entry in enumerate(ranked[:top_k], start=1):
        annotation = entry["annotation"]
        rows.append(
            RankedPathway(
                pathway_rank=rank,
                pathway_target_id=annotation.pathway_id,
                pathway_target_type=annotation.pathway_target_type,
                pathway_name=annotation.pathway_name,
                score=float(entry["score"]),
                confidence_level=str(entry["confidence"]),
                support_kegg_compound_ids=tuple(sorted(entry["support_ids"])),
                support_kegg_names=tuple(sorted(entry["support_names"])),
                relation_vstamp=str(entry["relation_vstamp"]),
                support_reaction_count=int(entry["support_reaction_count"]),
                role_summary=str(entry["role_summary"]),
                brite_summary=str(entry["brite_summary"]),
                go_best_term=annotation.go_best_term,
                plant_context_tags=annotation.plant_context_tags,
                plant_reactome_best_category=annotation.plant_reactome_best_category,
                annotation_confidence=annotation.annotation_confidence,
                reason=str(entry["reason"]),
            )
        )
    return rows


def rank_pmn_fallback_pathways(
    resolved: ResolvedName,
    plant_evidence,
    top_k: int,
    had_kegg_mapping: bool,
) -> list[RankedPathway]:
    """Use direct AraCyc/PlantCyc support when KEGG coverage is missing.

    This is the main mitigation for step-2 limitation #1: compounds that are
    plausibly resolved by name but have no reliable KEGG compound mapping (or
    no KEGG pathway links) should still return plant-specific pathways from PMN.
    """

    evidence_by_name = plant_evidence.get(resolved.compound_id, {})
    if not evidence_by_name:
        return []

    ranked = []
    for normalized_name, plant in evidence_by_name.items():
        example_names = [value for value in plant.plant_support_examples.split(";") if value]
        pathway_name = example_names[0] if example_names else normalized_name
        pathway_id = plant.plant_pathway_ids[0] if plant.plant_pathway_ids else f"{plant.plant_support_source}:{normalized_name}"

        gene_bonus = 0.10 if plant.plant_support_gene_count else 0.0
        dense_gene_bonus = 0.05 if plant.plant_support_gene_count >= 3 else 0.0
        source_bonus = 0.07 if plant.plant_support_source == "AraCyc" else 0.05
        score = min(0.999, 0.45 + plant.plant_support_bonus + gene_bonus + dense_gene_bonus + source_bonus)
        confidence = "medium" if plant.plant_support_gene_count > 0 or plant.plant_support_source == "AraCyc" else "low"

        reason_parts = [
            "direct PMN fallback",
            f"score {score:.3f}",
            f"{plant.plant_support_source} compound-pathway support",
        ]
        if not had_kegg_mapping:
            reason_parts.append("used PMN fallback because PlantCyc/AraCyc candidate had no KEGG bridge")
        else:
            reason_parts.append("used because no KEGG pathway links were available")
        if plant.plant_pathway_ids:
            reason_parts.append(f"pathway ids: {','.join(plant.plant_pathway_ids[:3])}")
        if plant.plant_support_gene_count:
            reason_parts.append(f"PMN gene support: {plant.plant_support_gene_count}")
        if plant.plant_support_examples:
            reason_parts.append(f"pathway names: {plant.plant_support_examples}")

        ranked.append(
            RankedPathway(
                pathway_rank=0,
                pathway_target_id=pathway_id,
                pathway_target_type="pmn_direct",
                pathway_name=pathway_name,
                score=score,
                confidence_level=confidence,
                support_kegg_compound_ids=(),
                support_kegg_names=(),
                relation_vstamp="",
                support_reaction_count=0,
                role_summary="",
                brite_summary="",
                go_best_term="",
                plant_context_tags=(),
                plant_reactome_best_category="",
                annotation_confidence=confidence,
                reason="; ".join(reason_parts),
            )
        )

    ranked.sort(
        key=lambda item: (
            item.score,
            item.confidence_level == "high",
            item.pathway_target_id,
        ),
        reverse=True,
    )
    return [
        RankedPathway(
            pathway_rank=index,
            pathway_target_id=item.pathway_target_id,
            pathway_target_type=item.pathway_target_type,
            pathway_name=item.pathway_name,
            score=item.score,
            confidence_level=item.confidence_level,
            support_kegg_compound_ids=item.support_kegg_compound_ids,
            support_kegg_names=item.support_kegg_names,
            relation_vstamp=item.relation_vstamp,
            support_reaction_count=item.support_reaction_count,
            role_summary=item.role_summary,
            brite_summary=item.brite_summary,
            go_best_term=item.go_best_term,
            plant_context_tags=item.plant_context_tags,
            plant_reactome_best_category=item.plant_reactome_best_category,
            annotation_confidence=item.annotation_confidence,
            reason=item.reason,
        )
        for index, item in enumerate(ranked[:top_k], start=1)
    ]


def print_result(query: str, resolved: ResolvedName, mappings, pathways: list[RankedPathway], top_k: int, expanded: list[ExpandedPathway] | None = None) -> None:
    """Render a one-shot query result to the terminal."""

    print(f"Query: {query}")
    print(f"Standard name: {resolved.canonical_name}")
    print(f"Matched alias: {resolved.matched_alias}")
    print(f"Match stage: {resolved.match_stage}")
    print(f"Match variant: {resolved.matched_variant}")
    if resolved.match_stage == "fuzzy_typo_correction":
        print(f"Formula validated: {'yes' if resolved.formula_validated else 'no'}")
        print(f"InChIKey prefix validated: {'yes' if resolved.inchi_prefix_validated else 'no'}")
        print(f"Full InChIKey validated: {'yes' if resolved.inchi_full_validated else 'no'}")
        print(f"Semantic validation passed: {'yes' if resolved.semantic_validated else 'no'}")
        print(f"Formula-blocked candidates: {resolved.formula_blocked_candidates}")
        print(f"Semantic-blocked candidates: {resolved.semantic_blocked_candidates}")
        print(f"Structure-blocked candidates: {resolved.structure_blocked_candidates}")
    if mappings:
        print(
            "KEGG compound IDs: "
            + "; ".join(f"{mapping.kegg_compound_id} ({mapping.kegg_primary_name})" for mapping in mappings)
        )
    else:
        print("KEGG compound IDs: none")

    # Primary pathways
    if not pathways:
        print("Top pathways: none")
    else:
        print(f"Top {top_k} pathways:")
        for pathway in pathways:
            print(
                f"{pathway.pathway_rank}. {pathway.pathway_name} [{pathway.pathway_target_id}] "
                f"score={pathway.score:.3f} confidence={pathway.confidence_level}"
            )
            if pathway.brite_summary:
                print(f"   brite: {pathway.brite_summary}")
            if pathway.go_best_term:
                print(f"   go_best_term: {pathway.go_best_term}")
            if pathway.plant_context_tags:
                print(f"   plant_context_tags: {', '.join(pathway.plant_context_tags)}")
            if pathway.plant_reactome_best_category:
                print(f"   plant_reactome_category: {pathway.plant_reactome_best_category}")
            if pathway.relation_vstamp:
                print(f"   relation_vstamp: {pathway.relation_vstamp}")
            if pathway.support_reaction_count:
                print(f"   support_reactions: {pathway.support_reaction_count}")
            if pathway.role_summary:
                print(f"   role_summary: {pathway.role_summary}")
            if pathway.annotation_confidence:
                print(f"   annotation_confidence: {pathway.annotation_confidence}")
            if pathway.pathway_target_type == "pmn_direct":
                print("   source: direct AraCyc/PlantCyc (PMN) fallback")
            print(f"   reason: {pathway.reason}")

    # Expanded pathways (shown when primary is empty OR always as separate block)
    if expanded:
        if not pathways:
            print("Expanded pathways [experimental]:")
        else:
            print("--- Expanded pathways [experimental] ---")
        for i, ep in enumerate(expanded, 1):
            print(
                f"  E{i}. {ep.pathway_name} [{ep.pathway_source}] "
                f"ml_score={ep.ml_score:.3f} confidence={ep.ml_confidence}"
            )
            print(f"      origin: {ep.candidate_origin}")
            print(f"      reason: {ep.reason}")


def load_preprocessed_state(workdir: Path, verbose: bool = True):
    """Load all query-time indexes from outputs/preprocessed/."""

    data_dir = preprocessed_dir(workdir)
    paths = {
        "name_normalization_index": data_dir / "name_normalization_index.tsv",
        "name_to_formula_index": data_dir / "name_to_formula_index.tsv",
        "name_to_kegg_index": data_dir / "name_to_kegg_index.tsv",
        "compound_structure_kegg_index": data_dir / "compound_structure_kegg_index.tsv",
        "plant_to_kegg_bridge": data_dir / "plant_to_kegg_bridge.tsv",
        "compound_to_pathway_index": data_dir / "compound_to_pathway_index.tsv",
        "compound_to_pathway_role_index": data_dir / "compound_to_pathway_role_index.tsv",
        "map_to_ath_index": data_dir / "map_to_ath_index.tsv",
        "pathway_annotation_index": data_dir / "pathway_annotation_index.tsv",
        "plant_evidence_index": data_dir / "plant_evidence_index.tsv",
        "preprocess_metadata": data_dir / "preprocess_metadata.json",
    }
    for path in paths.values():
        ensure_exists(path)
    if verbose:
        print("Loading preprocessed indexes...", flush=True)
    with paths["preprocess_metadata"].open(encoding="utf-8") as handle:
        metadata = json.load(handle)
    if verbose:
        print("- metadata", flush=True)
    if verbose:
        print("- name normalization index", flush=True)
    name_indexes = load_name_indexes(paths["name_normalization_index"])
    if verbose:
        print("- name-to-KEGG index", flush=True)
    mappings_by_name, best_mapping_score_by_compound = load_name_to_kegg(paths["name_to_kegg_index"])
    if verbose:
        print("- structure-to-KEGG index", flush=True)
    structure_mappings_by_compound = load_structure_to_kegg(paths["compound_structure_kegg_index"])
    if verbose:
        print("- PMN-to-KEGG bridge index", flush=True)
    bridge_mappings_by_compound, best_bridge_score_by_compound = load_plant_to_kegg_bridge(paths["plant_to_kegg_bridge"])
    for compound_id, score in best_bridge_score_by_compound.items():
        best_mapping_score_by_compound[compound_id] = max(best_mapping_score_by_compound.get(compound_id, 0.0), score)
    if verbose:
        print("- KEGG compound-to-pathway index", flush=True)
    compound_to_pathway = load_compound_to_pathway(paths["compound_to_pathway_index"])
    if verbose:
        print("- KEGG compound-to-pathway role index", flush=True)
    compound_to_pathway_roles = load_compound_to_pathway_roles(paths["compound_to_pathway_role_index"])
    if verbose:
        print("- map-to-ath index", flush=True)
    map_to_ath = load_map_to_ath(paths["map_to_ath_index"])
    if verbose:
        print("- pathway annotation index", flush=True)
    pathway_annotations = load_pathway_annotations(paths["pathway_annotation_index"])
    if verbose:
        print("- plant evidence index", flush=True)
    plant_evidence = load_plant_evidence(paths["plant_evidence_index"])
    expanded_predictions_path = data_dir / "ml_pathway_predictions.tsv"
    expanded_predictions = load_expanded_predictions(expanded_predictions_path) if expanded_predictions_path.exists() else {}
    if verbose:
        n_expanded = sum(len(v) for v in expanded_predictions.values())
        print(f"- expanded predictions: {n_expanded} entries for {len(expanded_predictions)} compounds", flush=True)
    if verbose:
        print("Preprocessed indexes loaded.", flush=True)
    return {
        "metadata": metadata,
        "name_indexes": name_indexes,
        "structure_indexes": None,
        "structure_index_path": paths["name_to_formula_index"],
        "verbose": verbose,
        "mappings_by_name": mappings_by_name,
        "structure_mappings_by_compound": structure_mappings_by_compound,
        "bridge_mappings_by_compound": bridge_mappings_by_compound,
        "best_mapping_score_by_compound": best_mapping_score_by_compound,
        "compound_to_pathway": compound_to_pathway,
        "compound_to_pathway_roles": compound_to_pathway_roles,
        "map_to_ath": map_to_ath,
        "pathway_annotations": pathway_annotations,
        "plant_evidence": plant_evidence,
        "expanded_predictions": expanded_predictions,
    }


def get_structure_indexes(state):
    """Load the structure index lazily because only weak matches need it."""

    if state["structure_indexes"] is None:
        if state.get("verbose"):
            print("- name-to-structure index (on demand for weak-match validation)", flush=True)
        state["structure_indexes"] = load_structure_indexes(state["structure_index_path"])
    return state["structure_indexes"]


def run_query(query: str, state, top_k: int) -> tuple[ResolvedName | None, list[MappingRecord], list[RankedPathway], list[ExpandedPathway]]:
    """Run the full query pipeline for one free-text name.

    Returns (resolved, mappings, primary_pathways, expanded_pathways).
    """

    standard_indexes, alias_indexes, primary_records = state["name_indexes"]
    resolved = resolve_name(
        query,
        standard_indexes,
        alias_indexes,
        primary_records,
        lambda: get_structure_indexes(state),
        state["best_mapping_score_by_compound"],
    )
    if resolved is None:
        return None, [], [], []
    name_mappings = lookup_mappings_for_standard_name(resolved.canonical_name, state["mappings_by_name"])
    same_compound_name_mappings = [mapping for mapping in name_mappings if mapping.compound_id == resolved.compound_id]
    if same_compound_name_mappings:
        name_mappings = same_compound_name_mappings
    structure_mappings = state["structure_mappings_by_compound"].get(resolved.compound_id, [])
    bridge_mappings = state["bridge_mappings_by_compound"].get(resolved.compound_id, [])
    direct_name_mappings = [mapping for mapping in name_mappings if mapping.direct_kegg_xref]
    if direct_name_mappings:
        mappings = direct_name_mappings
    elif structure_mappings:
        mappings = structure_mappings
    elif bridge_mappings:
        mappings = bridge_mappings
    else:
        mappings = name_mappings
    deduped_mappings = {}
    for mapping in mappings:
        existing = deduped_mappings.get(mapping.kegg_compound_id)
        if existing is None or (
            mapping.mapping_score,
            mapping.direct_kegg_xref,
            mapping.has_structure_evidence,
            mapping.arabidopsis_supported,
        ) > (
            existing.mapping_score,
            existing.direct_kegg_xref,
            existing.has_structure_evidence,
            existing.arabidopsis_supported,
        ):
            deduped_mappings[mapping.kegg_compound_id] = mapping
    mappings = sorted(
        deduped_mappings.values(),
        key=lambda item: (
            item.mapping_score,
            item.direct_kegg_xref,
            item.has_structure_evidence,
            item.arabidopsis_supported,
            item.kegg_compound_id,
        ),
        reverse=True,
    )
    pathways = rank_pathways(
        resolved,
        mappings,
        state["compound_to_pathway"],
        state["compound_to_pathway_roles"],
        state["map_to_ath"],
        state["pathway_annotations"],
        state["plant_evidence"],
        top_k,
    )
    if not pathways:
        pathways = rank_pmn_fallback_pathways(
            resolved,
            state["plant_evidence"],
            top_k,
            had_kegg_mapping=bool(mappings),
        )

    # Look up expanded predictions for this compound
    expanded = state.get("expanded_predictions", {}).get(resolved.compound_id, [])[:top_k]

    return resolved, mappings, pathways, expanded


def repl(state, top_k: int) -> None:
    """Run a simple terminal REPL on top of the preprocessed indexes."""

    print("Enter a metabolite name. Type 'exit' or press Enter on an empty line to quit.")
    while True:
        try:
            query = input("metabolite> ").strip()
        except EOFError:
            print()
            break
        if not query or query.lower() == "exit":
            break
        resolved, mappings, pathways, expanded = run_query(query, state, top_k)
        if resolved is None:
            print("No matching standard name found.")
            continue
        print_result(query, resolved, mappings, pathways, top_k, expanded)


def main() -> None:
    """Load the preprocessed indexes once, then run one-shot or interactive queries."""

    args = parse_args()
    workdir = Path(args.workdir).resolve()
    state = load_preprocessed_state(workdir)
    if args.name:
        resolved, mappings, pathways, expanded = run_query(args.name, state, args.top_k)
        if resolved is None:
            print("No matching standard name found.")
            return
        print_result(args.name, resolved, mappings, pathways, args.top_k, expanded)
        return
    repl(state, args.top_k)


if __name__ == "__main__":
    main()
