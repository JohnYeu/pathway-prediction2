#!/usr/bin/env python3
"""Build a ChEBI-to-KEGG/PlantCyc-supported pathway ranking table.

The pipeline in this file is intentionally split into a few conceptual stages:

1. Normalize ChEBI names and collect aliases from multiple sources.
2. Build KEGG name indexes that support exact, alias-based, and conservative
   fuzzy matching.
3. Map each ChEBI compound to one or more KEGG compound IDs with explicit
   evidence tracking.
4. Expand selected KEGG compounds to pathways, then score pathways with a
   transparent rule-based explanation layer.

The code favors traceability over compactness: each major step keeps enough
metadata to explain why a mapping or pathway recommendation was produced.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import html
import json
import math
import re
import sys
import unicodedata
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path


csv.field_size_limit(min(sys.maxsize, 2**31 - 1))

TAG_RE = re.compile(r"<[^>]+>")
WS_RE = re.compile(r"\s+")
NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")
KEGG_ID_RE = re.compile(r"^C\d{5}$")
PUBCHEM_CID_RE = re.compile(r"^\d+$")
CAS_RE = re.compile(r"^\d{2,7}-\d{2}-\d$")
PATHWAY_LINE_RE = re.compile(r"^C\s+(\d{5})\s+(.+)$")

ALLOWED_CHEBI_ALIAS_TYPES = {"SYNONYM", "IUPAC NAME", "INN"}
VARIANT_ORDER = ("exact", "compact", "singular", "stereo_stripped")
FUZZY_VARIANT_ORDER = ("exact", "singular", "stereo_stripped")
VARIANT_BASE_SCORES = {
    "exact": 0.84,
    "compact": 0.81,
    "singular": 0.79,
    "stereo_stripped": 0.77,
}
CHAR_EDIT2_BASE_SCORE = 0.78
MAX_CHAR_FUZZY_EDITS = 2
MIN_CHAR_FUZZY_COMPACT_LEN = 6
TOKEN_EDIT1_BASE_SCORES = {
    "exact": 0.78,
    "singular": 0.76,
    "stereo_stripped": 0.74,
}
ALIAS_SOURCE_WEIGHTS = {
    "compound_name": 0.10,
    "ascii_name": 0.08,
    "chebi_synonym": 0.07,
    "chebi_iupac": 0.06,
    "chebi_inn": 0.05,
    "pubchem_synonym": 0.05,
    "plantcyc_common_name": 0.07,
    "plantcyc_synonym": 0.05,
    "aracyc_common_name": 0.08,
    "aracyc_synonym": 0.06,
    "lipidmaps_common_name": 0.08,
    "lipidmaps_systematic_name": 0.06,
    "lipidmaps_synonym": 0.05,
}
LEADING_STEREO_TOKENS = {
    "d",
    "l",
    "dl",
    "ld",
    "rs",
    "sr",
    "cis",
    "trans",
    "alpha",
    "beta",
    "gamma",
    "delta",
}
IGNORABLE_FUZZY_TOKENS = {
    "e",
    "z",
    "r",
    "s",
    "zwitterion",
    "anion",
    "cation",
    "hydrate",
    "monohydrate",
    "dihydrate",
    "trihydrate",
    "hydrochloride",
    "hydrobromide",
    "hydroiodide",
    "hemihydrate",
    "sesquihydrate",
    "sodium",
    "disodium",
    "trisodium",
    "potassium",
    "dipotassium",
    "tripotassium",
}
GENERIC_KEGG_MAPS = {"map01100", "map01110", "map01120"}
EXTERNAL_NAME_MATCH_MAX_RECORDS = 5
MAX_EXTRA_SYNONYMS_PER_RECORD = 10
MAX_PUBCHEM_SYNONYMS_PER_CID = 15
GREEK_MAP = str.maketrans(
    {
        "α": " alpha ",
        "β": " beta ",
        "γ": " gamma ",
        "δ": " delta ",
        "ε": " epsilon ",
        "κ": " kappa ",
        "λ": " lambda ",
        "μ": " mu ",
        "ω": " omega ",
        "Α": " alpha ",
        "Β": " beta ",
        "Γ": " gamma ",
        "Δ": " delta ",
        "Ε": " epsilon ",
        "Κ": " kappa ",
        "Λ": " lambda ",
        "Μ": " mu ",
        "Ω": " omega ",
    }
)


@dataclass(slots=True)
class AliasRecord:
    """One normalized alias plus the source that contributed it.

    The same raw alias is materialized into several normalized variants so the
    matcher can progressively relax from strict to looser name forms.
    """

    raw_name: str
    source_type: str
    language_code: str
    exact: str
    compact: str
    singular: str
    stereo_stripped: str


@dataclass(slots=True)
class ChEBICompound:
    """Core ChEBI row used as the starting point of the pipeline."""

    compound_id: str
    chebi_accession: str
    name: str
    ascii_name: str
    definition: str
    stars: int
    status_id: str


@dataclass(slots=True)
class StructureInfo:
    """Preferred structural metadata attached to a ChEBI compound."""

    compound_id: str
    smiles: str = ""
    standard_inchi: str = ""
    standard_inchi_key: str = ""
    formula: str = ""
    formula_key: str = ""
    monoisotopic_mass: str = ""


@dataclass(slots=True)
class XrefInfo:
    """External database identifiers already curated for one ChEBI entry."""

    kegg_ids: set[str] = field(default_factory=set)
    pubchem_cids: set[str] = field(default_factory=set)
    hmdb_ids: set[str] = field(default_factory=set)
    chemspider_ids: set[str] = field(default_factory=set)


@dataclass(slots=True)
class KeggCompound:
    """KEGG compound row plus precomputed normalized name representations."""

    compound_id: str
    primary_name: str
    aliases: list[str]
    primary_exact: str
    primary_compact: str
    primary_singular: str
    primary_stereo_stripped: str
    fuzzy_tokens: dict[str, tuple[tuple[str, ...], ...]]
    fuzzy_compacts: tuple[str, ...]


@dataclass
class CandidateMapping:
    """A single ChEBI -> KEGG candidate with aggregated evidence.

    Multiple independent signals may point to the same KEGG compound. Rather
    than overwriting earlier evidence, the pipeline accumulates all signals here
    and later decides whether the candidate is strong enough to keep.
    """

    kegg_compound_id: str
    kegg_primary_name: str
    best_score: float = 0.0
    final_score: float = 0.0
    direct_kegg_xref: bool = False
    best_alias: str = ""
    best_source_type: str = ""
    best_variant: str = ""
    evidence_count: int = 0
    reasons: list[str] = field(default_factory=list)
    methods: set[str] = field(default_factory=set)
    external_sources: set[str] = field(default_factory=set)
    primary_name_match: bool = False
    has_structure_evidence: bool = False
    used_pubchem_synonym: bool = False

    def add_evidence(
        self,
        *,
        score: float,
        method: str,
        reason: str,
        alias: str = "",
        source_type: str = "",
        variant: str = "",
        direct_kegg_xref: bool = False,
        primary_name_match: bool = False,
        external_source: str = "",
        has_structure_evidence: bool = False,
        used_pubchem_synonym: bool = False,
    ) -> None:
        """Merge one new piece of evidence into the candidate summary."""

        self.evidence_count += 1
        self.methods.add(method)
        if reason and reason not in self.reasons and len(self.reasons) < 8:
            self.reasons.append(reason)
        if score > self.best_score:
            self.best_score = score
            self.best_alias = alias or self.best_alias
            self.best_source_type = source_type or self.best_source_type
            self.best_variant = variant or self.best_variant
        self.direct_kegg_xref = self.direct_kegg_xref or direct_kegg_xref
        self.primary_name_match = self.primary_name_match or primary_name_match
        self.has_structure_evidence = self.has_structure_evidence or has_structure_evidence
        self.used_pubchem_synonym = self.used_pubchem_synonym or used_pubchem_synonym
        if external_source:
            self.external_sources.add(external_source)


@dataclass(slots=True)
class PlantCycCompound:
    """PlantCyc/AraCyc compound row plus parsed external links."""

    record_id: str
    source_db: str
    compound_id: str
    common_name: str
    synonyms: set[str] = field(default_factory=set)
    formula: str = ""
    formula_key: str = ""
    smiles: str = ""
    chebi_ids: set[str] = field(default_factory=set)
    pubchem_cids: set[str] = field(default_factory=set)
    kegg_ids: set[str] = field(default_factory=set)
    hmdb_ids: set[str] = field(default_factory=set)
    pathways: set[str] = field(default_factory=set)


@dataclass(slots=True)
class LipidMapsRecord:
    """Subset of LIPID MAPS fields that are useful for matching and support."""

    lm_id: str
    common_name: str
    systematic_name: str
    synonyms: set[str] = field(default_factory=set)
    inchi_key: str = ""
    smiles: str = ""
    formula: str = ""
    formula_key: str = ""
    pubchem_cids: set[str] = field(default_factory=set)
    kegg_ids: set[str] = field(default_factory=set)
    hmdb_ids: set[str] = field(default_factory=set)
    chebi_ids: set[str] = field(default_factory=set)
    category: str = ""
    main_class: str = ""
    sub_class: str = ""


@dataclass
class CompoundContext:
    """All alias/support information collected for one input compound."""

    all_aliases: list[AliasRecord]
    matched_plantcyc_methods: dict[str, set[str]]
    matched_lipidmaps_methods: dict[str, set[str]]
    pubchem_cids: set[str]
    aracyc_pathways: dict[str, set[str]]
    plantcyc_pathways: dict[str, set[str]]


@dataclass(slots=True)
class StoredSupportContext:
    """Compact, serializable support signals kept after mapping selection."""

    aracyc_pathways: dict[str, tuple[str, ...]]
    plantcyc_pathways: dict[str, tuple[str, ...]]
    matched_plantcyc_records: int
    matched_lipidmaps_records: int


def clean_markup(text: str) -> str:
    """Remove HTML-like markup and normalize Unicode-heavy chemistry names."""

    text = html.unescape(text or "")
    text = text.translate(GREEK_MAP)
    text = TAG_RE.sub(" ", text)
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return text


def normalize_name(text: str) -> str:
    """Create a conservative normalized name for exact-style matching."""

    text = clean_markup(text).lower()
    text = text.replace("'", "")
    text = NON_ALNUM_RE.sub(" ", text)
    text = WS_RE.sub(" ", text).strip()
    return text


def formula_key(text: str) -> str:
    """Normalize molecular formula strings so equivalent spacing collapses."""

    return re.sub(r"\s+", "", (text or "")).upper()


def singularize_token(token: str) -> str:
    """Apply a lightweight singularization pass to a single token."""

    if len(token) <= 4:
        return token
    if token.endswith("ies"):
        return token[:-3] + "y"
    if token.endswith("sses") or token.endswith("osis"):
        return token
    if token.endswith("s") and not token.endswith(("is", "us", "ss")):
        return token[:-1]
    return token


def strip_stereo_tokens(normalized: str) -> str:
    """Drop leading stereochemical markers such as L/D/cis/trans."""

    tokens = normalized.split()
    while tokens and tokens[0] in LEADING_STEREO_TOKENS:
        tokens.pop(0)
    return " ".join(tokens)


def build_variants(text: str) -> dict[str, str]:
    """Build the normalized name forms used throughout the matcher."""

    exact = normalize_name(text)
    singular = " ".join(singularize_token(token) for token in exact.split())
    stereo_stripped = strip_stereo_tokens(singular)
    return {
        "exact": exact,
        "compact": exact.replace(" ", ""),
        "singular": singular,
        "stereo_stripped": stereo_stripped,
    }


def tokenize_variant(text: str) -> tuple[str, ...]:
    """Split a normalized variant into stable token tuples."""

    return tuple(token for token in text.split() if token)


def deletion_signatures(tokens: tuple[str, ...]) -> set[tuple[str, ...]]:
    """Generate all signatures created by deleting exactly one token."""

    if len(tokens) < 2:
        return set()
    return {tokens[:index] + tokens[index + 1 :] for index in range(len(tokens))}


def char_edit_distance_at_most_one(left: str, right: str) -> bool:
    """Fast bounded edit-distance check for <= 1 character change."""

    if left == right:
        return True
    if abs(len(left) - len(right)) > 1:
        return False
    if len(left) == len(right):
        return sum(left_char != right_char for left_char, right_char in zip(left, right)) <= 1
    if len(left) > len(right):
        left, right = right, left
    left_index = 0
    right_index = 0
    skipped = False
    while left_index < len(left) and right_index < len(right):
        if left[left_index] == right[right_index]:
            left_index += 1
            right_index += 1
            continue
        if skipped:
            return False
        skipped = True
        right_index += 1
    return True


def char_edit_distance_at_most_two(left: str, right: str) -> bool:
    """Bounded Levenshtein check used by the final fuzzy fallback."""

    if left == right:
        return True
    if abs(len(left) - len(right)) > 2:
        return False
    previous = list(range(len(right) + 1))
    for left_index, left_char in enumerate(left, start=1):
        current = [left_index]
        row_min = left_index
        for right_index, right_char in enumerate(right, start=1):
            substitution_cost = 0 if left_char == right_char else 1
            value = min(
                previous[right_index] + 1,
                current[right_index - 1] + 1,
                previous[right_index - 1] + substitution_cost,
            )
            current.append(value)
            row_min = min(row_min, value)
        if row_min > 2:
            return False
        previous = current
    return previous[-1] <= 2


def generate_deletes(text: str, max_deletes: int) -> set[str]:
    """Generate deletion keys used for cheap fuzzy candidate recall."""

    deletes: set[str] = set()
    frontier = {text}
    for _ in range(max_deletes):
        next_frontier: set[str] = set()
        for item in frontier:
            if len(item) <= 1:
                continue
            for index in range(len(item)):
                deleted = item[:index] + item[index + 1 :]
                if deleted in deletes:
                    continue
                deletes.add(deleted)
                next_frontier.add(deleted)
        frontier = next_frontier
        if not frontier:
            break
    return deletes


def is_ignorable_fuzzy_token(token: str) -> bool:
    """Return True for tokens that are safe to ignore in fuzzy comparisons."""

    return token.isdigit() or token in LEADING_STEREO_TOKENS or token in IGNORABLE_FUZZY_TOKENS


def token_edit_distance_at_most_one(left: tuple[str, ...], right: tuple[str, ...]) -> bool:
    """Token-level fuzzy check kept for diagnostic and optional use."""

    if left == right:
        return True
    if abs(len(left) - len(right)) > 1:
        return False
    if len(left) == len(right):
        mismatches = [(left_token, right_token) for left_token, right_token in zip(left, right) if left_token != right_token]
        return len(mismatches) == 1 and char_edit_distance_at_most_one(mismatches[0][0], mismatches[0][1])
    if len(left) > len(right):
        left, right = right, left
    left_index = 0
    right_index = 0
    skipped_token = ""
    while left_index < len(left) and right_index < len(right):
        if left[left_index] == right[right_index]:
            left_index += 1
            right_index += 1
            continue
        if skipped_token:
            return False
        skipped_token = right[right_index]
        right_index += 1
    if not skipped_token and right_index < len(right):
        skipped_token = right[right_index]
    return bool(skipped_token) and is_ignorable_fuzzy_token(skipped_token)


def split_multi_value(text: str) -> list[str]:
    """Split semicolon/pipe/star-separated lists into individual values."""

    values: list[str] = []
    for part in re.split(r"[;|*]", text or ""):
        item = part.strip()
        if item:
            values.append(item)
    return values


def normalize_chebi_id(value: str) -> str:
    """Normalize plain numeric or mixed-form ChEBI identifiers."""

    value = value.strip()
    if not value:
        return ""
    if value.upper().startswith("CHEBI:"):
        return f"CHEBI:{value.split(':', 1)[1]}"
    if value.isdigit():
        return f"CHEBI:{value}"
    return value


def keep_pubchem_synonym(text: str) -> bool:
    """Filter PubChem synonyms down to human-readable chemical names."""

    text = text.strip()
    if len(text) < 3 or len(text) > 120:
        return False
    if text.startswith("InChI=") or text.startswith("InChIKey="):
        return False
    if CAS_RE.fullmatch(text):
        return False
    if re.fullmatch(r"(CID|SID|DTXSID|DTXCID|CHEBI|HMDB|LM)[A-Z0-9:\-]+", text, re.I):
        return False
    if re.fullmatch(r"[0-9\W_]+", text):
        return False
    if text.count(" ") == 0 and sum(ch.isdigit() for ch in text) >= 3 and sum(ch.islower() for ch in text) == 0:
        return False
    return True


def record_alias(
    alias_bucket: dict[str, AliasRecord],
    *,
    raw_name: str,
    source_type: str,
    language_code: str = "",
) -> None:
    """Insert one alias into a bucket while keeping the strongest source."""

    variants = build_variants(raw_name)
    if not variants["exact"]:
        return
    record = AliasRecord(
        raw_name=raw_name,
        source_type=source_type,
        language_code=language_code,
        exact=variants["exact"],
        compact=variants["compact"],
        singular=variants["singular"],
        stereo_stripped=variants["stereo_stripped"],
    )
    existing = alias_bucket.get(record.exact)
    if existing is None or ALIAS_SOURCE_WEIGHTS.get(record.source_type, 0.0) > ALIAS_SOURCE_WEIGHTS.get(existing.source_type, 0.0):
        alias_bucket[record.exact] = record


def ensure_exists(path: Path) -> None:
    """Fail fast when a required input file is missing."""

    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")


def load_compounds(path: Path) -> dict[str, ChEBICompound]:
    """Load the local ChEBI compound table into memory."""

    compounds: dict[str, ChEBICompound] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            compounds[row["id"]] = ChEBICompound(
                compound_id=row["id"],
                chebi_accession=row["chebi_accession"],
                name=row["name"],
                ascii_name=row["ascii_name"],
                definition=row["definition"],
                stars=int(row["stars"] or 0),
                status_id=row["status_id"],
            )
    return compounds


def load_comments_profile(path: Path) -> dict[str, int]:
    """Profile comments.tsv so the summary can explain why it is not used."""

    counter: Counter[str] = Counter()
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            counter[row["datatype"] or "<empty>"] += 1
    return dict(counter.most_common())


def load_base_aliases(
    compounds: dict[str, ChEBICompound],
    names_path: Path,
) -> dict[str, list[AliasRecord]]:
    """Build the initial alias table from ChEBI primary names and names.tsv.

    This stage intentionally stays close to ChEBI itself. External synonym
    sources are added later only after we know which records they plausibly
    belong to.
    """

    aliases: dict[str, dict[str, AliasRecord]] = {compound_id: {} for compound_id in compounds}
    for compound_id, compound in compounds.items():
        record_alias(aliases[compound_id], raw_name=compound.name, source_type="compound_name", language_code="en")
        if compound.ascii_name and compound.ascii_name != compound.name:
            record_alias(aliases[compound_id], raw_name=compound.ascii_name, source_type="ascii_name", language_code="en")

    with gzip.open(names_path, "rt", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            compound_id = row["compound_id"]
            if compound_id not in aliases:
                continue
            if row["type"] not in ALLOWED_CHEBI_ALIAS_TYPES:
                continue
            if row["language_code"] not in {"", "en"}:
                continue
            source_type = {
                "SYNONYM": "chebi_synonym",
                "IUPAC NAME": "chebi_iupac",
                "INN": "chebi_inn",
            }[row["type"]]
            record_alias(aliases[compound_id], raw_name=row["name"], source_type=source_type, language_code=row["language_code"])
            if row["ascii_name"] and row["ascii_name"] != row["name"]:
                record_alias(aliases[compound_id], raw_name=row["ascii_name"], source_type=source_type, language_code=row["language_code"])

    return {compound_id: list(alias_map.values()) for compound_id, alias_map in aliases.items()}


def load_xrefs(path: Path) -> dict[str, XrefInfo]:
    """Load curated external identifiers from ChEBI database_accession."""

    xrefs: defaultdict[str, XrefInfo] = defaultdict(XrefInfo)
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            compound_id = row["compound_id"]
            accession = row["accession_number"].strip()
            source_id = row["source_id"]
            xref = xrefs[compound_id]
            if source_id == "45" and KEGG_ID_RE.fullmatch(accession):
                xref.kegg_ids.add(accession)
            elif source_id == "68" and row["type"] == "MANUAL_X_REF" and PUBCHEM_CID_RE.fullmatch(accession):
                xref.pubchem_cids.add(accession)
            elif source_id == "35" and accession:
                xref.hmdb_ids.add(accession if accession.upper().startswith("HMDB") else f"HMDB:{accession}")
            elif source_id == "19" and accession:
                xref.chemspider_ids.add(accession)
    return dict(xrefs)


def load_formula_info(path: Path) -> dict[str, tuple[str, str]]:
    """Load preferred molecular formula/mass records for each ChEBI compound."""

    formulas: dict[str, tuple[str, str]] = {}
    preferred_status: dict[str, int] = {}
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            compound_id = row["compound_id"]
            status_rank = 0 if row["status_id"] == "1" else 1
            if compound_id in formulas and status_rank >= preferred_status[compound_id]:
                continue
            formulas[compound_id] = (row["formula"] or "", row["monoisotopic_mass"] or "")
            preferred_status[compound_id] = status_rank
    return formulas


def load_structures(path: Path, formulas: dict[str, tuple[str, str]]) -> dict[str, StructureInfo]:
    """Load the preferred ChEBI structure row and attach formula metadata."""

    structures: dict[str, StructureInfo] = {}
    preferred_default: dict[str, int] = {}
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            compound_id = row["compound_id"]
            is_default = 0 if row["default_structure"].lower() == "true" else 1
            if compound_id in structures and is_default >= preferred_default[compound_id]:
                continue
            formula, monoisotopic_mass = formulas.get(compound_id, ("", ""))
            structures[compound_id] = StructureInfo(
                compound_id=compound_id,
                smiles=row["smiles"] or "",
                standard_inchi=row["standard_inchi"] or "",
                standard_inchi_key=row["standard_inchi_key"] or "",
                formula=formula,
                formula_key=formula_key(formula),
                monoisotopic_mass=monoisotopic_mass,
            )
            preferred_default[compound_id] = is_default
    return structures


def build_name_formula_index(
    base_aliases: dict[str, list[AliasRecord]],
    structures: dict[str, StructureInfo],
    plantcyc_records: dict[str, PlantCycCompound],
    lipidmaps_records: dict[str, LipidMapsRecord],
) -> dict[str, set[str]]:
    """Create a compact-name -> formula-set lookup used by fuzzy validation.

    The key design choice is that fuzzy name similarity alone is never trusted.
    When a near-name candidate is found later, this index is consulted to see
    whether both names can be associated with the same molecular formula.
    """

    index: defaultdict[str, set[str]] = defaultdict(set)
    for compound_id, aliases in base_aliases.items():
        structure = structures.get(compound_id)
        if not structure or not structure.formula_key:
            continue
        for alias in aliases:
            if alias.compact:
                index[alias.compact].add(structure.formula_key)
    for record in plantcyc_records.values():
        if not record.formula_key:
            continue
        for name in [record.common_name, *sorted(record.synonyms)]:
            compact = build_variants(name)["compact"]
            if compact:
                index[compact].add(record.formula_key)
    for record in lipidmaps_records.values():
        if not record.formula_key:
            continue
        for name in [record.common_name, record.systematic_name, *sorted(record.synonyms)]:
            compact = build_variants(name)["compact"]
            if compact:
                index[compact].add(record.formula_key)
    return dict(index)


def load_kegg_compounds(
    path: Path,
) -> tuple[
    dict[str, KeggCompound],
    dict[str, defaultdict[str, set[str]]],
    dict[str, defaultdict[tuple[str, ...], set[str]]],
    dict[str, defaultdict[tuple[str, ...], defaultdict[int, set[str]]]],
    defaultdict[str, set[str]],
    dict[str, defaultdict[str, set[str]]],
    defaultdict[str, set[str]],
]:
    """Load KEGG compound names and precompute every lookup index we need.

    Returned structures serve different roles:
    - compounds: full KEGG compound metadata keyed by KEGG ID
    - indexes: all names (primary + aliases) for broad exact matching
    - token_* indexes: token-deletion helpers retained for conservative fuzzy
      experiments and debugging
    - compact_delete_index: all-name character-deletion recall table
    - alias_indexes: names that are not the primary KEGG standard name
    - primary_compact_delete_index: only primary names, used by the final
      typo-correction fallback so we correct toward a KEGG standard name
    """

    compounds: dict[str, KeggCompound] = {}
    indexes = {
        "exact": defaultdict(set),
        "compact": defaultdict(set),
        "singular": defaultdict(set),
        "stereo_stripped": defaultdict(set),
    }
    standard_indexes = {
        "exact": defaultdict(set),
        "compact": defaultdict(set),
        "singular": defaultdict(set),
        "stereo_stripped": defaultdict(set),
    }
    alias_indexes = {
        "exact": defaultdict(set),
        "compact": defaultdict(set),
        "singular": defaultdict(set),
        "stereo_stripped": defaultdict(set),
    }
    token_exact_indexes = {variant_name: defaultdict(set) for variant_name in FUZZY_VARIANT_ORDER}
    token_delete_indexes = {variant_name: defaultdict(lambda: defaultdict(set)) for variant_name in FUZZY_VARIANT_ORDER}
    compact_delete_index: defaultdict[str, set[str]] = defaultdict(set)
    primary_compact_delete_index: defaultdict[str, set[str]] = defaultdict(set)
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            compound_id, raw_names = line.rstrip("\n").split("\t", 1)
            aliases = [name.strip() for name in raw_names.split(";") if name.strip()]
            primary = aliases[0]
            variants = build_variants(primary)
            fuzzy_tokens: dict[str, list[tuple[str, ...]]] = {variant_name: [] for variant_name in FUZZY_VARIANT_ORDER}
            fuzzy_compacts: list[str] = []
            compounds[compound_id] = KeggCompound(
                compound_id=compound_id,
                primary_name=primary,
                aliases=aliases,
                primary_exact=variants["exact"],
                primary_compact=variants["compact"],
                primary_singular=variants["singular"],
                primary_stereo_stripped=variants["stereo_stripped"],
                fuzzy_tokens={},
                fuzzy_compacts=(),
            )
            for variant_name, variant_value in variants.items():
                if variant_value:
                    standard_indexes[variant_name][variant_value].add(compound_id)
            if variants["compact"] and len(variants["compact"]) >= MIN_CHAR_FUZZY_COMPACT_LEN:
                # Only primary names feed the typo-correction fallback. This
                # keeps fuzzy correction pointed at a canonical KEGG label.
                for deleted in generate_deletes(variants["compact"], MAX_CHAR_FUZZY_EDITS):
                    primary_compact_delete_index[deleted].add(compound_id)
            for alias in aliases:
                alias_variants = build_variants(alias)
                for variant_name, variant_value in alias_variants.items():
                    if variant_value:
                        indexes[variant_name][variant_value].add(compound_id)
                if alias != primary:
                    # Alias hits are allowed, but they are intentionally kept in
                    # a separate table so the caller can distinguish
                    # "matched KEGG standard name" from "matched a synonym".
                    for variant_name, variant_value in alias_variants.items():
                        if variant_value:
                            alias_indexes[variant_name][variant_value].add(compound_id)
                if alias_variants["compact"]:
                    fuzzy_compacts.append(alias_variants["compact"])
                    if len(alias_variants["compact"]) >= MIN_CHAR_FUZZY_COMPACT_LEN:
                        for deleted in generate_deletes(alias_variants["compact"], MAX_CHAR_FUZZY_EDITS):
                            compact_delete_index[deleted].add(compound_id)
                for variant_name in FUZZY_VARIANT_ORDER:
                    tokens = tokenize_variant(alias_variants[variant_name])
                    if len(tokens) < 2:
                        continue
                    fuzzy_tokens[variant_name].append(tokens)
                    token_exact_indexes[variant_name][tokens].add(compound_id)
                    for signature in deletion_signatures(tokens):
                        token_delete_indexes[variant_name][signature][len(tokens)].add(compound_id)
            compounds[compound_id].fuzzy_tokens = {
                variant_name: tuple(dict.fromkeys(token_lists))
                for variant_name, token_lists in fuzzy_tokens.items()
            }
            compounds[compound_id].fuzzy_compacts = tuple(dict.fromkeys(fuzzy_compacts))
    return (
        compounds,
        indexes,
        token_exact_indexes,
        token_delete_indexes,
        compact_delete_index,
        alias_indexes,
        primary_compact_delete_index,
    )


def load_ath_pathways(path: Path) -> tuple[dict[str, str], dict[str, str]]:
    """Load ath pathway names and a mapXXXX -> athXXXX conversion table."""

    ath_pathways: dict[str, str] = {}
    map_to_ath: dict[str, str] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            pathway_id, raw_name = line.rstrip("\n").split("\t", 1)
            pathway_name = raw_name.split(" - ", 1)[0]
            ath_pathways[pathway_id] = pathway_name
            map_to_ath[f"map{pathway_id[3:]}"] = pathway_id
    return ath_pathways, map_to_ath


def load_map_pathways(path: Path) -> dict[str, str]:
    """Load KEGG reference pathway names keyed by map IDs."""

    map_pathways: dict[str, str] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            pathway_id, pathway_name = line.rstrip("\n").split("\t", 1)
            map_pathways[pathway_id] = pathway_name
    return map_pathways


def load_pathway_categories(path: Path) -> dict[str, tuple[str, str, str]]:
    """Parse KEGG pathway hierarchy into group/category/name triples."""

    categories: dict[str, tuple[str, str, str]] = {}
    top_level = ""
    sub_level = ""
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.rstrip("\n")
            if line.startswith("A"):
                top_level = line[1:].strip()
                continue
            if line.startswith("B"):
                sub_level = line[1:].strip()
                continue
            match = PATHWAY_LINE_RE.match(line)
            if match:
                pathway_code, pathway_name = match.groups()
                categories[f"map{pathway_code}"] = (top_level, sub_level, pathway_name)
    return categories


def load_kegg_pathway_links(
    path: Path,
    map_to_ath: dict[str, str],
) -> tuple[dict[str, list[tuple[str, str]]], dict[str, int]]:
    """Load KEGG compound -> pathway links and precompute pathway sizes."""

    links: defaultdict[str, list[tuple[str, str]]] = defaultdict(list)
    map_pathway_compound_counts: Counter[str] = Counter()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            compound_ref, pathway_ref = line.rstrip("\n").split("\t", 1)
            kegg_compound_id = compound_ref.replace("cpd:", "")
            map_pathway_id = pathway_ref.replace("path:", "")
            links[kegg_compound_id].append((map_pathway_id, map_to_ath.get(map_pathway_id, "")))
            map_pathway_compound_counts[map_pathway_id] += 1
    return dict(links), dict(map_pathway_compound_counts)


def load_ath_gene_counts(path: Path) -> dict[str, int]:
    """Count how many ath genes are linked to each ath pathway."""

    genes_by_pathway: defaultdict[str, set[str]] = defaultdict(set)
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            gene_ref, pathway_ref = line.rstrip("\n").split("\t", 1)
            pathway_id = pathway_ref.replace("path:", "")
            genes_by_pathway[pathway_id].add(gene_ref.replace("ath:", ""))
    return {pathway_id: len(genes) for pathway_id, genes in genes_by_pathway.items()}


def parse_links_field(text: str) -> dict[str, set[str]]:
    """Parse PMN/PlantCyc mixed link fields into typed identifier buckets."""

    parsed = {
        "chebi_ids": set(),
        "pubchem_cids": set(),
        "kegg_ids": set(),
        "hmdb_ids": set(),
    }
    for item in split_multi_value(text):
        if ":" not in item:
            continue
        prefix, value = item.split(":", 1)
        prefix = prefix.strip().upper()
        value = value.strip()
        if not value:
            continue
        if prefix == "CHEBI":
            parsed["chebi_ids"].add(normalize_chebi_id(value))
        elif prefix == "PUBCHEM" and PUBCHEM_CID_RE.fullmatch(value):
            parsed["pubchem_cids"].add(value)
        elif prefix == "HMDB":
            parsed["hmdb_ids"].add(value if value.upper().startswith("HMDB") else f"HMDB:{value}")
        elif prefix in {"LIGAND-CPD", "KEGG", "CPD"}:
            if KEGG_ID_RE.fullmatch(value):
                parsed["kegg_ids"].add(value)
    return parsed


def load_plantcyc_compounds(
    files: list[tuple[str, Path]],
) -> tuple[
    dict[str, PlantCycCompound],
    dict[str, defaultdict[str, set[str]]],
    set[str],
]:
    """Load AraCyc/PlantCyc compound exports and build cross-ID indexes."""

    records: dict[str, PlantCycCompound] = {}
    all_pubchem_cids: set[str] = set()
    for source_db, path in files:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            for row in reader:
                record_id = f"{source_db}:{row['Compound_id']}"
                record = records.get(record_id)
                if record is None:
                    record = PlantCycCompound(
                        record_id=record_id,
                        source_db=source_db,
                        compound_id=row["Compound_id"],
                        common_name=row["Compound_common_name"] or row["Compound_id"],
                        formula=row["Chemical_formula"] or "",
                        formula_key=formula_key(row["Chemical_formula"] or ""),
                        smiles=row["Smiles"] or "",
                    )
                    records[record_id] = record
                record.synonyms.update(split_multi_value(row["Compound_synonyms"]))
                if row["Pathway"]:
                    record.pathways.add(row["Pathway"])
                parsed_links = parse_links_field(row["Links"] or "")
                record.chebi_ids.update(parsed_links["chebi_ids"])
                record.pubchem_cids.update(parsed_links["pubchem_cids"])
                record.kegg_ids.update(parsed_links["kegg_ids"])
                record.hmdb_ids.update(parsed_links["hmdb_ids"])
                all_pubchem_cids.update(parsed_links["pubchem_cids"])

    indexes = {
        "by_chebi": defaultdict(set),
        "by_pubchem": defaultdict(set),
        "by_kegg": defaultdict(set),
        "by_name": defaultdict(set),
    }
    for record_id, record in records.items():
        for chebi_id in record.chebi_ids:
            indexes["by_chebi"][chebi_id].add(record_id)
        for cid in record.pubchem_cids:
            indexes["by_pubchem"][cid].add(record_id)
        for kegg_id in record.kegg_ids:
            indexes["by_kegg"][kegg_id].add(record_id)
        names = [record.common_name, *sorted(record.synonyms)]
        for name in names:
            normalized = normalize_name(name)
            if normalized:
                indexes["by_name"][normalized].add(record_id)
    return records, indexes, all_pubchem_cids


def load_plantcyc_pathway_stats(files: list[tuple[str, Path]]) -> dict[str, dict[str, dict[str, object]]]:
    """Collect lightweight pathway support summaries from PMN exports."""

    stats = {"AraCyc": {}, "PlantCyc": {}}
    for source_db, path in files:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            for row in reader:
                pathway_name = row["Pathway-name"] or ""
                normalized = normalize_name(pathway_name)
                if not normalized:
                    continue
                entry = stats[source_db].setdefault(
                    normalized,
                    {"names": set(), "pathway_ids": set(), "gene_ids": set()},
                )
                entry["names"].add(pathway_name)
                if row["Pathway-id"]:
                    entry["pathway_ids"].add(row["Pathway-id"])
                if row["Gene-id"]:
                    entry["gene_ids"].add(row["Gene-id"])
    return stats


def parse_sdf_records_from_zip(path: Path):
    """Stream SDF records from a zipped LMSD archive without full extraction."""

    with zipfile.ZipFile(path) as archive:
        member = archive.namelist()[0]
        with archive.open(member) as handle:
            fields: dict[str, str] = {}
            current_field = None
            current_lines: list[str] = []
            for raw in handle:
                line = raw.decode("utf-8", "replace").rstrip("\n")
                if line == "$$$$":
                    if current_field is not None:
                        fields[current_field] = "\n".join(current_lines).strip()
                    if fields:
                        yield fields
                    fields = {}
                    current_field = None
                    current_lines = []
                    continue
                if line.startswith("> <") and line.endswith(">"):
                    if current_field is not None:
                        fields[current_field] = "\n".join(current_lines).strip()
                    current_field = line[3:-1]
                    current_lines = []
                    continue
                if current_field is not None:
                    if line == "":
                        fields[current_field] = "\n".join(current_lines).strip()
                        current_field = None
                        current_lines = []
                    else:
                        current_lines.append(line)
            if current_field is not None:
                fields[current_field] = "\n".join(current_lines).strip()
            if fields:
                yield fields


def load_lipidmaps_records(
    path: Path,
) -> tuple[
    dict[str, LipidMapsRecord],
    dict[str, defaultdict[str, set[str]]],
    set[str],
]:
    """Load the subset of LIPID MAPS fields needed for matching/support."""

    records: dict[str, LipidMapsRecord] = {}
    all_pubchem_cids: set[str] = set()
    for entry in parse_sdf_records_from_zip(path):
        lm_id = entry.get("LM_ID", "").strip()
        if not lm_id:
            continue
        record = LipidMapsRecord(
            lm_id=lm_id,
            common_name=entry.get("COMMON_NAME", "").strip(),
            systematic_name=entry.get("SYSTEMATIC_NAME", "").strip(),
            synonyms=set(split_multi_value(entry.get("SYNONYMS", ""))),
            inchi_key=entry.get("INCHI_KEY", "").strip(),
            smiles=entry.get("SMILES", "").strip(),
            formula=entry.get("FORMULA", "").strip(),
            formula_key=formula_key(entry.get("FORMULA", "")),
            pubchem_cids={value for value in split_multi_value(entry.get("PUBCHEM_CID", "")) if PUBCHEM_CID_RE.fullmatch(value)},
            kegg_ids={value for value in split_multi_value(entry.get("KEGG_ID", "")) if KEGG_ID_RE.fullmatch(value)},
            hmdb_ids={value for value in split_multi_value(entry.get("HMDB_ID", "")) if value},
            chebi_ids={normalize_chebi_id(value) for value in split_multi_value(entry.get("CHEBI_ID", "")) if normalize_chebi_id(value)},
            category=entry.get("CATEGORY", "").strip(),
            main_class=entry.get("MAIN_CLASS", "").strip(),
            sub_class=entry.get("SUB_CLASS", "").strip(),
        )
        records[lm_id] = record
        all_pubchem_cids.update(record.pubchem_cids)

    indexes = {
        "by_chebi": defaultdict(set),
        "by_pubchem": defaultdict(set),
        "by_kegg": defaultdict(set),
        "by_inchi_key": defaultdict(set),
        "by_name": defaultdict(set),
    }
    for lm_id, record in records.items():
        for chebi_id in record.chebi_ids:
            indexes["by_chebi"][chebi_id].add(lm_id)
        for cid in record.pubchem_cids:
            indexes["by_pubchem"][cid].add(lm_id)
        for kegg_id in record.kegg_ids:
            indexes["by_kegg"][kegg_id].add(lm_id)
        if record.inchi_key:
            indexes["by_inchi_key"][record.inchi_key].add(lm_id)
        names = [record.common_name, record.systematic_name, *sorted(record.synonyms)]
        for name in names:
            normalized = normalize_name(name)
            if normalized:
                indexes["by_name"][normalized].add(lm_id)
    return records, indexes, all_pubchem_cids


def load_pubchem_synonyms(path: Path, target_cids: set[str]) -> tuple[dict[str, list[str]], dict[str, int]]:
    """Load only PubChem synonyms that are relevant to observed target CIDs."""

    synonyms: defaultdict[str, list[str]] = defaultdict(list)
    seen: defaultdict[str, set[str]] = defaultdict(set)
    stats = {"target_cids": len(target_cids), "matched_lines": 0, "kept_synonyms": 0}
    if not target_cids:
        return {}, stats
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            cid, synonym = line.rstrip("\n").split("\t", 1)
            if cid not in target_cids:
                continue
            stats["matched_lines"] += 1
            synonym = synonym.strip()
            if not keep_pubchem_synonym(synonym):
                continue
            normalized = normalize_name(synonym)
            if not normalized or normalized in seen[cid]:
                continue
            if len(synonyms[cid]) >= MAX_PUBCHEM_SYNONYMS_PER_CID:
                continue
            synonyms[cid].append(synonym)
            seen[cid].add(normalized)
            stats["kept_synonyms"] += 1
    return dict(synonyms), stats


def load_reactome_pathways(path: Path) -> dict[str, list[tuple[str, str]]]:
    """Index Reactome pathways by normalized name for explanation support."""

    index: defaultdict[str, list[tuple[str, str]]] = defaultdict(list)
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            pathway_id, pathway_name, species = line.rstrip("\n").split("\t", 2)
            normalized = normalize_name(pathway_name)
            if normalized:
                index[normalized].append((pathway_id, species))
    return dict(index)


def gather_external_match_methods(
    compound: ChEBICompound,
    structure: StructureInfo | None,
    xrefs: XrefInfo,
    base_aliases: list[AliasRecord],
    plantcyc_records: dict[str, PlantCycCompound],
    plantcyc_indexes: dict[str, defaultdict[str, set[str]]],
    lipidmaps_records: dict[str, LipidMapsRecord],
    lipidmaps_indexes: dict[str, defaultdict[str, set[str]]],
) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    """Collect how this compound touched external resources.

    The return value is not a final mapping. It is only a provenance summary,
    for example "matched PlantCyc by PubChem CID" or "matched LIPID MAPS by
    InChIKey". Later scoring code uses these method labels as evidence tiers.
    """

    alias_exacts = {alias.exact for alias in base_aliases if alias.exact}
    matched_plant: defaultdict[str, set[str]] = defaultdict(set)
    matched_lipid: defaultdict[str, set[str]] = defaultdict(set)

    if compound.chebi_accession in plantcyc_indexes["by_chebi"]:
        for record_id in plantcyc_indexes["by_chebi"][compound.chebi_accession]:
            matched_plant[record_id].add("chebi")
    if compound.chebi_accession in lipidmaps_indexes["by_chebi"]:
        for record_id in lipidmaps_indexes["by_chebi"][compound.chebi_accession]:
            matched_lipid[record_id].add("chebi")

    for cid in xrefs.pubchem_cids:
        for record_id in plantcyc_indexes["by_pubchem"].get(cid, set()):
            matched_plant[record_id].add("pubchem")
        for record_id in lipidmaps_indexes["by_pubchem"].get(cid, set()):
            matched_lipid[record_id].add("pubchem")

    for kegg_id in xrefs.kegg_ids:
        for record_id in plantcyc_indexes["by_kegg"].get(kegg_id, set()):
            matched_plant[record_id].add("kegg")
        for record_id in lipidmaps_indexes["by_kegg"].get(kegg_id, set()):
            matched_lipid[record_id].add("kegg")

    if structure and structure.standard_inchi_key:
        for record_id in lipidmaps_indexes["by_inchi_key"].get(structure.standard_inchi_key, set()):
            matched_lipid[record_id].add("inchi_key")

    for alias_exact in alias_exacts:
        plant_ids = plantcyc_indexes["by_name"].get(alias_exact)
        if plant_ids and len(plant_ids) <= EXTERNAL_NAME_MATCH_MAX_RECORDS:
            for record_id in plant_ids:
                matched_plant[record_id].add("name")
        lipid_ids = lipidmaps_indexes["by_name"].get(alias_exact)
        if lipid_ids and len(lipid_ids) <= EXTERNAL_NAME_MATCH_MAX_RECORDS:
            for record_id in lipid_ids:
                matched_lipid[record_id].add("name")

    return dict(matched_plant), dict(matched_lipid)


def build_compound_context(
    compound: ChEBICompound,
    structure: StructureInfo | None,
    xrefs: XrefInfo,
    base_aliases: list[AliasRecord],
    plantcyc_records: dict[str, PlantCycCompound],
    plantcyc_indexes: dict[str, defaultdict[str, set[str]]],
    lipidmaps_records: dict[str, LipidMapsRecord],
    lipidmaps_indexes: dict[str, defaultdict[str, set[str]]],
    pubchem_synonyms: dict[str, list[str]],
) -> CompoundContext:
    """Merge all alias/support information available for one ChEBI compound."""

    matched_plantcyc_methods, matched_lipidmaps_methods = gather_external_match_methods(
        compound=compound,
        structure=structure,
        xrefs=xrefs,
        base_aliases=base_aliases,
        plantcyc_records=plantcyc_records,
        plantcyc_indexes=plantcyc_indexes,
        lipidmaps_records=lipidmaps_records,
        lipidmaps_indexes=lipidmaps_indexes,
    )

    alias_bucket: dict[str, AliasRecord] = {}
    for alias in base_aliases:
        record_alias(alias_bucket, raw_name=alias.raw_name, source_type=alias.source_type, language_code=alias.language_code)

    pubchem_cids = set(xrefs.pubchem_cids)
    aracyc_pathways: defaultdict[str, set[str]] = defaultdict(set)
    plantcyc_pathways: defaultdict[str, set[str]] = defaultdict(set)

    for record_id in matched_plantcyc_methods:
        record = plantcyc_records[record_id]
        source_prefix = "aracyc" if record.source_db == "AraCyc" else "plantcyc"
        # Once a PlantCyc/AraCyc record is plausibly connected to the current
        # compound, its names become extra aliases and its pathways become
        # downstream support signals.
        record_alias(alias_bucket, raw_name=record.common_name, source_type=f"{source_prefix}_common_name")
        for synonym in list(sorted(record.synonyms))[:MAX_EXTRA_SYNONYMS_PER_RECORD]:
            record_alias(alias_bucket, raw_name=synonym, source_type=f"{source_prefix}_synonym")
        pubchem_cids.update(record.pubchem_cids)
        target = aracyc_pathways if record.source_db == "AraCyc" else plantcyc_pathways
        for pathway_name in record.pathways:
            normalized = normalize_name(pathway_name)
            if normalized:
                target[normalized].add(pathway_name)

    for record_id in matched_lipidmaps_methods:
        record = lipidmaps_records[record_id]
        if record.common_name:
            record_alias(alias_bucket, raw_name=record.common_name, source_type="lipidmaps_common_name")
        if record.systematic_name:
            record_alias(alias_bucket, raw_name=record.systematic_name, source_type="lipidmaps_systematic_name")
        for synonym in list(sorted(record.synonyms))[:MAX_EXTRA_SYNONYMS_PER_RECORD]:
            record_alias(alias_bucket, raw_name=synonym, source_type="lipidmaps_synonym")
        pubchem_cids.update(record.pubchem_cids)

    for cid in sorted(pubchem_cids):
        # PubChem is used only as a synonym expansion source here. It is not
        # treated as a canonical naming authority by itself.
        for synonym in pubchem_synonyms.get(cid, [])[:MAX_PUBCHEM_SYNONYMS_PER_CID]:
            record_alias(alias_bucket, raw_name=synonym, source_type="pubchem_synonym")

    all_aliases = list(alias_bucket.values())
    return CompoundContext(
        all_aliases=all_aliases,
        matched_plantcyc_methods=matched_plantcyc_methods,
        matched_lipidmaps_methods=matched_lipidmaps_methods,
        pubchem_cids=pubchem_cids,
        aracyc_pathways=dict(aracyc_pathways),
        plantcyc_pathways=dict(plantcyc_pathways),
    )


def reason_summary(candidate: CandidateMapping) -> str:
    """Collapse the most relevant evidence strings into a short explanation."""

    return "; ".join(candidate.reasons[:4])


def lookup_token_edit_candidates(
    tokens: tuple[str, ...],
    token_exact_index: defaultdict[tuple[str, ...], set[str]],
    token_delete_index: defaultdict[tuple[str, ...], defaultdict[int, set[str]]],
) -> set[str]:
    """Recall token-level fuzzy candidates from deletion signatures."""

    if len(tokens) < 2:
        return set()
    hit_ids: set[str] = set()
    for signature in deletion_signatures(tokens):
        hit_ids.update(token_exact_index.get(signature, set()))
        hit_ids.update(token_delete_index.get(signature, {}).get(len(tokens), set()))
    hit_ids.update(token_delete_index.get(tokens, {}).get(len(tokens) + 1, set()))
    return hit_ids


def lookup_compact_fuzzy_candidates(
    compact_name: str,
    compact_delete_index: defaultdict[str, set[str]],
) -> set[str]:
    """Recall candidate KEGG IDs for a compact name with <= N character edits."""

    if len(compact_name) < MIN_CHAR_FUZZY_COMPACT_LEN:
        return set()
    hit_ids: set[str] = set()
    for key in {compact_name, *generate_deletes(compact_name, MAX_CHAR_FUZZY_EDITS)}:
        hit_ids.update(compact_delete_index.get(key, set()))
    return hit_ids


def formula_sets_compatible(left: set[str], right: set[str]) -> bool:
    """Return True when formula evidence does not contradict a fuzzy match.

    Rules:
    - if both sides have formula evidence, they must share at least one formula
    - if one side is missing formula evidence, we keep the candidate alive
      because the missing formula may simply reflect incomplete coverage
    """

    if left and right:
        return bool(left & right)
    return True


def build_candidate_mappings(
    compound: ChEBICompound,
    structure: StructureInfo | None,
    xrefs: XrefInfo,
    context: CompoundContext,
    plantcyc_records: dict[str, PlantCycCompound],
    lipidmaps_records: dict[str, LipidMapsRecord],
    kegg_compounds: dict[str, KeggCompound],
    kegg_standard_indexes: dict[str, defaultdict[str, set[str]]],
    kegg_alias_indexes: dict[str, defaultdict[str, set[str]]],
    kegg_primary_compact_delete_index: defaultdict[str, set[str]],
    name_formula_index: dict[str, set[str]],
) -> list[CandidateMapping]:
    """Build and score all KEGG candidates for one ChEBI compound.

    Matching order is intentional:
    1. direct curated cross-references
    2. external structured support (PlantCyc, AraCyc, LIPID MAPS)
    3. exact match to a KEGG standard name
    4. exact match to a KEGG alias, then normalize back to the standard name
    5. typo-correction fallback against KEGG standard names only, guarded by
       molecular formula compatibility
    """

    candidates: dict[str, CandidateMapping] = {}

    def candidate_for(kegg_compound_id: str) -> CandidateMapping:
        candidate = candidates.get(kegg_compound_id)
        if candidate is None:
            kegg = kegg_compounds[kegg_compound_id]
            candidate = CandidateMapping(
                kegg_compound_id=kegg_compound_id,
                kegg_primary_name=kegg.primary_name,
            )
            candidates[kegg_compound_id] = candidate
        return candidate

    for kegg_id in sorted(xrefs.kegg_ids):
        if kegg_id in kegg_compounds:
            candidate_for(kegg_id).add_evidence(
                score=0.995,
                method="chebi_kegg_xref",
                reason=f"Direct ChEBI KEGG cross-reference points to {kegg_id}",
                direct_kegg_xref=True,
                external_source="ChEBI",
            )

    for record_id, methods in context.matched_plantcyc_methods.items():
        record = plantcyc_records[record_id]
        if not record.kegg_ids:
            continue
        for kegg_id in record.kegg_ids:
            if kegg_id not in kegg_compounds:
                continue
            score = 0.89
            match_label = "name"
            if "chebi" in methods or "kegg" in methods:
                score = 0.97 if record.source_db == "AraCyc" else 0.96
                match_label = "crossref"
            elif "pubchem" in methods:
                score = 0.95
                match_label = "pubchem"
            elif "name" in methods:
                score = 0.89
            if structure and structure.formula_key and record.formula_key and structure.formula_key == record.formula_key:
                score = min(score + 0.01, 0.98)
            candidate_for(kegg_id).add_evidence(
                score=score,
                method=f"{record.source_db.lower()}_{match_label}",
                reason=f"{record.source_db} links {compound.chebi_accession} to {kegg_id} via {','.join(sorted(methods))}",
                external_source=record.source_db,
            )

    for record_id, methods in context.matched_lipidmaps_methods.items():
        record = lipidmaps_records[record_id]
        if not record.kegg_ids:
            continue
        for kegg_id in record.kegg_ids:
            if kegg_id not in kegg_compounds:
                continue
            score = 0.88
            method_label = "name"
            has_structure_evidence = False
            if "inchi_key" in methods:
                score = 0.97
                method_label = "structure"
                has_structure_evidence = True
            elif "chebi" in methods or "kegg" in methods:
                score = 0.96
                method_label = "crossref"
            elif "pubchem" in methods:
                score = 0.94
                method_label = "pubchem"
            if structure and structure.formula_key and record.formula_key and structure.formula_key == record.formula_key:
                score = min(score + 0.01, 0.98)
            candidate_for(kegg_id).add_evidence(
                score=score,
                method=f"lipidmaps_{method_label}",
                reason=f"LIPID MAPS links {compound.chebi_accession} to {kegg_id} via {','.join(sorted(methods))}",
                external_source="LIPID MAPS",
                has_structure_evidence=has_structure_evidence,
            )

    for alias in context.all_aliases:
        matched_by_name = False
        for variant_name in VARIANT_ORDER:
            variant_value = getattr(alias, variant_name)
            if not variant_value:
                continue
            standard_hit_ids = kegg_standard_indexes[variant_name].get(variant_value)
            if standard_hit_ids:
                # Preferred case: the incoming name already resolves directly to
                # a KEGG canonical label.
                matched_by_name = True
                for kegg_id in standard_hit_ids:
                    score = VARIANT_BASE_SCORES[variant_name] + ALIAS_SOURCE_WEIGHTS.get(alias.source_type, 0.0) + 0.03
                    candidate_for(kegg_id).add_evidence(
                        score=min(score, 0.98),
                        method=f"name_match_{variant_name}",
                        reason=f"{variant_name} standard-name match via {alias.source_type}: {alias.raw_name}",
                        alias=alias.raw_name,
                        source_type=alias.source_type,
                        variant=variant_name,
                        primary_name_match=True,
                        used_pubchem_synonym=alias.source_type == "pubchem_synonym",
                    )
                break
            alias_hit_ids = kegg_alias_indexes[variant_name].get(variant_value)
            if not alias_hit_ids:
                continue
            # Second-best exact case: a KEGG synonym matches, so we keep the
            # hit but explicitly describe the correction back to the standard
            # KEGG primary name.
            matched_by_name = True
            for kegg_id in alias_hit_ids:
                kegg = kegg_compounds[kegg_id]
                score = VARIANT_BASE_SCORES[variant_name] + ALIAS_SOURCE_WEIGHTS.get(alias.source_type, 0.0)
                candidate_for(kegg_id).add_evidence(
                    score=min(score, 0.98),
                    method=f"name_match_{variant_name}",
                    reason=f"{variant_name} alias-table match via {alias.source_type}: {alias.raw_name}; corrected to standard name {kegg.primary_name}",
                    alias=alias.raw_name,
                    source_type=alias.source_type,
                    variant=variant_name,
                    primary_name_match=False,
                    used_pubchem_synonym=alias.source_type == "pubchem_synonym",
                )
            break
        if matched_by_name:
            continue
        compact_value = alias.compact
        input_formula_keys = {structure.formula_key} if structure and structure.formula_key else set(name_formula_index.get(compact_value, set()))
        hit_ids = lookup_compact_fuzzy_candidates(compact_value, kegg_primary_compact_delete_index)
        if not hit_ids:
            continue
        for kegg_id in hit_ids:
            kegg = kegg_compounds[kegg_id]
            candidate_formula_keys = set(name_formula_index.get(kegg.primary_compact, set()))
            # Final fallback: allow small spelling differences only when formula
            # evidence does not contradict the correction target.
            formula_validated = char_edit_distance_at_most_two(compact_value, kegg.primary_compact) and formula_sets_compatible(
                input_formula_keys,
                candidate_formula_keys,
            )
            formula_shared = bool(input_formula_keys and candidate_formula_keys and input_formula_keys & candidate_formula_keys)
            if not formula_validated:
                continue
            score = CHAR_EDIT2_BASE_SCORE + ALIAS_SOURCE_WEIGHTS.get(alias.source_type, 0.0)
            score += 0.02
            if formula_shared:
                score += 0.02
            candidate_for(kegg_id).add_evidence(
                score=min(score, 0.94),
                method="name_match_compact_formula_edit2",
                reason=f"compact edit<=2 with formula validation via {alias.source_type}: {alias.raw_name}; corrected to standard name {kegg.primary_name}",
                alias=alias.raw_name,
                source_type=alias.source_type,
                variant="compact_formula_edit2",
                primary_name_match=True,
                used_pubchem_synonym=alias.source_type == "pubchem_synonym",
            )

    ranked = sorted(
        candidates.values(),
        key=lambda item: (
            int(item.direct_kegg_xref),
            int(item.has_structure_evidence),
            item.best_score,
            int(item.primary_name_match),
            item.evidence_count,
            item.kegg_compound_id,
        ),
        reverse=True,
    )
    for candidate in ranked:
        bonus = min(0.05, 0.01 * max(candidate.evidence_count - 1, 0))
        if candidate.direct_kegg_xref:
            bonus += 0.01
        if candidate.has_structure_evidence:
            bonus += 0.01
        candidate.final_score = min(candidate.best_score + bonus, 0.999)
    return ranked


def select_candidates(ranked: list[CandidateMapping]) -> list[CandidateMapping]:
    """Choose final mappings after evidence aggregation and ranking.

    The function is intentionally conservative: weak or closely competing name
    matches are kept out of the final mapping table and later marked as
    ambiguous/unmapped by the caller.
    """

    if not ranked:
        return []
    direct = [candidate for candidate in ranked if candidate.direct_kegg_xref]
    if direct:
        return direct
    top = ranked[0]
    if top.final_score < 0.88:
        return []
    if len(ranked) == 1:
        return [top]
    second = ranked[1]
    if top.final_score >= 0.96:
        return [top]
    if top.final_score - second.final_score >= 0.03:
        return [top]
    return []


def mapping_method_label(mapping: CandidateMapping) -> str:
    """Convert internal evidence flags into a compact output label."""

    if mapping.direct_kegg_xref:
        return "chebi_kegg_xref"
    if mapping.has_structure_evidence and "LIPID MAPS" in mapping.external_sources:
        return "lipidmaps_structure_crossref"
    if mapping.external_sources & {"AraCyc", "PlantCyc"}:
        return "plantcyc_crossref"
    return f"name_match_{mapping.best_variant or 'unknown'}"


def mapping_confidence_label(score: float) -> str:
    """Bucket mapping scores into user-facing confidence bands."""

    if score >= 0.95:
        return "high"
    if score >= 0.88:
        return "medium"
    return "low"


def serialize_support_context(context: CompoundContext) -> StoredSupportContext:
    """Persist only the support context needed after mapping selection."""

    return StoredSupportContext(
        aracyc_pathways={key: tuple(sorted(values)) for key, values in context.aracyc_pathways.items()},
        plantcyc_pathways={key: tuple(sorted(values)) for key, values in context.plantcyc_pathways.items()},
        matched_plantcyc_records=len(context.matched_plantcyc_methods),
        matched_lipidmaps_records=len(context.matched_lipidmaps_methods),
    )


def process_compounds(
    *,
    compounds: dict[str, ChEBICompound],
    base_aliases: dict[str, list[AliasRecord]],
    xrefs: dict[str, XrefInfo],
    structures: dict[str, StructureInfo],
    plantcyc_records: dict[str, PlantCycCompound],
    plantcyc_indexes: dict[str, defaultdict[str, set[str]]],
    lipidmaps_records: dict[str, LipidMapsRecord],
    lipidmaps_indexes: dict[str, defaultdict[str, set[str]]],
    pubchem_synonyms: dict[str, list[str]],
    kegg_compounds: dict[str, KeggCompound],
    kegg_standard_indexes: dict[str, defaultdict[str, set[str]]],
    kegg_alias_indexes: dict[str, defaultdict[str, set[str]]],
    kegg_primary_compact_delete_index: defaultdict[str, set[str]],
    name_formula_index: dict[str, set[str]],
    alias_output_path: Path,
    mapping_summary_path: Path,
    mapping_selected_path: Path,
) -> tuple[dict[str, list[CandidateMapping]], dict[str, StoredSupportContext], Counter[str], int]:
    """Run alias expansion, KEGG mapping, and audit-table writing per compound."""

    selected_by_compound: dict[str, list[CandidateMapping]] = {}
    support_contexts: dict[str, StoredSupportContext] = {}
    mapping_status: Counter[str] = Counter()
    alias_rows = 0

    with alias_output_path.open("w", newline="", encoding="utf-8") as alias_handle, mapping_summary_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as mapping_handle, mapping_selected_path.open("w", newline="", encoding="utf-8") as selected_handle:
        alias_writer = csv.DictWriter(
            alias_handle,
            fieldnames=[
                "compound_id",
                "chebi_accession",
                "chebi_name",
                "alias",
                "source_type",
                "normalized_name",
                "compact_name",
                "singular_name",
                "stereo_stripped_name",
            ],
            delimiter="\t",
        )
        alias_writer.writeheader()

        mapping_writer = csv.DictWriter(
            mapping_handle,
            fieldnames=[
                "compound_id",
                "chebi_accession",
                "chebi_name",
                "selection_status",
                "selected_kegg_compound_ids",
                "selected_kegg_names",
                "selected_scores",
                "top_candidate_kegg_compound_id",
                "top_candidate_name",
                "top_candidate_score",
                "candidate_count",
                "alias_count",
                "matched_pubchem_cids",
                "mapping_reason",
            ],
            delimiter="\t",
        )
        mapping_writer.writeheader()

        selected_writer = csv.DictWriter(
            selected_handle,
            fieldnames=[
                "compound_id",
                "chebi_accession",
                "chebi_name",
                "kegg_compound_id",
                "kegg_primary_name",
                "mapping_score",
                "mapping_confidence_level",
                "mapping_method",
                "best_alias",
                "best_alias_source",
                "best_variant",
                "evidence_count",
                "external_sources",
                "has_structure_evidence",
                "used_pubchem_synonym",
                "mapping_reason",
            ],
            delimiter="\t",
        )
        selected_writer.writeheader()

        for compound_id in sorted(compounds, key=int):
            compound = compounds[compound_id]
            structure = structures.get(compound_id)
            compound_xrefs = xrefs.get(compound_id, XrefInfo())
            context = build_compound_context(
                compound=compound,
                structure=structure,
                xrefs=compound_xrefs,
                base_aliases=base_aliases.get(compound_id, []),
                plantcyc_records=plantcyc_records,
                plantcyc_indexes=plantcyc_indexes,
                lipidmaps_records=lipidmaps_records,
                lipidmaps_indexes=lipidmaps_indexes,
                pubchem_synonyms=pubchem_synonyms,
            )

            for alias in sorted(context.all_aliases, key=lambda item: (item.source_type, item.raw_name)):
                alias_writer.writerow(
                    {
                        "compound_id": compound_id,
                        "chebi_accession": compound.chebi_accession,
                        "chebi_name": compound.name,
                        "alias": alias.raw_name,
                        "source_type": alias.source_type,
                        "normalized_name": alias.exact,
                        "compact_name": alias.compact,
                        "singular_name": alias.singular,
                        "stereo_stripped_name": alias.stereo_stripped,
                    }
                )
                alias_rows += 1

            ranked = build_candidate_mappings(
                compound=compound,
                structure=structure,
                xrefs=compound_xrefs,
                context=context,
                plantcyc_records=plantcyc_records,
                lipidmaps_records=lipidmaps_records,
                kegg_compounds=kegg_compounds,
                kegg_standard_indexes=kegg_standard_indexes,
                kegg_alias_indexes=kegg_alias_indexes,
                kegg_primary_compact_delete_index=kegg_primary_compact_delete_index,
                name_formula_index=name_formula_index,
            )
            selected = select_candidates(ranked)
            selected_by_compound[compound_id] = selected
            if selected:
                selection_status = "selected"
                support_contexts[compound_id] = serialize_support_context(context)
            elif ranked:
                # We saw plausible candidates, but the ranking gap or absolute
                # score was not strong enough to trust a correction.
                selection_status = "ambiguous"
            else:
                selection_status = "unmapped"
            mapping_status[selection_status] += 1

            top = ranked[0] if ranked else None
            mapping_writer.writerow(
                {
                    "compound_id": compound_id,
                    "chebi_accession": compound.chebi_accession,
                    "chebi_name": compound.name,
                    "selection_status": selection_status,
                    "selected_kegg_compound_ids": ";".join(mapping.kegg_compound_id for mapping in selected),
                    "selected_kegg_names": ";".join(mapping.kegg_primary_name for mapping in selected),
                    "selected_scores": ";".join(f"{mapping.final_score:.3f}" for mapping in selected),
                    "top_candidate_kegg_compound_id": top.kegg_compound_id if top else "",
                    "top_candidate_name": top.kegg_primary_name if top else "",
                    "top_candidate_score": f"{top.final_score:.3f}" if top else "",
                    "candidate_count": len(ranked),
                    "alias_count": len(context.all_aliases),
                    "matched_pubchem_cids": ";".join(sorted(context.pubchem_cids)),
                    "mapping_reason": reason_summary(top) if top else "",
                }
            )

            for mapping in selected:
                selected_writer.writerow(
                    {
                        "compound_id": compound_id,
                        "chebi_accession": compound.chebi_accession,
                        "chebi_name": compound.name,
                        "kegg_compound_id": mapping.kegg_compound_id,
                        "kegg_primary_name": mapping.kegg_primary_name,
                        "mapping_score": f"{mapping.final_score:.3f}",
                        "mapping_confidence_level": mapping_confidence_label(mapping.final_score),
                        "mapping_method": mapping_method_label(mapping),
                        "best_alias": mapping.best_alias,
                        "best_alias_source": mapping.best_source_type,
                        "best_variant": mapping.best_variant,
                        "evidence_count": mapping.evidence_count,
                        "external_sources": ";".join(sorted(mapping.external_sources)),
                        "has_structure_evidence": str(mapping.has_structure_evidence).lower(),
                        "used_pubchem_synonym": str(mapping.used_pubchem_synonym).lower(),
                        "mapping_reason": reason_summary(mapping),
                    }
                )

    return selected_by_compound, support_contexts, mapping_status, alias_rows


def pathway_support_bonus(
    pathway_name: str,
    support_context: StoredSupportContext | None,
    plantcyc_pathway_stats: dict[str, dict[str, dict[str, object]]],
) -> tuple[float, str, str, int]:
    """Translate PlantCyc/AraCyc pathway presence into a scoring bonus."""

    if support_context is None:
        return 0.0, "", "", 0
    normalized = normalize_name(pathway_name)
    if normalized in support_context.aracyc_pathways:
        names = support_context.aracyc_pathways[normalized]
        gene_count = len(plantcyc_pathway_stats["AraCyc"].get(normalized, {}).get("gene_ids", set()))
        return 0.06, "AraCyc", ";".join(names[:3]), gene_count
    if normalized in support_context.plantcyc_pathways:
        names = support_context.plantcyc_pathways[normalized]
        gene_count = len(plantcyc_pathway_stats["PlantCyc"].get(normalized, {}).get("gene_ids", set()))
        return 0.04, "PlantCyc", ";".join(names[:3]), gene_count
    return 0.0, "", "", 0


def top_feature_labels(contributions: dict[str, float]) -> tuple[str, str]:
    """Extract short positive/negative feature summaries for reporting."""

    positives = [f"{key}={value:.3f}" for key, value in sorted(contributions.items(), key=lambda item: item[1], reverse=True) if value > 0][:3]
    negatives = [f"{key}={value:.3f}" for key, value in sorted(contributions.items(), key=lambda item: item[1]) if value < 0][:2]
    return ";".join(positives), ";".join(negatives)


def build_pathway_explanation(
    *,
    mapping: CandidateMapping,
    pathway_name: str,
    target_type: str,
    gene_count: int,
    plant_source: str,
    plant_examples: str,
    score: float,
    confidence_level: str,
    contributions: dict[str, float],
) -> str:
    """Compose a human-readable explanation for one pathway recommendation."""

    positives, negatives = top_feature_labels(contributions)
    parts = [
        f"{confidence_level} confidence pathway for {mapping.kegg_compound_id} -> {pathway_name}",
        f"score {score:.3f}",
        f"mapping reason: {reason_summary(mapping)}",
    ]
    if target_type == "map_fallback":
        parts.append("used map fallback because no ath-specific pathway was available")
    if gene_count:
        parts.append(f"ath gene support: {gene_count}")
    if plant_source:
        parts.append(f"{plant_source} support: {plant_examples}")
    if positives:
        parts.append(f"top positive features: {positives}")
    if negatives:
        parts.append(f"top negative features: {negatives}")
    return "; ".join(parts)


def write_pathway_table(
    *,
    path: Path,
    compounds: dict[str, ChEBICompound],
    selected_by_compound: dict[str, list[CandidateMapping]],
    support_contexts: dict[str, StoredSupportContext],
    kegg_to_pathways: dict[str, list[tuple[str, str]]],
    map_pathways: dict[str, str],
    ath_pathways: dict[str, str],
    pathway_categories: dict[str, tuple[str, str, str]],
    map_pathway_compound_counts: dict[str, int],
    ath_gene_counts: dict[str, int],
    reactome_pathways: dict[str, list[tuple[str, str]]],
    plantcyc_pathway_stats: dict[str, dict[str, dict[str, object]]],
) -> Counter[str]:
    """Expand selected mappings to pathways and write the ranked output table."""

    pathway_status: Counter[str] = Counter()
    max_map_count = max(map_pathway_compound_counts.values(), default=1)
    max_gene_count = max(ath_gene_counts.values(), default=1)

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "compound_id",
                "chebi_accession",
                "chebi_name",
                "pathway_rank",
                "score",
                "confidence_level",
                "evidence_type",
                "mapping_confidence",
                "support_kegg_compound_ids",
                "support_kegg_names",
                "pathway_target_id",
                "pathway_target_type",
                "ath_pathway_id",
                "map_pathway_id",
                "pathway_name",
                "pathway_group",
                "pathway_category",
                "map_pathway_compound_count",
                "ath_gene_count",
                "plantcyc_support_source",
                "plantcyc_support_examples",
                "reactome_matches",
                "top_positive_features",
                "top_negative_features",
                "feature_contributions_json",
                "reason",
            ],
            delimiter="\t",
        )
        writer.writeheader()

        for compound_id in sorted(compounds, key=int):
            selected = selected_by_compound.get(compound_id, [])
            if not selected:
                pathway_status["without_mapping"] += 1
                continue

            compound = compounds[compound_id]
            support_context = support_contexts.get(compound_id)
            aggregated: dict[str, dict[str, object]] = {}

            for mapping in selected:
                for map_pathway_id, ath_pathway_id in kegg_to_pathways.get(mapping.kegg_compound_id, []):
                    target_id = ath_pathway_id or map_pathway_id
                    target_type = "ath" if ath_pathway_id else "map_fallback"
                    pathway_name = ath_pathways.get(ath_pathway_id, map_pathways.get(map_pathway_id, map_pathway_id))
                    pathway_group, pathway_category, _ = pathway_categories.get(map_pathway_id, ("", "", ""))
                    map_compound_count = map_pathway_compound_counts.get(map_pathway_id, 0)
                    specificity = 1 - math.log1p(map_compound_count) / math.log1p(max_map_count)
                    gene_count = ath_gene_counts.get(ath_pathway_id, 0) if ath_pathway_id else 0
                    gene_ratio = math.log1p(gene_count) / math.log1p(max_gene_count) if gene_count else 0.0
                    plant_bonus, plant_source, plant_examples, _ = pathway_support_bonus(
                        pathway_name=pathway_name,
                        support_context=support_context,
                        plantcyc_pathway_stats=plantcyc_pathway_stats,
                    )
                    generic_penalty = -0.06 if map_pathway_id in GENERIC_KEGG_MAPS or pathway_category == "Global and overview maps" else 0.0
                    # These contributions are deliberately explicit so the TSV
                    # can later be turned into a SHAP-like explanation view or
                    # replaced by a trained model with similar semantics.
                    contributions = {
                        "mapping_confidence": round(0.45 * mapping.final_score, 3),
                        "pathway_specificity": round(0.15 * specificity, 3),
                        "ath_exists": 0.12 if ath_pathway_id else 0.0,
                        "ath_gene_support": round(0.10 * gene_ratio, 3),
                        "direct_kegg_xref": 0.05 if mapping.direct_kegg_xref else 0.0,
                        "structure_support": 0.05 if mapping.has_structure_evidence else 0.0,
                        "plantcyc_support": round(plant_bonus, 3),
                        "generic_pathway_penalty": generic_penalty,
                        "map_fallback_penalty": -0.10 if not ath_pathway_id else 0.0,
                    }
                    score = max(0.0, min(sum(contributions.values()), 0.999))
                    confidence_level = "high" if score >= 0.70 and ath_pathway_id and gene_count > 0 else "medium" if score >= 0.40 else "low"
                    reactome_matches = reactome_pathways.get(normalize_name(pathway_name), [])
                    reactome_text = ";".join(f"{pathway_id}|{species}" for pathway_id, species in reactome_matches[:3])
                    top_positive, top_negative = top_feature_labels(contributions)
                    explanation = build_pathway_explanation(
                        mapping=mapping,
                        pathway_name=pathway_name,
                        target_type=target_type,
                        gene_count=gene_count,
                        plant_source=plant_source,
                        plant_examples=plant_examples,
                        score=score,
                        confidence_level=confidence_level,
                        contributions=contributions,
                    )
                    entry = aggregated.setdefault(
                        target_id,
                        {
                            "score": -1.0,
                            "confidence_level": confidence_level,
                            "mapping_confidence": 0.0,
                            "support_kegg_ids": set(),
                            "support_kegg_names": set(),
                            "pathway_target_id": target_id,
                            "pathway_target_type": target_type,
                            "ath_pathway_id": ath_pathway_id,
                            "map_pathway_id": map_pathway_id,
                            "pathway_name": pathway_name,
                            "pathway_group": pathway_group,
                            "pathway_category": pathway_category,
                            "map_pathway_compound_count": map_compound_count,
                            "ath_gene_count": gene_count,
                            "plantcyc_support_source": plant_source,
                            "plantcyc_support_examples": plant_examples,
                            "reactome_matches": reactome_text,
                            "top_positive_features": top_positive,
                            "top_negative_features": top_negative,
                            "feature_contributions_json": json.dumps(contributions, ensure_ascii=False, sort_keys=True),
                            "reason": explanation,
                        },
                    )
                    if score > float(entry["score"]):
                        entry["score"] = score
                        entry["confidence_level"] = confidence_level
                        entry["mapping_confidence"] = mapping.final_score
                        entry["ath_gene_count"] = gene_count
                        entry["plantcyc_support_source"] = plant_source
                        entry["plantcyc_support_examples"] = plant_examples
                        entry["reactome_matches"] = reactome_text
                        entry["top_positive_features"] = top_positive
                        entry["top_negative_features"] = top_negative
                        entry["feature_contributions_json"] = json.dumps(contributions, ensure_ascii=False, sort_keys=True)
                        entry["reason"] = explanation
                    entry["support_kegg_ids"].add(mapping.kegg_compound_id)
                    entry["support_kegg_names"].add(mapping.kegg_primary_name)

            if not aggregated:
                pathway_status["mapped_without_pathway"] += 1
                continue

            ranked = sorted(
                aggregated.values(),
                key=lambda item: (
                    float(item["score"]),
                    int(item["ath_gene_count"]),
                    -int(item["map_pathway_compound_count"]),
                    item["pathway_target_id"],
                ),
                reverse=True,
            )
            pathway_status["with_pathway"] += 1
            pathway_status["pathway_rows"] += len(ranked)

            for rank, entry in enumerate(ranked, start=1):
                writer.writerow(
                    {
                        "compound_id": compound_id,
                        "chebi_accession": compound.chebi_accession,
                        "chebi_name": compound.name,
                        "pathway_rank": rank,
                        "score": f"{float(entry['score']):.3f}",
                        "confidence_level": entry["confidence_level"],
                        "evidence_type": "direct_compound_pathway",
                        "mapping_confidence": f"{float(entry['mapping_confidence']):.3f}",
                        "support_kegg_compound_ids": ";".join(sorted(entry["support_kegg_ids"])),
                        "support_kegg_names": ";".join(sorted(entry["support_kegg_names"])),
                        "pathway_target_id": entry["pathway_target_id"],
                        "pathway_target_type": entry["pathway_target_type"],
                        "ath_pathway_id": entry["ath_pathway_id"],
                        "map_pathway_id": entry["map_pathway_id"],
                        "pathway_name": entry["pathway_name"],
                        "pathway_group": entry["pathway_group"],
                        "pathway_category": entry["pathway_category"],
                        "map_pathway_compound_count": entry["map_pathway_compound_count"],
                        "ath_gene_count": entry["ath_gene_count"],
                        "plantcyc_support_source": entry["plantcyc_support_source"],
                        "plantcyc_support_examples": entry["plantcyc_support_examples"],
                        "reactome_matches": entry["reactome_matches"],
                        "top_positive_features": entry["top_positive_features"],
                        "top_negative_features": entry["top_negative_features"],
                        "feature_contributions_json": entry["feature_contributions_json"],
                        "reason": entry["reason"],
                    }
                )
    return pathway_status


def build_summary(
    *,
    compounds: dict[str, ChEBICompound],
    mapping_status: Counter[str],
    pathway_status: Counter[str],
    alias_rows: int,
    comments_profile: dict[str, int],
    plantcyc_records: dict[str, PlantCycCompound],
    lipidmaps_records: dict[str, LipidMapsRecord],
    pubchem_stats: dict[str, int],
) -> dict[str, object]:
    """Build the JSON summary used for quick run auditing."""

    return {
        "compounds_total": len(compounds),
        "standardized_alias_rows": alias_rows,
        "mapping_status": dict(mapping_status),
        "pathway_status": dict(pathway_status),
        "comments_profile": comments_profile,
        "database_usage": {
            "ChEBI": "used for primary compound table, aliases, structures and direct KEGG xrefs",
            "KEGG": "used for compound IDs, compound-pathway links, ath-specific pathways and gene counts",
            "PubChem": pubchem_stats,
            "LIPID MAPS": {"records_loaded": len(lipidmaps_records)},
            "AraCyc_PlantCyc": {"records_loaded": len(plantcyc_records)},
            "Reactome": "exact-name pathway context only",
            "GO": "downloaded for future ontology integration; not used in direct compound-pathway mapping",
            "KNApSAcK": "downloaded public archive was a metabolomics spectra package, so it was not used as a synonym table in v2",
        },
        "notes": [
            "comments.tsv was profiled but still not used as an alias source because its CompoundName rows are mostly editorial notes.",
            "v2 prefers direct KEGG xrefs and external cross-references over pure name matching.",
            "When an ath-specific pathway is absent, v2 keeps the map pathway as a lower-confidence fallback instead of dropping it.",
            "The feature_contributions_json field is a rule-based explanation layer that prepares the pipeline for later ML/SHAP integration.",
        ],
    }


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for local batch execution."""

    parser = argparse.ArgumentParser(description="Process local ChEBI compounds into KEGG/AraCyc/PlantCyc-supported pathway rankings.")
    parser.add_argument("--workdir", default=".", help="Workspace containing compounds.tsv, comments.tsv, refs/, and outputs/.")
    return parser.parse_args()


def main() -> None:
    """Wire together all loaders, matchers, scorers, and output writers."""

    args = parse_args()
    workdir = Path(args.workdir).resolve()
    refs = workdir / "refs"
    outputs = workdir / "outputs"
    outputs.mkdir(parents=True, exist_ok=True)

    required_paths = [
        workdir / "compounds.tsv",
        workdir / "comments.tsv",
        refs / "names.tsv.gz",
        refs / "database_accession.tsv.gz",
        refs / "chemical_data.tsv.gz",
        refs / "structures.tsv.gz",
        refs / "kegg_compound_list.tsv",
        refs / "kegg_compound_pathway.tsv",
        refs / "kegg_pathway_ath.tsv",
        refs / "kegg_pathway_map.tsv",
        refs / "kegg_pathway_hierarchy.txt",
        refs / "kegg_ath_gene_pathway.tsv",
        refs / "pubchem_cid_synonym_filtered.gz",
        refs / "lmsd_extended.sdf.zip",
        refs / "aracyc_compounds.20230103",
        refs / "aracyc_pathways.20230103",
        refs / "plantcyc_compounds.20220103",
        refs / "plantcyc_pathways.20230103",
        refs / "ReactomePathways.txt",
    ]
    for required_path in required_paths:
        ensure_exists(required_path)

    compounds = load_compounds(workdir / "compounds.tsv")
    comments_profile = load_comments_profile(workdir / "comments.tsv")
    base_aliases = load_base_aliases(compounds, refs / "names.tsv.gz")
    xrefs = load_xrefs(refs / "database_accession.tsv.gz")
    formulas = load_formula_info(refs / "chemical_data.tsv.gz")
    structures = load_structures(refs / "structures.tsv.gz", formulas)
    (
        kegg_compounds,
        _kegg_all_indexes,
        _kegg_token_exact_indexes,
        _kegg_token_delete_indexes,
        _kegg_all_compact_delete_index,
        kegg_alias_indexes,
        kegg_primary_compact_delete_index,
    ) = load_kegg_compounds(refs / "kegg_compound_list.tsv")
    # The loader returns several specialized indexes. For the main matching
    # path we only need alias indexes and a standard-name typo-correction index.
    kegg_standard_indexes = {
        "exact": defaultdict(set),
        "compact": defaultdict(set),
        "singular": defaultdict(set),
        "stereo_stripped": defaultdict(set),
    }
    for kegg_id, kegg in kegg_compounds.items():
        kegg_standard_indexes["exact"][kegg.primary_exact].add(kegg_id)
        kegg_standard_indexes["compact"][kegg.primary_compact].add(kegg_id)
        kegg_standard_indexes["singular"][kegg.primary_singular].add(kegg_id)
        kegg_standard_indexes["stereo_stripped"][kegg.primary_stereo_stripped].add(kegg_id)
    ath_pathways, map_to_ath = load_ath_pathways(refs / "kegg_pathway_ath.tsv")
    map_pathways = load_map_pathways(refs / "kegg_pathway_map.tsv")
    pathway_categories = load_pathway_categories(refs / "kegg_pathway_hierarchy.txt")
    kegg_to_pathways, map_pathway_compound_counts = load_kegg_pathway_links(refs / "kegg_compound_pathway.tsv", map_to_ath)
    ath_gene_counts = load_ath_gene_counts(refs / "kegg_ath_gene_pathway.tsv")
    plantcyc_records, plantcyc_indexes, plantcyc_pubchem_cids = load_plantcyc_compounds(
        [
            ("AraCyc", refs / "aracyc_compounds.20230103"),
            ("PlantCyc", refs / "plantcyc_compounds.20220103"),
        ]
    )
    plantcyc_pathway_stats = load_plantcyc_pathway_stats(
        [
            ("AraCyc", refs / "aracyc_pathways.20230103"),
            ("PlantCyc", refs / "plantcyc_pathways.20230103"),
        ]
    )
    lipidmaps_records, lipidmaps_indexes, lipidmaps_pubchem_cids = load_lipidmaps_records(refs / "lmsd_extended.sdf.zip")
    name_formula_index = build_name_formula_index(base_aliases, structures, plantcyc_records, lipidmaps_records)
    target_pubchem_cids = set(plantcyc_pubchem_cids) | set(lipidmaps_pubchem_cids)
    for info in xrefs.values():
        target_pubchem_cids.update(info.pubchem_cids)
    pubchem_synonyms, pubchem_stats = load_pubchem_synonyms(refs / "pubchem_cid_synonym_filtered.gz", target_pubchem_cids)
    reactome_pathways = load_reactome_pathways(refs / "ReactomePathways.txt")

    selected_by_compound, support_contexts, mapping_status, alias_rows = process_compounds(
        compounds=compounds,
        base_aliases=base_aliases,
        xrefs=xrefs,
        structures=structures,
        plantcyc_records=plantcyc_records,
        plantcyc_indexes=plantcyc_indexes,
        lipidmaps_records=lipidmaps_records,
        lipidmaps_indexes=lipidmaps_indexes,
        pubchem_synonyms=pubchem_synonyms,
        kegg_compounds=kegg_compounds,
        kegg_standard_indexes=kegg_standard_indexes,
        kegg_alias_indexes=kegg_alias_indexes,
        kegg_primary_compact_delete_index=kegg_primary_compact_delete_index,
        name_formula_index=name_formula_index,
        alias_output_path=outputs / "chebi_aliases_standardized_v2.tsv",
        mapping_summary_path=outputs / "chebi_kegg_mapping_v2.tsv",
        mapping_selected_path=outputs / "chebi_kegg_selected_v2.tsv",
    )

    pathway_status = write_pathway_table(
        path=outputs / "chebi_pathways_ranked_v2.tsv",
        compounds=compounds,
        selected_by_compound=selected_by_compound,
        support_contexts=support_contexts,
        kegg_to_pathways=kegg_to_pathways,
        map_pathways=map_pathways,
        ath_pathways=ath_pathways,
        pathway_categories=pathway_categories,
        map_pathway_compound_counts=map_pathway_compound_counts,
        ath_gene_counts=ath_gene_counts,
        reactome_pathways=reactome_pathways,
        plantcyc_pathway_stats=plantcyc_pathway_stats,
    )

    summary = build_summary(
        compounds=compounds,
        mapping_status=mapping_status,
        pathway_status=pathway_status,
        alias_rows=alias_rows,
        comments_profile=comments_profile,
        plantcyc_records=plantcyc_records,
        lipidmaps_records=lipidmaps_records,
        pubchem_stats=pubchem_stats,
    )
    with (outputs / "processing_summary_v2.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
