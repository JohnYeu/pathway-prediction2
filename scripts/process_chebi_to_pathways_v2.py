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
import subprocess
import sys
import unicodedata
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

try:
    from rdkit import Chem
except Exception:  # pragma: no cover - optional dependency
    Chem = None


csv.field_size_limit(min(sys.maxsize, 2**31 - 1))

TAG_RE = re.compile(r"<[^>]+>")
WS_RE = re.compile(r"\s+")
NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")
KEGG_ID_RE = re.compile(r"^C\d{5}$")
PUBCHEM_CID_RE = re.compile(r"^\d+$")
CAS_RE = re.compile(r"^\d{2,7}-\d{2}-\d$")
PATHWAY_LINE_RE = re.compile(r"^C\s+(\d{5})\s+(.+)$")
GO_ID_RE = re.compile(r"^GO:\d{7}$")
AGI_LOCUS_RE = re.compile(r"\bAT[1-5CM]G\d{5}\b", re.IGNORECASE)


def latest_versioned_ref(refs: Path, prefix: str, fallback_name: str) -> Path:
    """Return the newest versioned reference file available in refs/."""

    fallback = refs / fallback_name
    candidates = [
        path
        for path in refs.glob(f"{prefix}.*")
        if path.is_file() and path.name[len(prefix) + 1 :].isdigit()
    ]
    if not candidates:
        return fallback
    return max(candidates, key=lambda path: path.name)

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
PLANT_BRIDGE_BASE_SCORES = {
    ("plant_direct_kegg_xref", "AraCyc"): 0.955,
    ("plant_direct_kegg_xref", "PlantCyc"): 0.935,
    ("plant_via_chebi", "AraCyc"): 0.930,
    ("plant_via_chebi", "PlantCyc"): 0.910,
    ("plant_inchikey_exact", "AraCyc"): 0.920,
    ("plant_inchikey_exact", "PlantCyc"): 0.900,
    ("plant_via_pubchem", "AraCyc"): 0.900,
    ("plant_via_pubchem", "PlantCyc"): 0.880,
    ("plant_smiles_exact", "AraCyc"): 0.890,
    ("plant_smiles_exact", "PlantCyc"): 0.870,
}
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
PLANT_REACTOME_HIERARCHY_URL = "https://plantreactome.gramene.org/ContentService/data/eventsHierarchy/3702"
PLANT_REACTOME_QUERY_URL = "https://plantreactome.gramene.org/ContentService/data/query/{st_id}"
PLANT_REACTOME_PARTICIPANTS_URL = "https://plantreactome.gramene.org/ContentService/data/participants/{st_id}"
GO_TAIR_GAF_URLS = (
    "https://current.geneontology.org/annotations/tair.gaf.gz",
    "https://www.arabidopsis.org/download_files/GO_and_PO_Annotations/"
    "Gene_Ontology_Annotations/gene_association.tair.gz",
)
PLANT_CONTEXT_TAGS = (
    "primary_metabolism",
    "secondary_metabolism",
    "hormone_related",
    "stress_related",
    "development_related",
    "transport_related",
)
PRIMARY_BRITE_KEYWORDS = (
    "carbohydrate metabolism",
    "amino acid metabolism",
    "lipid metabolism",
    "nucleotide metabolism",
    "energy metabolism",
    "metabolism of cofactors and vitamins",
    "glycan biosynthesis",
)
SECONDARY_BRITE_KEYWORDS = (
    "biosynthesis of other secondary metabolites",
    "xenobiotics biodegradation",
)
SECONDARY_NAME_KEYWORDS = (
    "flavonoid",
    "phenylpropanoid",
    "terpenoid",
    "alkaloid",
    "glucosinolate",
    "anthocyanin",
)
HORMONE_KEYWORDS = (
    "auxin",
    "cytokinin",
    "gibberellin",
    "abscisic",
    "brassinosteroid",
    "ethylene",
    "jasmon",
    "salicyl",
)
STRESS_KEYWORDS = (
    "glutathione",
    "detox",
    "pathogen",
    "defense",
    "defence",
    "stress",
    "oxidative",
)
DEVELOPMENT_KEYWORDS = (
    "development",
    "morphogenesis",
    "meristem",
    "embryo",
    "seed",
)
TRANSPORT_KEYWORDS = (
    "transport",
    "transporter",
    "membrane transport",
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
    plant_bridge_methods: set[str] = field(default_factory=set)
    plant_bridge_sources: set[str] = field(default_factory=set)
    arabidopsis_supported: bool = False

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
        plant_bridge_method: str = "",
        plant_bridge_source: str = "",
        arabidopsis_supported: bool = False,
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
        self.arabidopsis_supported = self.arabidopsis_supported or arabidopsis_supported
        if external_source:
            self.external_sources.add(external_source)
        if plant_bridge_method:
            self.plant_bridge_methods.add(plant_bridge_method)
        if plant_bridge_source:
            self.plant_bridge_sources.add(plant_bridge_source)


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
class PlantToKeggBridge:
    """Aggregated PMN -> KEGG bridge evidence for one compound and KEGG target."""

    compound_id: str
    chebi_accession: str
    canonical_name: str
    plant_db: str
    plant_compound_id: str
    kegg_cid: str
    bridge_method: str
    bridge_score: float
    bridge_confidence: str
    bridge_record_ids: tuple[str, ...]
    supporting_ids: tuple[str, ...]
    arabidopsis_supported: bool
    has_structure_evidence: bool
    bridge_reason: str


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
    """Compatibility wrapper for the step-1-owned compound loader."""

    from pathway_pipeline.step1_standardize_names import load_compounds as _impl

    return _impl(path)


def load_comments_profile(path: Path) -> dict[str, int]:
    """Compatibility wrapper for the step-1-owned comments profiler."""

    from pathway_pipeline.step1_standardize_names import load_comments_profile as _impl

    return _impl(path)


def load_base_aliases(
    compounds: dict[str, ChEBICompound],
    names_path: Path,
) -> dict[str, list[AliasRecord]]:
    """Compatibility wrapper for the step-1-owned base alias loader."""

    from pathway_pipeline.step1_standardize_names import load_base_aliases as _impl

    return _impl(compounds, names_path)


def load_xrefs(path: Path) -> dict[str, XrefInfo]:
    """Compatibility wrapper for the step-1-owned xref loader."""

    from pathway_pipeline.step1_standardize_names import load_xrefs as _impl

    return _impl(path)


def load_formula_info(path: Path) -> dict[str, tuple[str, str]]:
    """Compatibility wrapper for the step-1-owned formula loader."""

    from pathway_pipeline.step1_standardize_names import load_formula_info as _impl

    return _impl(path)


def load_structures(path: Path, formulas: dict[str, tuple[str, str]]) -> dict[str, StructureInfo]:
    """Compatibility wrapper for the step-1-owned structure loader."""

    from pathway_pipeline.step1_standardize_names import load_structures as _impl

    return _impl(path, formulas)


def build_name_formula_index(
    base_aliases: dict[str, list[AliasRecord]],
    structures: dict[str, StructureInfo],
    plantcyc_records: dict[str, PlantCycCompound],
    lipidmaps_records: dict[str, LipidMapsRecord],
) -> dict[str, set[str]]:
    """Compatibility wrapper for the step-1-owned formula validation index."""

    from pathway_pipeline.step1_standardize_names import build_name_formula_index as _impl

    return _impl(base_aliases, structures, plantcyc_records, lipidmaps_records)


def normalize_inchi_key(value: str) -> str:
    """Return a normalized full InChIKey string."""

    return (value or "").strip().upper()


def rdkit_smiles_available() -> bool:
    """Return whether RDKit-backed SMILES canonicalization is available."""

    return Chem is not None


def canonicalize_smiles(value: str) -> str:
    """Return a canonical isomeric SMILES string when RDKit is available."""

    text = (value or "").strip()
    if not text or Chem is None:
        return ""
    try:
        molecule = Chem.MolFromSmiles(text)
        if molecule is None:
            return ""
        return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)
    except Exception:
        return ""


def build_kegg_structure_indexes(
    structures: dict[str, StructureInfo],
    xrefs: dict[str, XrefInfo],
    lipidmaps_records: dict[str, LipidMapsRecord],
) -> dict[str, dict[str, dict[str, set[str]]]]:
    """Compatibility wrapper for the step-2-owned KEGG structure bridge logic."""

    from pathway_pipeline.name_mapping_support import build_kegg_structure_indexes as _impl

    return _impl(structures, xrefs, lipidmaps_records)


def build_plantcyc_compound_index(
    plantcyc_records: dict[str, PlantCycCompound],
) -> list[dict[str, object]]:
    """Compatibility wrapper for the step-2-owned PMN audit table builder."""

    from pathway_pipeline.name_mapping_support import build_plantcyc_compound_index as _impl

    return _impl(plantcyc_records)


def build_kegg_structure_index(
    structures: dict[str, StructureInfo],
    xrefs: dict[str, XrefInfo],
    lipidmaps_records: dict[str, LipidMapsRecord],
) -> list[dict[str, object]]:
    """Compatibility wrapper for the step-2-owned KEGG structure table writer."""

    from pathway_pipeline.name_mapping_support import build_kegg_structure_index as _impl

    return _impl(structures, xrefs, lipidmaps_records)


def _build_chebi_to_kegg_bridge_index(
    compounds: dict[str, ChEBICompound],
    xrefs: dict[str, XrefInfo],
) -> dict[str, set[str]]:
    """Index KEGG IDs by ChEBI accession for PMN bridge lookups."""

    table: defaultdict[str, set[str]] = defaultdict(set)
    for compound_id, compound in compounds.items():
        for kegg_id in xrefs.get(compound_id, XrefInfo()).kegg_ids:
            if KEGG_ID_RE.fullmatch(kegg_id):
                table[compound.chebi_accession].add(kegg_id)
    return {chebi_id: set(kegg_ids) for chebi_id, kegg_ids in table.items()}


def _build_pubchem_to_kegg_bridge_index(
    xrefs: dict[str, XrefInfo],
    lipidmaps_records: dict[str, LipidMapsRecord],
) -> dict[str, set[str]]:
    """Index KEGG IDs by PubChem CID for PMN bridge lookups."""

    table: defaultdict[str, set[str]] = defaultdict(set)
    for info in xrefs.values():
        valid_kegg_ids = {kegg_id for kegg_id in info.kegg_ids if KEGG_ID_RE.fullmatch(kegg_id)}
        if not valid_kegg_ids:
            continue
        for cid in info.pubchem_cids:
            table[cid].update(valid_kegg_ids)
    for record in lipidmaps_records.values():
        valid_kegg_ids = {kegg_id for kegg_id in record.kegg_ids if KEGG_ID_RE.fullmatch(kegg_id)}
        if not valid_kegg_ids:
            continue
        for cid in record.pubchem_cids:
            table[cid].update(valid_kegg_ids)
    return {cid: set(kegg_ids) for cid, kegg_ids in table.items()}


def bridge_confidence_label(score: float) -> str:
    """Compatibility wrapper for the step-2-owned PMN bridge label helper."""

    from pathway_pipeline.name_mapping_support import bridge_confidence_label as _impl

    return _impl(score)


def build_plant_to_kegg_bridge(
    *,
    compound: ChEBICompound,
    context: CompoundContext,
    plantcyc_records: dict[str, PlantCycCompound],
    chebi_to_kegg_index: dict[str, set[str]],
    pubchem_to_kegg_index: dict[str, set[str]],
    kegg_structure_indexes: dict[str, dict[str, dict[str, set[str]]]],
) -> list[PlantToKeggBridge]:
    """Compatibility wrapper for the step-2-owned PMN -> KEGG bridge logic."""

    from pathway_pipeline.name_mapping_support import build_plant_to_kegg_bridge as _impl

    return _impl(
        compound=compound,
        context=context,
        plantcyc_records=plantcyc_records,
        chebi_to_kegg_index=chebi_to_kegg_index,
        pubchem_to_kegg_index=pubchem_to_kegg_index,
        kegg_structure_indexes=kegg_structure_indexes,
    )


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
    """Compatibility wrapper for the step-4-owned ath conversion loader."""

    from pathway_pipeline.pathway_annotation_support import load_ath_pathways as _impl

    return _impl(path)


def load_map_pathways(path: Path) -> dict[str, str]:
    """Compatibility wrapper for the step-5-owned KEGG map loader."""

    from pathway_pipeline.pathway_annotation_support import load_map_pathways as _impl

    return _impl(path)


def load_pathway_categories(path: Path) -> dict[str, tuple[str, str, str]]:
    """Compatibility wrapper for the step-5-owned BRITE parser."""

    from pathway_pipeline.pathway_annotation_support import load_pathway_categories as _impl

    return _impl(path)


def load_kegg_pathway_links(
    path: Path,
    map_to_ath: dict[str, str],
) -> tuple[dict[str, list[tuple[str, str]]], dict[str, int]]:
    """Compatibility wrapper for the step-3-owned KEGG pathway linker."""

    from pathway_pipeline.step3_link_compounds_to_pathways import load_kegg_pathway_links as _impl

    return _impl(path, map_to_ath)


def load_ath_gene_counts(path: Path) -> dict[str, int]:
    """Compatibility wrapper for the step-5-owned ath gene counter."""

    from pathway_pipeline.pathway_annotation_support import load_ath_gene_counts as _impl

    return _impl(path)


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
    """Compatibility wrapper for the step-1-owned PMN compound loader."""

    from pathway_pipeline.step1_standardize_names import load_plantcyc_compounds as _impl

    return _impl(files)


def load_plantcyc_pathway_stats(files: list[tuple[str, Path]]) -> dict[str, dict[str, dict[str, object]]]:
    """Compatibility wrapper for the step-5-owned PMN pathway stats loader."""

    from pathway_pipeline.pathway_annotation_support import load_plantcyc_pathway_stats as _impl

    return _impl(files)


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
    """Compatibility wrapper for the step-1-owned LIPID MAPS loader."""

    from pathway_pipeline.step1_standardize_names import load_lipidmaps_records as _impl

    return _impl(path)


def load_pubchem_synonyms(path: Path, target_cids: set[str]) -> tuple[dict[str, list[str]], dict[str, int]]:
    """Compatibility wrapper for the step-1-owned PubChem synonym loader."""

    from pathway_pipeline.step1_standardize_names import load_pubchem_synonyms as _impl

    return _impl(path, target_cids)


def load_reactome_pathways(path: Path) -> dict[str, list[tuple[str, str]]]:
    """Compatibility wrapper for the step-5-owned Reactome name index."""

    from pathway_pipeline.pathway_annotation_support import load_reactome_pathways as _impl

    return _impl(path)


def fetch_text_with_fallback(url: str) -> str:
    """Fetch UTF-8 text with urllib first, then curl as a fallback."""

    try:
        with urlopen(url, timeout=120) as response:  # noqa: S310 - explicit data-source fetch
            return response.read().decode("utf-8")
    except URLError:
        result = subprocess.run(
            ["curl", "-Lsf", url],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout


def fetch_json_with_fallback(url: str):
    """Fetch JSON data from a remote endpoint with curl fallback."""

    return json.loads(fetch_text_with_fallback(url))


def download_binary_with_fallback(url: str, destination: Path) -> None:
    """Download a binary asset to disk, using curl if urllib fails."""

    try:
        with urlopen(url, timeout=180) as response:  # noqa: S310 - explicit data-source fetch
            destination.write_bytes(response.read())
            return
    except URLError:
        result = subprocess.run(
            ["curl", "-Lsf", url, "-o", str(destination)],
            check=True,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:  # pragma: no cover - subprocess.check_call handles this
            raise RuntimeError(f"Failed to download {url} to {destination}")


def iso_mtime(path: Path) -> str:
    """Compatibility wrapper for the step-5-owned file timestamp helper."""

    from pathway_pipeline.pathway_annotation_support import iso_mtime as _impl

    return _impl(path)


def load_go_basic_terms(path: Path) -> tuple[dict[str, str], dict[str, str]]:
    """Parse GO term names/namespaces from go-basic.obo."""

    go_names: dict[str, str] = {}
    go_namespaces: dict[str, str] = {}
    current_id = ""
    current_name = ""
    current_namespace = ""

    def commit() -> None:
        if current_id and current_namespace == "biological_process":
            go_names[current_id] = current_name or current_id
            go_namespaces[current_id] = current_namespace

    with path.open(encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.rstrip("\n")
            if line == "[Term]":
                commit()
                current_id = ""
                current_name = ""
                current_namespace = ""
                continue
            if not line:
                continue
            if line.startswith("id: "):
                current_id = line[4:].strip()
            elif line.startswith("name: "):
                current_name = line[6:].strip()
            elif line.startswith("namespace: "):
                current_namespace = line[11:].strip()
        commit()
    return go_names, go_namespaces


def extract_agi_loci(*fields: str) -> set[str]:
    """Extract normalized Arabidopsis AGI loci from free-text fields."""

    matches: set[str] = set()
    for field in fields:
        if not field:
            continue
        for match in AGI_LOCUS_RE.findall(field):
            matches.add(match.upper())
    return matches


def ensure_go_annotations(path: Path) -> str:
    """Compatibility wrapper for the step-5-owned GO snapshot loader."""

    from pathway_pipeline.pathway_annotation_support import ensure_go_annotations as _impl

    return _impl(path)


def load_gene_to_go_bp(gaf_path: Path, go_basic_path: Path) -> tuple[dict[str, set[tuple[str, str]]], dict[str, str], str]:
    """Compatibility wrapper for the step-5-owned GO gene index loader."""

    from pathway_pipeline.pathway_annotation_support import load_gene_to_go_bp as _impl

    return _impl(gaf_path, go_basic_path)


def build_go_term_gene_sets(gene_to_go: dict[str, set[tuple[str, str]]]) -> tuple[dict[str, set[str]], dict[str, str]]:
    """Compatibility wrapper for the step-5-owned GO inversion helper."""

    from pathway_pipeline.pathway_annotation_support import build_go_term_gene_sets as _impl

    return _impl(gene_to_go)


@lru_cache(maxsize=None)
def log_choose(n: int, k: int) -> float:
    """Return log(n choose k)."""

    if k < 0 or k > n:
        return float("-inf")
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


def hypergeom_sf(population_size: int, success_population: int, draw_count: int, observed_successes: int) -> float:
    """Compute P(X >= observed_successes) for a hypergeometric distribution."""

    min_successes = max(0, draw_count - (population_size - success_population))
    max_successes = min(draw_count, success_population)
    if observed_successes <= min_successes:
        return 1.0
    if observed_successes > max_successes:
        return 0.0
    denominator = log_choose(population_size, draw_count)
    probabilities = []
    for value in range(observed_successes, max_successes + 1):
        log_prob = log_choose(success_population, value) + log_choose(population_size - success_population, draw_count - value) - denominator
        probabilities.append(math.exp(log_prob))
    return min(1.0, max(0.0, sum(probabilities)))


def benjamini_hochberg(p_values: list[float]) -> list[float]:
    """Adjust p-values with the Benjamini-Hochberg FDR procedure."""

    count = len(p_values)
    if count == 0:
        return []
    ranked = sorted(enumerate(p_values), key=lambda item: item[1])
    adjusted = [1.0] * count
    running = 1.0
    for rank, (index, p_value) in enumerate(reversed(ranked), start=1):
        true_rank = count - rank + 1
        candidate = min(1.0, (p_value * count) / true_rank)
        running = min(running, candidate)
        adjusted[index] = running
    return adjusted


def compute_pathway_go_enrichment(
    pathway_gene_ids: set[str],
    term_to_genes: dict[str, set[str]],
    go_names: dict[str, str],
    background_genes: set[str],
    top_k: int = 5,
) -> dict[str, object]:
    """Compatibility wrapper for the step-5-owned GO enrichment helper."""

    from pathway_pipeline.pathway_annotation_support import compute_pathway_go_enrichment as _impl

    return _impl(pathway_gene_ids, term_to_genes, go_names, background_genes, top_k=top_k)


def extract_identifier_strings(data) -> set[str]:
    """Recursively collect gene-like identifiers from Plant Reactome payloads."""

    identifiers: set[str] = set()
    if isinstance(data, dict):
        for key, value in data.items():
            if key == "identifier" and isinstance(value, str):
                identifiers.update(extract_agi_loci(value))
            else:
                identifiers.update(extract_identifier_strings(value))
    elif isinstance(data, list):
        for item in data:
            identifiers.update(extract_identifier_strings(item))
    return identifiers


def flatten_plant_reactome_hierarchy(nodes, top_level_category: str = "") -> dict[str, dict[str, str]]:
    """Flatten Plant Reactome hierarchy JSON into pathway metadata rows."""

    flattened: dict[str, dict[str, str]] = {}
    for node in nodes or []:
        pathway_id = node.get("stId", "")
        pathway_name = node.get("name") or node.get("displayName") or ""
        species = node.get("species") or node.get("speciesName") or ""
        node_type = node.get("type") or node.get("className") or ""
        current_top = pathway_name if node_type == "TopLevelPathway" else top_level_category
        if pathway_id and node_type in {"TopLevelPathway", "Pathway"}:
            flattened[pathway_id] = {
                "pathway_id": pathway_id,
                "pathway_name": pathway_name,
                "species": species,
                "top_level_category": current_top,
            }
        child_nodes = node.get("children") or []
        if child_nodes:
            flattened.update(flatten_plant_reactome_hierarchy(child_nodes, current_top))
    return flattened


def ensure_plant_reactome_refs(
    pathways_path: Path,
    gene_path: Path,
    version_path: Path,
) -> str:
    """Compatibility wrapper for the step-5-owned Plant Reactome snapshotter."""

    from pathway_pipeline.pathway_annotation_support import ensure_plant_reactome_refs as _impl

    return _impl(pathways_path, gene_path, version_path)


def load_plant_reactome_pathways(path: Path) -> dict[str, dict[str, str]]:
    """Compatibility wrapper for the step-5-owned Plant Reactome pathway loader."""

    from pathway_pipeline.pathway_annotation_support import load_plant_reactome_pathways as _impl

    return _impl(path)


def load_plant_reactome_gene_index(path: Path) -> dict[str, set[str]]:
    """Compatibility wrapper for the step-5-owned Plant Reactome gene loader."""

    from pathway_pipeline.pathway_annotation_support import load_plant_reactome_gene_index as _impl

    return _impl(path)


def token_sorensen_dice(left_text: str, right_text: str) -> float:
    """Compute Sørensen-Dice similarity on normalized token sets."""

    left_tokens = set(normalize_name(left_text).split())
    right_tokens = set(normalize_name(right_text).split())
    if not left_tokens or not right_tokens:
        return 0.0
    overlap = len(left_tokens & right_tokens)
    return (2.0 * overlap) / (len(left_tokens) + len(right_tokens))


def infer_plant_context_tags(
    *,
    brite_l1: str,
    brite_l2: str,
    brite_l3: str,
    pathway_name: str,
    has_aracyc: bool = False,
    has_plantcyc: bool = False,
    plant_reactome_texts: tuple[str, ...] = (),
    go_texts: tuple[str, ...] = (),
) -> tuple[str, ...]:
    """Compatibility wrapper for the step-5-owned plant tag inference."""

    from pathway_pipeline.pathway_annotation_support import infer_plant_context_tags as _impl

    return _impl(
        brite_l1=brite_l1,
        brite_l2=brite_l2,
        brite_l3=brite_l3,
        pathway_name=pathway_name,
        has_aracyc=has_aracyc,
        has_plantcyc=has_plantcyc,
        plant_reactome_texts=plant_reactome_texts,
        go_texts=go_texts,
    )


def plant_reactome_category_bonus(
    *,
    brite_l1: str,
    brite_l2: str,
    brite_l3: str,
    pathway_name: str,
    reactome_name: str,
    reactome_category: str,
    reactome_description: str,
) -> float:
    """Return a simple categorical bonus when KEGG and Plant Reactome agree."""

    left_tags = set(
        infer_plant_context_tags(
            brite_l1=brite_l1,
            brite_l2=brite_l2,
            brite_l3=brite_l3,
            pathway_name=pathway_name,
        )
    )
    right_tags = set(
        infer_plant_context_tags(
            brite_l1="",
            brite_l2=reactome_category,
            brite_l3="",
            pathway_name=reactome_name,
            plant_reactome_texts=(reactome_description,),
        )
    )
    return 1.0 if left_tags & right_tags else 0.0


def match_plant_reactome_context(
    *,
    pathway_name: str,
    gene_ids: set[str],
    brite_l1: str,
    brite_l2: str,
    brite_l3: str,
    plant_reactome_pathways: dict[str, dict[str, str]],
    plant_reactome_gene_sets: dict[str, set[str]],
    top_k: int = 3,
) -> list[dict[str, object]]:
    """Compatibility wrapper for the step-5-owned Plant Reactome matcher."""

    from pathway_pipeline.pathway_annotation_support import match_plant_reactome_context as _impl

    return _impl(
        pathway_name=pathway_name,
        gene_ids=gene_ids,
        brite_l1=brite_l1,
        brite_l2=brite_l2,
        brite_l3=brite_l3,
        plant_reactome_pathways=plant_reactome_pathways,
        plant_reactome_gene_sets=plant_reactome_gene_sets,
        top_k=top_k,
    )


def build_annotation_confidence(
    *,
    brite_l1: str,
    go_best_term: str,
    aracyc_evidence_score: float,
    reactome_matches: tuple[tuple[str, str], ...],
    plant_evidence_sources: tuple[str, ...],
    plant_reactome_alignment_confidence: str,
) -> str:
    """Compatibility wrapper for the step-5-owned annotation confidence rule."""

    from pathway_pipeline.pathway_annotation_support import build_annotation_confidence as _impl

    return _impl(
        brite_l1=brite_l1,
        go_best_term=go_best_term,
        aracyc_evidence_score=aracyc_evidence_score,
        reactome_matches=reactome_matches,
        plant_evidence_sources=plant_evidence_sources,
        plant_reactome_alignment_confidence=plant_reactome_alignment_confidence,
    )


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
    """Compatibility wrapper for the step-1-owned external-match summarizer."""

    from pathway_pipeline.step1_standardize_names import gather_external_match_methods as _impl

    return _impl(compound, structure, xrefs, base_aliases, plantcyc_records, plantcyc_indexes, lipidmaps_records, lipidmaps_indexes)


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
    """Compatibility wrapper for the step-1-owned compound-context builder."""

    from pathway_pipeline.step1_standardize_names import build_compound_context as _impl

    return _impl(
        compound,
        structure,
        xrefs,
        base_aliases,
        plantcyc_records,
        plantcyc_indexes,
        lipidmaps_records,
        lipidmaps_indexes,
        pubchem_synonyms,
    )


def reason_summary(candidate: CandidateMapping) -> str:
    """Compatibility wrapper for the step-2-owned explanation helper."""

    from pathway_pipeline.name_mapping_support import reason_summary as _impl

    return _impl(candidate)


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
    plant_bridge_rows: list[PlantToKeggBridge],
    lipidmaps_records: dict[str, LipidMapsRecord],
    kegg_compounds: dict[str, KeggCompound],
    kegg_standard_indexes: dict[str, defaultdict[str, set[str]]],
    kegg_alias_indexes: dict[str, defaultdict[str, set[str]]],
    kegg_primary_compact_delete_index: defaultdict[str, set[str]],
    name_formula_index: dict[str, set[str]],
    kegg_structure_indexes: dict[str, dict[str, set[str]]],
) -> list[CandidateMapping]:
    """Compatibility wrapper for the step-2-owned candidate builder."""

    from pathway_pipeline.name_mapping_support import build_candidate_mappings as _impl

    return _impl(
        compound=compound,
        structure=structure,
        xrefs=xrefs,
        context=context,
        plant_bridge_rows=plant_bridge_rows,
        lipidmaps_records=lipidmaps_records,
        kegg_compounds=kegg_compounds,
        kegg_standard_indexes=kegg_standard_indexes,
        kegg_alias_indexes=kegg_alias_indexes,
        kegg_primary_compact_delete_index=kegg_primary_compact_delete_index,
        name_formula_index=name_formula_index,
        kegg_structure_indexes=kegg_structure_indexes,
    )


def select_candidates(ranked: list[CandidateMapping]) -> list[CandidateMapping]:
    """Compatibility wrapper for the step-2-owned candidate selector."""

    from pathway_pipeline.name_mapping_support import select_candidates as _impl

    return _impl(ranked)


def mapping_method_label(mapping: CandidateMapping) -> str:
    """Compatibility wrapper for the step-2-owned method label helper."""

    from pathway_pipeline.name_mapping_support import mapping_method_label as _impl

    return _impl(mapping)


def mapping_confidence_label(score: float) -> str:
    """Compatibility wrapper for the step-2-owned confidence label helper."""

    from pathway_pipeline.name_mapping_support import mapping_confidence_label as _impl

    return _impl(score)


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
    kegg_structure_indexes: dict[str, dict[str, set[str]]],
    alias_output_path: Path,
    mapping_summary_path: Path,
    mapping_selected_path: Path,
) -> tuple[dict[str, list[CandidateMapping]], dict[str, StoredSupportContext], Counter[str], int]:
    """Run alias expansion, KEGG mapping, and audit-table writing per compound."""

    selected_by_compound: dict[str, list[CandidateMapping]] = {}
    support_contexts: dict[str, StoredSupportContext] = {}
    mapping_status: Counter[str] = Counter()
    alias_rows = 0
    chebi_to_kegg_index = _build_chebi_to_kegg_bridge_index(compounds, xrefs)
    pubchem_to_kegg_index = _build_pubchem_to_kegg_bridge_index(xrefs, lipidmaps_records)

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

            plant_bridge_rows = build_plant_to_kegg_bridge(
                compound=compound,
                context=context,
                plantcyc_records=plantcyc_records,
                chebi_to_kegg_index=chebi_to_kegg_index,
                pubchem_to_kegg_index=pubchem_to_kegg_index,
                kegg_structure_indexes=kegg_structure_indexes,
            )
            ranked = build_candidate_mappings(
                compound=compound,
                structure=structure,
                xrefs=compound_xrefs,
                context=context,
                plant_bridge_rows=plant_bridge_rows,
                lipidmaps_records=lipidmaps_records,
                kegg_compounds=kegg_compounds,
                kegg_standard_indexes=kegg_standard_indexes,
                kegg_alias_indexes=kegg_alias_indexes,
                kegg_primary_compact_delete_index=kegg_primary_compact_delete_index,
                name_formula_index=name_formula_index,
                kegg_structure_indexes=kegg_structure_indexes,
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
    aracyc_compounds_path = latest_versioned_ref(refs, "aracyc_compounds", "aracyc_compounds.20230103")
    aracyc_pathways_path = latest_versioned_ref(refs, "aracyc_pathways", "aracyc_pathways.20230103")
    plantcyc_compounds_path = latest_versioned_ref(refs, "plantcyc_compounds", "plantcyc_compounds.20220103")
    plantcyc_pathways_path = latest_versioned_ref(refs, "plantcyc_pathways", "plantcyc_pathways.20230103")

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
        aracyc_compounds_path,
        aracyc_pathways_path,
        plantcyc_compounds_path,
        plantcyc_pathways_path,
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
            ("AraCyc", aracyc_compounds_path),
            ("PlantCyc", plantcyc_compounds_path),
        ]
    )
    plantcyc_pathway_stats = load_plantcyc_pathway_stats(
        [
            ("AraCyc", aracyc_pathways_path),
            ("PlantCyc", plantcyc_pathways_path),
        ]
    )
    lipidmaps_records, lipidmaps_indexes, lipidmaps_pubchem_cids = load_lipidmaps_records(refs / "lmsd_extended.sdf.zip")
    name_formula_index = build_name_formula_index(base_aliases, structures, plantcyc_records, lipidmaps_records)
    kegg_structure_indexes = build_kegg_structure_indexes(structures, xrefs, lipidmaps_records)
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
        kegg_structure_indexes=kegg_structure_indexes,
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
