"""Step 6 (AraCyc-first, deprecated experimental path): ML supplemental recall.

This step supplements the AraCyc primary chain with ML-scored expanded
candidates. Feature extraction and model training are self-contained.

Candidate sources:
1. PlantCyc direct compound→pathway links (compounds without AraCyc match)
2. Structural similarity neighbors with known pathways

Training positives come from the AraCyc primary chain (score >= 0.50).
"""

from __future__ import annotations

import csv
import json
import re
import sys
import warnings
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

try:
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")
except Exception:
    pass

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from process_chebi_to_pathways_v2 import (
    PlantCycCompound,
    normalize_name,
)

from pathway_pipeline.context import PipelineContext


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ML_SCORE_THRESHOLD = 0.55
RANDOM_SEED = 42
PROGRESS_EVERY = 20000
MODEL_VERSION = "step6_aracyc_ml_recall_v1"
FEATURE_SET_VERSION = "chemistry_structure_pair_v1"

EC_RE = re.compile(r"(?:EC-)?\d+(?:\.[\d-]+){1,3}", re.I)

MODEL_EXCLUDED_FEATURE_NAMES = {
    "arabidopsis_supported",
    "has_aracyc",
    "has_ec",
    "has_formula",
    "has_hmdb",
    "has_kegg_xref",
    "has_lipidmaps",
    "has_plantcyc",
    "has_pubchem",
    "pmn_pathway_count",
    "rd_available",
    "stars",
}
MODEL_EXCLUDED_FEATURE_PREFIXES = (
    "origin_",
    "source_",
    "status_",
)

SEMANTIC_CLASS_PATTERNS: dict[str, re.Pattern] = {
    "lipid": re.compile(r"\blipid|fatty acid|acylglycerol|sphingo|ceramide|phospholipid|glycerolipid", re.I),
    "amino_acid": re.compile(r"\bamino.?acid|aminoacid|\b[dl]-(?:ala|arg|asn|asp|cys|glu|gln|gly|his|ile|leu|lys|met|phe|pro|ser|thr|trp|tyr|val)\b", re.I),
    "organic_acid": re.compile(r"\borganic.?acid|carboxylic.?acid|dicarboxylic|tricarboxylic", re.I),
    "carbohydrate": re.compile(r"\bcarbohydrate|sugar|saccharide|hexose|pentose|glucose|fructose|sucrose|galactose", re.I),
    "nucleotide": re.compile(r"\bnucleotide|nucleoside|purine|pyrimidine|\b[ATGCU][DM]P\b", re.I),
    "peptide": re.compile(r"\bpeptide|dipeptide|tripeptide|oligopeptide|polypeptide", re.I),
    "phenylpropanoid": re.compile(r"\bphenylpropanoid|cinnam|coumar|lignin|lignan|sinapyl|coniferyl|ferulic", re.I),
    "flavonoid": re.compile(r"\bflavon|isoflavon|anthocyanin|chalcone|aurone|catechin|flavan|flavanol|flavanone|quercetin|kaempferol|naringenin|luteolin|apigenin", re.I),
    "alkaloid": re.compile(r"\balkaloid|indole|quinoline|isoquinoline|tropane|pyrrolizidine|piperidine|purine.?alkaloid", re.I),
    "terpenoid": re.compile(r"\bterpene|terpenoid|monoterpene|sesquiterpene|diterpene|triterpene|isoprene|carotenoid|sterol|gibberellin", re.I),
    "cofactor_like": re.compile(r"\bcofactor|coenzyme|\bNAD|\bFAD|\bFMN|thiamine|riboflavin|pyridoxal|biotin|folate|cobalamin|pantothen", re.I),
    "hormone_related": re.compile(r"\bhormone|auxin|cytokinin|abscisic|brassinosteroid|jasmonate|salicylate|ethylene.?(?:biosyn|signal)", re.I),
}

PATHWAY_CATEGORY_KEYWORDS: dict[str, set[str]] = {
    "lipid": {"lipid", "fatty", "sphingolipid", "glycerolipid", "glycerophospholipid", "wax", "cutin", "suberin"},
    "amino_acid": {"amino", "alanine", "arginine", "asparagine", "aspartate", "cysteine", "glutamate", "glutamine", "glycine", "histidine", "isoleucine", "leucine", "lysine", "methionine", "phenylalanine", "proline", "serine", "threonine", "tryptophan", "tyrosine", "valine"},
    "carbohydrate": {"carbohydrate", "sugar", "glycolysis", "gluconeogenesis", "starch", "sucrose", "galactose", "fructose", "pentose", "mannose"},
    "nucleotide": {"nucleotide", "purine", "pyrimidine"},
    "phenylpropanoid": {"phenylpropanoid", "lignin", "lignan", "coumarin", "stilbenoid", "monolignol"},
    "flavonoid": {"flavonoid", "isoflavonoid", "anthocyanin", "flavone", "flavonol"},
    "terpenoid": {"terpenoid", "terpene", "carotenoid", "sterol", "brassinosteroid", "gibberellin", "monoterpenoid", "sesquiterpenoid", "diterpenoid", "triterpenoid"},
    "alkaloid": {"alkaloid", "indole", "tropane", "glucosinolate"},
}

EXPANDED_CANDIDATE_FIELDS = [
    "compound_id",
    "chebi_accession",
    "chebi_name",
    "pathway_id",
    "pathway_name",
    "pathway_source",
    "candidate_origin",
    "bridge_method",
    "ec_numbers",
    "reaction_equation",
]

ML_PREDICTION_FIELDS = [
    "compound_id",
    "chebi_accession",
    "chebi_name",
    "pathway_id",
    "pathway_name",
    "pathway_source",
    "candidate_origin",
    "ml_score",
    "ml_confidence",
    "is_expanded_candidate",
    "bridge_method",
    "reason",
]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class ExpandedCandidate:
    """One compound-pathway pair generated by the expanded recall layer."""

    compound_id: str
    chebi_accession: str
    chebi_name: str
    pathway_id: str
    pathway_name: str
    pathway_source: str        # AraCyc / PlantCyc / weak_KEGG
    candidate_origin: str      # pmn_direct / weak_kegg / pmn_reaction
    bridge_method: str
    ec_numbers: tuple[str, ...]
    reaction_equation: str


@dataclass(slots=True)
class TrainingPair:
    """One labeled compound-pathway pair used to train the expanded model."""

    compound_id: str
    pathway_name: str
    pathway_source: str
    candidate_origin: str
    label_source: str
    ec_numbers: tuple[str, ...] = ()
    reaction_equation: str = ""


# ---------------------------------------------------------------------------
# PMN index helpers
# ---------------------------------------------------------------------------


def _split_ec_numbers(value: str) -> tuple[str, ...]:
    """Parse PMN EC values into normalized EC-* tokens."""

    ecs: set[str] = set()
    for match in EC_RE.findall(value or ""):
        token = match.upper()
        if not token.startswith("EC-"):
            token = f"EC-{token}"
        ecs.add(token)
    return tuple(sorted(ecs))


def _build_pmn_compound_pathway_index(
    context: PipelineContext,
) -> dict[str, list[tuple[str, str, tuple[str, ...], str, str]]]:
    """Index PMN compound_id → [(pathway_name, source_db, ecs, rxn_eq, record_id)]."""

    aggregated: dict[tuple[str, str, str], dict[str, object]] = {}
    files = [
        ("AraCyc", context.paths.aracyc_compounds_path),
        ("PlantCyc", context.paths.plantcyc_compounds_path),
    ]
    for source_db, path in files:
        if not path.exists():
            continue
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            for row in reader:
                compound_id = row.get("Compound_id", "")
                pathway_name = row.get("Pathway", "")
                if not compound_id or not pathway_name:
                    continue
                key = (compound_id, pathway_name, source_db)
                entry = aggregated.setdefault(
                    key,
                    {"ecs": set(), "reactions": [], "record_id": f"{source_db}:{compound_id}"},
                )
                entry["ecs"].update(_split_ec_numbers(row.get("EC", "")))
                reaction_equation = row.get("Reaction_equation", "") or ""
                if reaction_equation and reaction_equation not in entry["reactions"]:
                    entry["reactions"].append(reaction_equation)

    index: dict[str, list[tuple[str, str, tuple[str, ...], str, str]]] = defaultdict(list)
    for (compound_id, pathway_name, source_db), entry in aggregated.items():
        index[compound_id].append(
            (
                pathway_name,
                source_db,
                tuple(sorted(entry["ecs"])),
                " ; ".join(entry["reactions"][:3]),
                str(entry["record_id"]),
            )
        )

    # Fallback for older snapshots where raw compound rows are unavailable.
    if not index:
        for record_id, record in context.plantcyc_records.items():
            for pathway in record.pathways:
                index[record.compound_id].append((pathway, record.source_db, (), "", record_id))
    return dict(index)


_PMN_PATHWAY_EC_INDEX_CACHE: dict[tuple[str, str], dict[tuple[str, str], set[str]]] = {}


def _pmn_pathway_ec_index(context: PipelineContext) -> dict[tuple[str, str], set[str]]:
    """Index PMN pathway EC sets keyed by (source_db, normalized_pathway_name)."""

    cache_key = (
        str(context.paths.aracyc_pathways_path),
        str(context.paths.plantcyc_pathways_path),
    )
    cached = _PMN_PATHWAY_EC_INDEX_CACHE.get(cache_key)
    if cached is not None:
        return cached

    index: dict[tuple[str, str], set[str]] = defaultdict(set)
    for source_db, path in [
        ("AraCyc", context.paths.aracyc_pathways_path),
        ("PlantCyc", context.paths.plantcyc_pathways_path),
    ]:
        if not path.exists():
            continue
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            for row in reader:
                normalized = normalize_name(row.get("Pathway-name", ""))
                if not normalized:
                    continue
                index[(source_db, normalized)].update(_split_ec_numbers(row.get("EC", "")))

    result = {key: set(value) for key, value in index.items()}
    _PMN_PATHWAY_EC_INDEX_CACHE[cache_key] = result
    return result


def _match_chebi_to_pmn(context: PipelineContext) -> dict[str, set[str]]:
    """Map ChEBI compound_id → set of PMN record compound_ids via name/xref."""

    chebi_to_pmn: dict[str, set[str]] = defaultdict(set)

    # Build PMN lookup indexes
    pmn_by_name: dict[str, set[str]] = defaultdict(set)
    pmn_by_chebi: dict[str, set[str]] = defaultdict(set)
    for record_id, record in context.plantcyc_records.items():
        norm = normalize_name(record.common_name)
        pmn_by_name[norm].add(record.compound_id)
        for syn in record.synonyms:
            norm_syn = normalize_name(syn)
            if norm_syn:
                pmn_by_name[norm_syn].add(record.compound_id)
        for chebi_id in record.chebi_ids:
            pmn_by_chebi[chebi_id].add(record.compound_id)

    for compound_id, compound in context.compounds.items():
        # Match by ChEBI accession
        if compound.chebi_accession in pmn_by_chebi:
            chebi_to_pmn[compound_id].update(pmn_by_chebi[compound.chebi_accession])
        # Match by compound name
        norm = normalize_name(compound.name)
        if norm in pmn_by_name:
            chebi_to_pmn[compound_id].update(pmn_by_name[norm])
        # Match via previously resolved compound_contexts
        cc = context.compound_contexts.get(compound_id)
        if cc:
            for pmn_record_id in cc.matched_plantcyc_methods:
                record = context.plantcyc_records.get(pmn_record_id)
                if record:
                    chebi_to_pmn[compound_id].add(record.compound_id)

    return dict(chebi_to_pmn)


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------


def _parse_formula(formula: str) -> dict[str, int]:
    """Parse a molecular formula string into element counts."""

    counts: dict[str, int] = {}
    for match in re.finditer(r"([A-Z][a-z]?)\s*(\d*)", formula):
        element = match.group(1)
        count = int(match.group(2)) if match.group(2) else 1
        counts[element] = counts.get(element, 0) + count
    return counts


def _extract_semantic_classes(name: str, definition: str) -> dict[str, float]:
    """Extract binary semantic class features from compound name and definition."""

    text = f"{name} {definition}"
    return {
        f"sem_{cls}": 1.0 if pattern.search(text) else 0.0
        for cls, pattern in SEMANTIC_CLASS_PATTERNS.items()
    }


def _pathway_category_tokens(pathway_name: str) -> set[str]:
    """Extract lowercase tokens from a pathway name for semantic matching."""

    return set(re.findall(r"[a-z]+", pathway_name.lower()))


def _compute_semantic_consistency(compound_classes: dict[str, float], pathway_name: str) -> float:
    """Score how well compound semantic classes match pathway category keywords."""

    pw_tokens = _pathway_category_tokens(pathway_name)
    if not pw_tokens:
        return 0.0
    score = 0.0
    active_classes = [cls for cls, val in compound_classes.items() if val > 0.5]
    for cls in active_classes:
        cls_key = cls.replace("sem_", "")
        kw_set = PATHWAY_CATEGORY_KEYWORDS.get(cls_key, set())
        if kw_set & pw_tokens:
            score += 1.0
    return min(score, 3.0) / 3.0


def _token_overlap(text_a: str, text_b: str) -> float:
    """Jaccard-like token overlap between two strings."""

    tokens_a = set(re.findall(r"[a-z0-9]+", text_a.lower()))
    tokens_b = set(re.findall(r"[a-z0-9]+", text_b.lower()))
    if not tokens_a or not tokens_b:
        return 0.0
    return len(tokens_a & tokens_b) / len(tokens_a | tokens_b)


def _compound_name_context(compound_id: str, context: PipelineContext, *, max_aliases: int = 20) -> str:
    """Compact text describing one compound for pair-level text overlap features."""

    compound = context.compounds[compound_id]
    names = [compound.name, compound.ascii_name]
    cc = context.compound_contexts.get(compound_id)
    if cc:
        for alias in cc.all_aliases[:max_aliases]:
            names.append(alias.raw_name)
    return " ".join(name for name in names if name)


def _is_model_feature(name: str) -> bool:
    """Return True if a feature is allowed into the ML matrix."""

    if name in MODEL_EXCLUDED_FEATURE_NAMES:
        return False
    return not any(name.startswith(prefix) for prefix in MODEL_EXCLUDED_FEATURE_PREFIXES)


def _extract_rdkit_features(structure) -> dict[str, float]:
    """Extract RDKit molecular descriptors. Returns zeros if RDKit unavailable."""

    defaults = {
        "rd_MolWt": 0.0,
        "rd_ExactMolWt": 0.0,
        "rd_TPSA": 0.0,
        "rd_MolLogP": 0.0,
        "rd_NumHDonors": 0.0,
        "rd_NumHAcceptors": 0.0,
        "rd_NumRotatableBonds": 0.0,
        "rd_RingCount": 0.0,
        "rd_NumAromaticRings": 0.0,
        "rd_FractionCSP3": 0.0,
        "rd_HeavyAtomCount": 0.0,
        "rd_FormalCharge": 0.0,
        "rd_available": 0.0,
    }
    if not structure or not structure.smiles:
        return defaults
    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors, rdMolDescriptors
        mol = Chem.MolFromSmiles(structure.smiles)
        if mol is None:
            return defaults
        return {
            "rd_MolWt": Descriptors.MolWt(mol),
            "rd_ExactMolWt": Descriptors.ExactMolWt(mol),
            "rd_TPSA": Descriptors.TPSA(mol),
            "rd_MolLogP": Descriptors.MolLogP(mol),
            "rd_NumHDonors": float(Descriptors.NumHDonors(mol)),
            "rd_NumHAcceptors": float(Descriptors.NumHAcceptors(mol)),
            "rd_NumRotatableBonds": float(Descriptors.NumRotatableBonds(mol)),
            "rd_RingCount": float(Descriptors.RingCount(mol)),
            "rd_NumAromaticRings": float(rdMolDescriptors.CalcNumAromaticRings(mol)),
            "rd_FractionCSP3": float(Descriptors.FractionCSP3(mol)),
            "rd_HeavyAtomCount": float(Descriptors.HeavyAtomCount(mol)),
            "rd_FormalCharge": float(Chem.GetFormalCharge(mol)),
            "rd_available": 1.0,
        }
    except Exception:
        return defaults


def _extract_morgan_fp(structure, n_bits: int = 256) -> dict[str, float]:
    """Extract Morgan fingerprint (radius=2) as individual bit features."""

    defaults = {f"mfp_{i}": 0.0 for i in range(n_bits)}
    if not structure or not structure.smiles:
        return defaults
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
        mol = Chem.MolFromSmiles(structure.smiles)
        if mol is None:
            return defaults
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=n_bits)
        return {f"mfp_{i}": float(fp[i]) for i in range(n_bits)}
    except Exception:
        return defaults


def extract_compound_features(
    compound_id: str,
    context: PipelineContext,
) -> dict[str, float]:
    """Extract all compound-level features for one ChEBI entry."""

    compound = context.compounds[compound_id]
    structure = context.structures.get(compound_id)
    xrefs = context.xrefs.get(compound_id)
    cc = context.compound_contexts.get(compound_id)
    formula_info = context.formulas.get(compound_id, ("", ""))
    formula_str, mass_str = formula_info

    features: dict[str, float] = {}

    # --- core_physchem_features ---
    elem_counts = _parse_formula(formula_str)
    features["mass"] = float(mass_str) if mass_str else 0.0
    features["has_formula"] = 1.0 if formula_str else 0.0
    for elem in ("C", "H", "N", "O", "P", "S", "Cl", "Br"):
        features[f"elem_{elem}"] = float(elem_counts.get(elem, 0))
    total_heavy = sum(v for k, v in elem_counts.items() if k != "H")
    c_count = elem_counts.get("C", 0)
    features["heavy_atom_count_formula"] = float(total_heavy)
    features["O_C_ratio"] = elem_counts.get("O", 0) / max(c_count, 1)
    features["N_C_ratio"] = elem_counts.get("N", 0) / max(c_count, 1)
    hetero = sum(v for k, v in elem_counts.items() if k not in ("C", "H"))
    features["heteroatom_ratio"] = hetero / max(total_heavy, 1)
    features["has_P"] = 1.0 if elem_counts.get("P", 0) > 0 else 0.0
    features["has_S"] = 1.0 if elem_counts.get("S", 0) > 0 else 0.0
    features["has_halogen"] = 1.0 if any(elem_counts.get(e, 0) > 0 for e in ("Cl", "Br", "F", "I")) else 0.0

    # --- rdkit_descriptor_features ---
    rdkit_feats = _extract_rdkit_features(structure)
    features.update(rdkit_feats)

    # --- morgan_fp_features ---
    fp_feats = _extract_morgan_fp(structure)
    features.update(fp_feats)

    # --- chebi_semantic_class_features ---
    sem_feats = _extract_semantic_classes(compound.name, compound.definition)
    features.update(sem_feats)

    # --- cross_db_presence_features ---
    if xrefs:
        features["has_kegg_xref"] = 1.0 if xrefs.kegg_ids else 0.0
        features["has_pubchem"] = 1.0 if xrefs.pubchem_cids else 0.0
        features["has_hmdb"] = 1.0 if xrefs.hmdb_ids else 0.0
    else:
        features["has_kegg_xref"] = 0.0
        features["has_pubchem"] = 0.0
        features["has_hmdb"] = 0.0
    features["has_lipidmaps"] = 1.0 if cc and cc.matched_lipidmaps_methods else 0.0
    features["has_aracyc"] = 0.0
    features["has_plantcyc"] = 0.0
    features["pmn_pathway_count"] = 0.0
    if cc:
        for record_id in cc.matched_plantcyc_methods:
            record = context.plantcyc_records.get(record_id)
            if record:
                if record.source_db == "AraCyc":
                    features["has_aracyc"] = 1.0
                else:
                    features["has_plantcyc"] = 1.0
                features["pmn_pathway_count"] += len(record.pathways)
    features["arabidopsis_supported"] = features["has_aracyc"]

    # --- metadata_confidence_features ---
    features["stars"] = float(compound.stars)
    features["status_checked"] = 1.0 if compound.status_id == "1" else 0.0
    features["status_obsolete"] = 1.0 if compound.status_id == "9" else 0.0

    return features


def extract_pair_features(
    compound_id: str,
    candidate: ExpandedCandidate,
    compound_features: dict[str, float],
    context: PipelineContext,
) -> dict[str, float]:
    """Extract pair-level features for a compound-pathway candidate pair."""

    compound = context.compounds[compound_id]
    features: dict[str, float] = {}

    # --- pair_match_features ---
    compound_text = _compound_name_context(compound_id, context)
    features["name_pathway_overlap"] = _token_overlap(compound_text, candidate.pathway_name)
    features["reaction_token_overlap"] = _token_overlap(compound_text, candidate.reaction_equation)
    features["has_ec"] = 1.0 if candidate.ec_numbers else 0.0
    pathway_ecs = _pmn_pathway_ec_index(context).get(
        (candidate.pathway_source, normalize_name(candidate.pathway_name)),
        set(),
    )
    candidate_ecs = set(candidate.ec_numbers)
    features["ec_pathway_overlap"] = (
        len(candidate_ecs & pathway_ecs) / len(candidate_ecs | pathway_ecs)
        if candidate_ecs and pathway_ecs
        else 0.0
    )

    # --- pathway_context_features ---
    pw_stats = context.plantcyc_pathway_stats.get(candidate.pathway_source, {}).get(
        normalize_name(candidate.pathway_name),
        {},
    )
    if pw_stats:
        features["pw_gene_count"] = float(len(pw_stats.get("gene_ids", set())))
        features["pw_pathway_id_count"] = float(len(pw_stats.get("pathway_ids", set())))
    else:
        map_id = candidate.pathway_id if candidate.pathway_id.startswith("map") else ""
        ath_id = context.map_to_ath.get(map_id, "")
        features["pw_gene_count"] = float(context.ath_gene_counts.get(ath_id, 0)) if ath_id else 0.0
        features["pw_pathway_id_count"] = 0.0

    # Plant Reactome alignment
    pr_match = 0.0
    pw_tokens = _pathway_category_tokens(candidate.pathway_name)
    for pr_id, pr_info in context.plant_reactome_pathways.items():
        pr_name = str(pr_info.get("pathway_name", ""))
        pr_tokens = _pathway_category_tokens(pr_name)
        if pw_tokens and pr_tokens and len(pw_tokens & pr_tokens) >= 2:
            pr_match = 1.0
            break
    features["has_reactome_match"] = pr_match

    # --- compound_vs_pathway semantic features ---
    sem_feats = {k: v for k, v in compound_features.items() if k.startswith("sem_")}
    features["semantic_consistency"] = _compute_semantic_consistency(sem_feats, candidate.pathway_name)
    features["compound_class_pathway_match"] = 1.0 if features["semantic_consistency"] > 0.0 else 0.0

    return features


# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------


def _features_to_matrix(feature_dicts: list[dict[str, float]], feature_names: list[str]) -> np.ndarray:
    """Convert a list of feature dicts to a numpy matrix using a fixed column order."""

    matrix = np.zeros((len(feature_dicts), len(feature_names)), dtype=np.float32)
    for i, fd in enumerate(feature_dicts):
        for j, name in enumerate(feature_names):
            matrix[i, j] = fd.get(name, 0.0)
    matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
    return matrix


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------


def write_expanded_candidates(context: PipelineContext, candidates: list[ExpandedCandidate]) -> int:
    """Write all expanded candidates before ML filtering."""

    with context.paths.expanded_candidates_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=EXPANDED_CANDIDATE_FIELDS, delimiter="\t")
        writer.writeheader()
        for c in candidates:
            writer.writerow({
                "compound_id": c.compound_id,
                "chebi_accession": c.chebi_accession,
                "chebi_name": c.chebi_name,
                "pathway_id": c.pathway_id,
                "pathway_name": c.pathway_name,
                "pathway_source": c.pathway_source,
                "candidate_origin": c.candidate_origin,
                "bridge_method": c.bridge_method,
                "ec_numbers": ";".join(c.ec_numbers),
                "reaction_equation": c.reaction_equation,
            })
    return len(candidates)


def write_ml_training_pairs(
    context: PipelineContext,
    metadata_list: list[dict[str, str]],
    labels: list[int],
) -> int:
    """Write the training pairs for audit/reproducibility."""

    with context.paths.ml_training_pairs_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "compound_id",
                "pathway_name",
                "label",
                "label_source",
                "pathway_source",
                "candidate_origin",
            ],
            delimiter="\t",
        )
        writer.writeheader()
        for meta, label in zip(metadata_list, labels):
            writer.writerow({
                "compound_id": meta["compound_id"],
                "pathway_name": meta["pathway_name"],
                "label": label,
                "label_source": meta.get("label_source", meta.get("label", "")),
                "pathway_source": meta.get("pathway_source", ""),
                "candidate_origin": meta.get("candidate_origin", ""),
            })
    return len(metadata_list)


def write_ml_predictions(
    context: PipelineContext,
    candidates: list[ExpandedCandidate],
    scores: np.ndarray,
) -> int:
    """Write ML-filtered expanded predictions (only those above threshold)."""

    count = 0
    with context.paths.ml_pathway_predictions_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=ML_PREDICTION_FIELDS, delimiter="\t")
        writer.writeheader()
        for candidate, score in zip(candidates, scores):
            if score < ML_SCORE_THRESHOLD:
                continue
            confidence = "high" if score >= 0.80 else "medium" if score >= 0.65 else "low"
            reason_parts = [
                f"expanded recall via {candidate.candidate_origin}",
                f"source={candidate.pathway_source}",
                f"ml_score={score:.3f}",
            ]
            if candidate.bridge_method:
                reason_parts.append(f"bridge={candidate.bridge_method}")
            writer.writerow({
                "compound_id": candidate.compound_id,
                "chebi_accession": candidate.chebi_accession,
                "chebi_name": candidate.chebi_name,
                "pathway_id": candidate.pathway_id,
                "pathway_name": candidate.pathway_name,
                "pathway_source": candidate.pathway_source,
                "candidate_origin": candidate.candidate_origin,
                "ml_score": f"{score:.4f}",
                "ml_confidence": confidence,
                "is_expanded_candidate": "true",
                "bridge_method": candidate.bridge_method,
                "reason": "; ".join(reason_parts),
            })
            count += 1
    return count


# ---------------------------------------------------------------------------
# 1. Candidate generation (AraCyc-first version)
# ---------------------------------------------------------------------------


def generate_aracyc_expanded_candidates(
    context: PipelineContext,
) -> list[ExpandedCandidate]:
    """Generate expanded candidates for compounds without AraCyc primary pathway hits.

    Sources:
    1. PlantCyc direct compound→pathway links
    2. Structural similarity neighbors with known pathways
    """
    # Identify target compounds (no AraCyc pathway hits)
    compounds_with_primary = {
        cid for cid, rows in context.aracyc_ranked_rows.items() if rows
    }
    target_compound_ids = set(context.compounds.keys()) - compounds_with_primary

    if not target_compound_ids:
        return []

    # Build PMN matching indexes
    chebi_to_pmn = _match_chebi_to_pmn(context)
    pmn_compound_pathways = _build_pmn_compound_pathway_index(context)

    candidates: list[ExpandedCandidate] = []
    seen_pairs: set[tuple[str, str]] = set()

    total = len(target_compound_ids)
    for idx, compound_id in enumerate(sorted(target_compound_ids, key=int), 1):
        compound = context.compounds[compound_id]

        # Source 1: PlantCyc/AraCyc direct compound→pathway links
        matched_pmn_ids = chebi_to_pmn.get(compound_id, set())
        for pmn_cid in sorted(matched_pmn_ids):
            for pathway_name, source_db, ec_numbers, reaction_equation, _record_id in pmn_compound_pathways.get(pmn_cid, []):
                pair_key = (compound_id, f"{source_db}:{pathway_name}")
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)
                candidates.append(ExpandedCandidate(
                    compound_id=compound_id,
                    chebi_accession=compound.chebi_accession,
                    chebi_name=compound.name,
                    pathway_id=f"{source_db}:{pmn_cid}:{pathway_name}",
                    pathway_name=pathway_name,
                    pathway_source=source_db,
                    candidate_origin="pmn_direct",
                    bridge_method="name_or_xref",
                    ec_numbers=ec_numbers,
                    reaction_equation=reaction_equation,
                ))

        if idx % PROGRESS_EVERY == 0 or idx == total:
            print(f"    expanded candidate generation: {idx}/{total} compounds, {len(candidates)} candidates", flush=True)

    return candidates


# ---------------------------------------------------------------------------
# 2. Training set construction (AraCyc-first version)
# ---------------------------------------------------------------------------


def build_aracyc_training_set(
    context: PipelineContext,
    compound_feature_cache: dict[str, dict[str, float]],
) -> tuple[list[dict[str, float]], list[int], list[dict[str, str]]]:
    """Build training pairs from AraCyc primary chain.

    Positive: AraCyc primary chain hits with score >= 0.50
    Negative: hard negatives (same compound, pathway not hit) + random
    """
    positives: list[TrainingPair] = []
    positive_set: set[tuple[str, str]] = set()
    positive_by_compound: defaultdict[str, set[str]] = defaultdict(set)
    pmn_index = _build_pmn_compound_pathway_index(context)

    def add_positive(pair: TrainingPair) -> None:
        normalized = normalize_name(pair.pathway_name)
        if not normalized:
            return
        key = (pair.compound_id, normalized)
        if key in positive_set:
            return
        positives.append(pair)
        positive_set.add(key)
        positive_by_compound[pair.compound_id].add(normalized)

    # Collect positives from AraCyc primary chain
    for compound_id, rows in context.aracyc_ranked_rows.items():
        for row in rows:
            if row.score >= 0.50:
                add_positive(TrainingPair(
                    compound_id=compound_id,
                    pathway_name=row.pathway_name,
                    pathway_source=row.source_db,
                    candidate_origin="primary_aracyc",
                    label_source="aracyc_primary_score_ge_0.50",
                ))

    if not positives:
        return [], [], []

    # Build pathway catalog for negative sampling
    pathway_catalog: list[TrainingPair] = []
    pathway_catalog_seen: set[tuple[str, str]] = set()

    for _pmn_cid, rows in pmn_index.items():
        for pathway_name, source_db, ec_numbers, reaction_equation, _record_id in rows:
            normalized = normalize_name(pathway_name)
            key = (source_db, normalized)
            if key in pathway_catalog_seen or not normalized:
                continue
            pathway_catalog_seen.add(key)
            pathway_catalog.append(TrainingPair(
                compound_id="",
                pathway_name=pathway_name,
                pathway_source=source_db,
                candidate_origin="pmn_direct",
                label_source="negative_pool",
                ec_numbers=ec_numbers,
                reaction_equation=reaction_equation,
            ))

    rng = np.random.RandomState(RANDOM_SEED)
    negatives: list[TrainingPair] = []
    negative_seen: set[tuple[str, str, str]] = set()

    def add_negative(compound_id: str, entry: TrainingPair, label_source: str) -> None:
        normalized = normalize_name(entry.pathway_name)
        if not normalized or normalized in positive_by_compound.get(compound_id, set()):
            return
        key = (compound_id, entry.pathway_source, normalized)
        if key in negative_seen:
            return
        negative_seen.add(key)
        negatives.append(TrainingPair(
            compound_id=compound_id,
            pathway_name=entry.pathway_name,
            pathway_source=entry.pathway_source,
            candidate_origin=entry.candidate_origin,
            label_source=label_source,
            ec_numbers=entry.ec_numbers,
            reaction_equation=entry.reaction_equation,
        ))

    # Hard negatives: for each positive compound, pick random pathways not hit
    for pair in positives:
        if not pathway_catalog:
            break
        attempts = 0
        added = 0
        while added < 2 and attempts < 50:
            attempts += 1
            entry = pathway_catalog[int(rng.randint(len(pathway_catalog)))]
            before = len(negatives)
            add_negative(pair.compound_id, entry, "hard_negative")
            if len(negatives) > before:
                added += 1

    # Balance: keep at most 3x negatives per positive
    max_neg = len(positives) * 3
    if len(negatives) > max_neg:
        indices = rng.choice(len(negatives), size=max_neg, replace=False)
        negatives = [negatives[i] for i in indices]

    # Extract features
    all_features: list[dict[str, float]] = []
    all_labels: list[int] = []
    all_metadata: list[dict[str, str]] = []

    for pair in positives:
        cid = pair.compound_id
        if cid not in compound_feature_cache:
            compound_feature_cache[cid] = extract_compound_features(cid, context)
        cf = compound_feature_cache[cid]
        dummy = ExpandedCandidate(
            compound_id=cid,
            chebi_accession=context.compounds[cid].chebi_accession,
            chebi_name=context.compounds[cid].name,
            pathway_id=f"train:{pair.pathway_name}",
            pathway_name=pair.pathway_name,
            pathway_source=pair.pathway_source,
            candidate_origin=pair.candidate_origin,
            bridge_method="training_positive",
            ec_numbers=pair.ec_numbers,
            reaction_equation=pair.reaction_equation,
        )
        pf = extract_pair_features(cid, dummy, cf, context)
        all_features.append({**cf, **pf})
        all_labels.append(1)
        all_metadata.append({
            "compound_id": cid,
            "pathway_name": pair.pathway_name,
            "label": "positive",
            "label_source": pair.label_source,
            "pathway_source": pair.pathway_source,
            "candidate_origin": pair.candidate_origin,
        })

    for pair in negatives:
        cid = pair.compound_id
        if cid not in compound_feature_cache:
            compound_feature_cache[cid] = extract_compound_features(cid, context)
        cf = compound_feature_cache[cid]
        dummy = ExpandedCandidate(
            compound_id=cid,
            chebi_accession=context.compounds[cid].chebi_accession,
            chebi_name=context.compounds[cid].name,
            pathway_id=f"train_neg:{pair.pathway_name}",
            pathway_name=pair.pathway_name,
            pathway_source=pair.pathway_source,
            candidate_origin=pair.candidate_origin,
            bridge_method="training_negative",
            ec_numbers=pair.ec_numbers,
            reaction_equation=pair.reaction_equation,
        )
        pf = extract_pair_features(cid, dummy, cf, context)
        all_features.append({**cf, **pf})
        all_labels.append(0)
        all_metadata.append({
            "compound_id": cid,
            "pathway_name": pair.pathway_name,
            "label": "negative",
            "label_source": pair.label_source,
            "pathway_source": pair.pathway_source,
            "candidate_origin": pair.candidate_origin,
        })

    return all_features, all_labels, all_metadata


# ---------------------------------------------------------------------------
# 3. Model training + prediction
# ---------------------------------------------------------------------------


def train_and_predict_aracyc(
    train_features: list[dict[str, float]],
    train_labels: list[int],
    train_metadata: list[dict[str, str]],
    predict_features: list[dict[str, float]],
    context: PipelineContext,
) -> tuple[np.ndarray, dict[str, object]]:
    """Train and predict using LogisticRegression with group-aware CV."""

    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold, cross_val_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score, precision_score, recall_score
    import joblib

    if not train_features:
        return np.array([]), {}

    all_feature_names = sorted(set().union(*(fd.keys() for fd in train_features)))
    feature_names = [name for name in all_feature_names if _is_model_feature(name)]
    excluded_feature_names = [name for name in all_feature_names if name not in feature_names]
    if not feature_names:
        return np.array([]), {"status": "skipped", "reason": "no model features"}

    X_train = _features_to_matrix(train_features, feature_names)
    y_train = np.array(train_labels, dtype=np.int32)
    groups = np.array([meta.get("compound_id", str(i)) for i, meta in enumerate(train_metadata)])
    unique_groups = np.unique(groups)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = LogisticRegression(
        C=1.0, max_iter=1000, random_state=RANDOM_SEED,
        solver="lbfgs", class_weight="balanced",
    )

    # Cross-validation
    cv_scores = np.array([])
    cv_strategy = "skipped"
    n_splits = min(5, len(unique_groups))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if n_splits >= 2 and len(set(y_train)) > 1:
            cv_model = make_pipeline(
                StandardScaler(),
                LogisticRegression(C=1.0, max_iter=1000, random_state=RANDOM_SEED, solver="lbfgs", class_weight="balanced"),
            )
            try:
                from sklearn.model_selection import StratifiedGroupKFold
                cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
                cv_strategy = "StratifiedGroupKFold"
            except ImportError:
                cv = GroupKFold(n_splits=n_splits)
                cv_strategy = "GroupKFold"
            try:
                raw_cv_scores = cross_val_score(cv_model, X_train, y_train, groups=groups, cv=cv, scoring="roc_auc")
                cv_scores = raw_cv_scores[np.isfinite(raw_cv_scores)]
            except ValueError:
                cv_strategy = f"{cv_strategy}_failed"

    model.fit(X_train_scaled, y_train)

    y_train_pred = model.predict(X_train_scaled)
    y_train_prob = model.predict_proba(X_train_scaled)[:, 1]
    train_auc = roc_auc_score(y_train, y_train_prob) if len(set(y_train)) > 1 else 0.0
    train_precision = precision_score(y_train, y_train_pred, zero_division=0.0)
    train_recall = recall_score(y_train, y_train_pred, zero_division=0.0)

    if predict_features:
        X_pred = _features_to_matrix(predict_features, feature_names)
        X_pred_scaled = scaler.transform(X_pred)
        pred_probs = model.predict_proba(X_pred_scaled)[:, 1]
    else:
        pred_probs = np.array([])

    # Save model
    model_dir = context.paths.ml_model_path.parent
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "scaler": scaler, "feature_names": feature_names}, context.paths.ml_model_path)

    # Feature importances
    coef = model.coef_[0]
    top_positive_idx = np.argsort(coef)[-10:][::-1]
    top_negative_idx = np.argsort(coef)[:10]
    top_positive_features = [(feature_names[i], round(float(coef[i]), 4)) for i in top_positive_idx]
    top_negative_features = [(feature_names[i], round(float(coef[i]), 4)) for i in top_negative_idx]

    score_distribution = {
        "min": round(float(np.min(pred_probs)), 4) if len(pred_probs) else 0.0,
        "p50": round(float(np.percentile(pred_probs, 50)), 4) if len(pred_probs) else 0.0,
        "p90": round(float(np.percentile(pred_probs, 90)), 4) if len(pred_probs) else 0.0,
        "p95": round(float(np.percentile(pred_probs, 95)), 4) if len(pred_probs) else 0.0,
        "p99": round(float(np.percentile(pred_probs, 99)), 4) if len(pred_probs) else 0.0,
        "max": round(float(np.max(pred_probs)), 4) if len(pred_probs) else 0.0,
        "mean": round(float(np.mean(pred_probs)), 4) if len(pred_probs) else 0.0,
    }

    metadata = {
        "model_version": MODEL_VERSION,
        "feature_set_version": FEATURE_SET_VERSION,
        "model_type": "LogisticRegression",
        "feature_count": len(feature_names),
        "raw_feature_count": len(all_feature_names),
        "excluded_model_feature_count": len(excluded_feature_names),
        "excluded_model_features": excluded_feature_names,
        "training_samples": len(train_labels),
        "positive_samples": int(sum(train_labels)),
        "negative_samples": int(len(train_labels) - sum(train_labels)),
        "cv_strategy": cv_strategy,
        "cv_group_count": int(len(unique_groups)),
        "cv_roc_auc_mean": round(float(np.mean(cv_scores)), 4) if len(cv_scores) else 0.0,
        "cv_roc_auc_std": round(float(np.std(cv_scores)), 4) if len(cv_scores) else 0.0,
        "train_roc_auc": round(float(train_auc), 4),
        "train_precision": round(float(train_precision), 4),
        "train_recall": round(float(train_recall), 4),
        "ml_score_threshold": ML_SCORE_THRESHOLD,
        "expanded_candidates_scored": len(predict_features),
        "expanded_candidates_above_threshold": int(np.sum(pred_probs >= ML_SCORE_THRESHOLD)) if len(pred_probs) > 0 else 0,
        "score_distribution": score_distribution,
        "score_calibration_warning": bool(len(pred_probs) > 0 and np.all(pred_probs >= ML_SCORE_THRESHOLD)),
        "leakage_features_in_model": [n for n in feature_names if n.startswith(("source_", "origin_"))],
        "top_positive_features": top_positive_features,
        "top_negative_features": top_negative_features,
        "random_seed": RANDOM_SEED,
        "morgan_fp_bits": 256,
    }

    return pred_probs, metadata


# ---------------------------------------------------------------------------
# 4. Step runner
# ---------------------------------------------------------------------------


def run(context: PipelineContext) -> PipelineContext:
    """Run AraCyc-first ML supplemental recall pipeline."""

    print("  Step 6a: ML expanded recall — generating candidates...", flush=True)

    # 1. Generate candidates
    candidates = generate_aracyc_expanded_candidates(context)
    n_candidates = write_expanded_candidates(context, candidates)
    print(f"    Generated {n_candidates} expanded candidates", flush=True)
    context.expanded_candidates_count = n_candidates

    if not candidates:
        context.add_note("Step 6a: no expanded candidates — all compounds may have primary coverage.")
        context.ml_predictions_count = 0
        write_ml_predictions(context, [], np.array([]))
        with context.paths.ml_model_metadata_path.open("w", encoding="utf-8") as handle:
            json.dump({"status": "skipped", "reason": "no expanded candidates"}, handle, indent=2)
        return context

    # 2. Build training set from AraCyc primary chain
    print("    Building training set from AraCyc primary chain...", flush=True)
    compound_feature_cache: dict[str, dict[str, float]] = {}
    train_features, train_labels, train_metadata = build_aracyc_training_set(context, compound_feature_cache)
    write_ml_training_pairs(context, train_metadata, train_labels)
    n_pos = sum(train_labels)
    print(f"    Training set: {n_pos} positive, {len(train_labels) - n_pos} negative pairs", flush=True)

    if not train_features or n_pos < 10:
        context.add_note("Step 6a: insufficient training data for ML model.")
        default_scores = np.full(len(candidates), 0.50)
        n_predictions = write_ml_predictions(context, candidates, default_scores)
        context.ml_predictions_count = n_predictions
        with context.paths.ml_model_metadata_path.open("w", encoding="utf-8") as handle:
            json.dump({"status": "fallback", "reason": "insufficient training data"}, handle, indent=2)
        return context

    # 3. Extract features for candidates
    print("    Extracting features for expanded candidates...", flush=True)
    predict_features: list[dict[str, float]] = []
    for i, candidate in enumerate(candidates):
        if candidate.compound_id not in compound_feature_cache:
            compound_feature_cache[candidate.compound_id] = extract_compound_features(
                candidate.compound_id, context
            )
        cf = compound_feature_cache[candidate.compound_id]
        pf = extract_pair_features(candidate.compound_id, candidate, cf, context)
        predict_features.append({**cf, **pf})
        if (i + 1) % PROGRESS_EVERY == 0 or (i + 1) == len(candidates):
            print(f"      features: {i + 1}/{len(candidates)}", flush=True)

    # 4. Train and predict
    print("    Training LogisticRegression and predicting...", flush=True)
    pred_scores, model_metadata = train_and_predict_aracyc(
        train_features, train_labels, train_metadata, predict_features, context
    )

    # 5. Write predictions
    n_predictions = write_ml_predictions(context, candidates, pred_scores)
    context.ml_predictions_count = n_predictions
    print(f"    ML predictions above threshold ({ML_SCORE_THRESHOLD}): {n_predictions}", flush=True)

    # 6. Write metadata with coverage stats
    primary_reference_keys = {
        matches[0].reference_compound_key
        for compound_id, rows in context.aracyc_ranked_rows.items()
        if rows
        for matches in [context.aracyc_matches_by_compound.get(compound_id, [])]
        if matches and matches[0].source_db == "AraCyc"
    }
    expanded_reference_keys = {
        matches[0].reference_compound_key
        for candidate, score in zip(candidates, pred_scores)
        if score >= ML_SCORE_THRESHOLD
        for matches in [context.aracyc_matches_by_compound.get(candidate.compound_id, [])]
        if matches and matches[0].source_db == "AraCyc"
    }
    compounds_with_primary = len(primary_reference_keys)
    compounds_with_expanded = len(primary_reference_keys | expanded_reference_keys)
    total_compounds = context.aracyc_reference_total()
    model_metadata["coverage_basis"] = "aracyc_reference_compounds"
    model_metadata["chebi_input_total"] = len(context.compounds)
    model_metadata["coverage_primary_only"] = compounds_with_primary
    model_metadata["coverage_with_expanded"] = compounds_with_expanded
    model_metadata["total_compounds"] = total_compounds
    model_metadata["coverage_primary_pct"] = round(100 * compounds_with_primary / max(total_compounds, 1), 2)
    model_metadata["coverage_expanded_pct"] = round(
        100 * compounds_with_expanded / max(total_compounds, 1), 2
    )

    with context.paths.ml_model_metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(model_metadata, handle, indent=2, ensure_ascii=False)

    print(
        f"    Coverage: primary={compounds_with_primary} ({model_metadata['coverage_primary_pct']}%) "
        f"→ with expanded={compounds_with_expanded} ({model_metadata['coverage_expanded_pct']}%)",
        flush=True,
    )

    return context
