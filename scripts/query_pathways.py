#!/usr/bin/env python3
"""Interactive and one-shot query interface for the preprocessed name-first flow."""

from __future__ import annotations

import argparse
import csv
import json
import math
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
    mapping_reason: str


@dataclass(slots=True)
class PathwayAnnotation:
    pathway_id: str
    pathway_target_type: str
    map_pathway_id: str
    ath_pathway_id: str
    pathway_name: str
    pathway_group: str
    pathway_category: str
    map_pathway_compound_count: int
    ath_gene_count: int
    reactome_matches: str


@dataclass(slots=True)
class PlantEvidence:
    plant_support_source: str
    plant_support_examples: str
    plant_support_bonus: float
    plant_support_gene_count: int


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
    formula_blocked_candidates: int


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


def load_formula_indexes(path: Path):
    """Load exact-name and compact-name formula lookups."""

    exact_index = defaultdict(set)
    compact_index = defaultdict(set)
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            formula_keys = {value for value in row["formula_keys"].split(";") if value}
            if row["exact_name"]:
                exact_index[row["exact_name"]].update(formula_keys)
            if row["compact_name"]:
                compact_index[row["compact_name"]].update(formula_keys)
    return dict(exact_index), dict(compact_index)


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
                pathway_group=row["pathway_group"],
                pathway_category=row["pathway_category"],
                map_pathway_compound_count=int(row["map_pathway_compound_count"] or 0),
                ath_gene_count=int(row["ath_gene_count"] or 0),
                reactome_matches=row["reactome_matches"],
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
                plant_support_examples=row["plant_support_examples"],
                plant_support_bonus=float(row["plant_support_bonus"]),
                plant_support_gene_count=int(row["plant_support_gene_count"] or 0),
            )
            current = evidence[row["compound_id"]].get(row["pathway_name_normalized"])
            if current is None or item.plant_support_bonus > current.plant_support_bonus:
                evidence[row["compound_id"]][row["pathway_name_normalized"]] = item
    return {compound_id: dict(values) for compound_id, values in evidence.items()}


def choose_best_hit(hits, variant_name, stage, best_mapping_score_by_compound):
    """Pick the best compound when a standard-name or alias tier produced hits."""

    best = None
    for record in hits.values():
        mapping_score = best_mapping_score_by_compound.get(record.compound_id, 0.0)
        score = (
            VARIANT_PRIORITY[variant_name]
            + (0.16 if stage == "standard_name" else 0.08)
            + ALIAS_SOURCE_WEIGHTS.get(record.source_type, 0.0)
            + (mapping_score * 0.10)
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
            formula_validated=False,
            formula_blocked_candidates=0,
        )
        if best is None or (candidate.resolution_score, candidate.compound_id) > (best.resolution_score, best.compound_id):
            best = candidate
    return best


def resolve_name(
    query: str,
    standard_indexes,
    alias_indexes,
    primary_records,
    formula_index_provider,
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
            return choose_best_hit(hits, variant_name, "standard_name", best_mapping_score_by_compound)

    for variant_name in VARIANT_ORDER:
        value = variants[variant_name]
        if not value:
            continue
        hits = alias_indexes[variant_name].get(value)
        if hits:
            return choose_best_hit(hits, variant_name, "alias_name", best_mapping_score_by_compound)

    compact_query = variants["compact"]
    if not compact_query or len(compact_query) < MIN_CHAR_FUZZY_COMPACT_LEN:
        return None

    exact_formula_index, compact_formula_index = formula_index_provider()
    input_formula_keys = set(compact_formula_index.get(compact_query, set()))
    best = None
    blocked = 0
    for primary in primary_records.values():
        if not primary.compact_name:
            continue
        if abs(len(compact_query) - len(primary.compact_name)) > 2:
            continue
        candidate_formula_keys = set(compact_formula_index.get(primary.compact_name, set()))
        if not char_edit_distance_at_most_two(compact_query, primary.compact_name):
            continue
        if not formula_sets_compatible(input_formula_keys, candidate_formula_keys):
            blocked += 1
            continue
        mapping_score = best_mapping_score_by_compound.get(primary.compound_id, 0.0)
        formula_validated = bool(input_formula_keys and candidate_formula_keys and (input_formula_keys & candidate_formula_keys))
        score = CHAR_EDIT2_BASE_SCORE + 0.16 + (mapping_score * 0.10) + (0.02 if formula_validated else 0.0)
        candidate = ResolvedName(
            compound_id=primary.compound_id,
            chebi_accession=primary.chebi_accession,
            canonical_name=primary.canonical_name,
            matched_alias=primary.alias,
            alias_source=primary.source_type,
            match_stage="fuzzy_typo_correction",
            matched_variant="compact_formula_edit2",
            resolution_score=score,
            formula_validated=formula_validated,
            formula_blocked_candidates=blocked,
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
                "plantcyc_support": round(plant_bonus, 3),
                "generic_pathway_penalty": generic_penalty,
                "map_fallback_penalty": -0.10 if not annotation.ath_pathway_id else 0.0,
            }
            score = max(0.0, min(sum(contributions.values()), 0.999))
            confidence = "high" if score >= 0.70 and annotation.ath_pathway_id and annotation.ath_gene_count > 0 else "medium" if score >= 0.40 else "low"
            positive_features = [f"{k}={v:.3f}" for k, v in sorted(contributions.items(), key=lambda item: item[1], reverse=True) if v > 0][:3]
            negative_features = [f"{k}={v:.3f}" for k, v in sorted(contributions.items(), key=lambda item: item[1]) if v < 0][:2]
            reason_parts = [
                f"{confidence} confidence",
                f"score {score:.3f}",
                f"mapping {mapping.kegg_compound_id} ({mapping.kegg_primary_name}) via {mapping.mapping_method}",
            ]
            if annotation.pathway_target_type == "map_fallback":
                reason_parts.append("used map fallback because no ath-specific pathway was available")
            if annotation.ath_gene_count:
                reason_parts.append(f"ath gene support: {annotation.ath_gene_count}")
            if plant:
                reason_parts.append(f"{plant.plant_support_source} support: {plant.plant_support_examples}")
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
                },
            )
            if score > float(entry["score"]):
                entry["score"] = score
                entry["confidence"] = confidence
                entry["reason"] = reason
                entry["annotation"] = annotation
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
                reason=str(entry["reason"]),
            )
        )
    return rows


def print_result(query: str, resolved: ResolvedName, mappings, pathways: list[RankedPathway], top_k: int) -> None:
    """Render a one-shot query result to the terminal."""

    print(f"Query: {query}")
    print(f"Standard name: {resolved.canonical_name}")
    print(f"Matched alias: {resolved.matched_alias}")
    print(f"Match stage: {resolved.match_stage}")
    print(f"Match variant: {resolved.matched_variant}")
    if resolved.match_stage == "fuzzy_typo_correction":
        print(f"Formula validated: {'yes' if resolved.formula_validated else 'no'}")
        print(f"Formula-blocked candidates: {resolved.formula_blocked_candidates}")
    if mappings:
        print(
            "KEGG compound IDs: "
            + "; ".join(f"{mapping.kegg_compound_id} ({mapping.kegg_primary_name})" for mapping in mappings)
        )
    else:
        print("KEGG compound IDs: none")
    if not pathways:
        print("Top pathways: none")
        return
    print(f"Top {top_k} pathways:")
    for pathway in pathways:
        print(
            f"{pathway.pathway_rank}. {pathway.pathway_name} [{pathway.pathway_target_id}] "
            f"score={pathway.score:.3f} confidence={pathway.confidence_level}"
        )
        print(f"   reason: {pathway.reason}")


def load_preprocessed_state(workdir: Path, verbose: bool = True):
    """Load all query-time indexes from outputs/preprocessed/."""

    data_dir = preprocessed_dir(workdir)
    paths = {
        "name_normalization_index": data_dir / "name_normalization_index.tsv",
        "name_to_formula_index": data_dir / "name_to_formula_index.tsv",
        "name_to_kegg_index": data_dir / "name_to_kegg_index.tsv",
        "compound_to_pathway_index": data_dir / "compound_to_pathway_index.tsv",
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
        print("- KEGG compound-to-pathway index", flush=True)
    compound_to_pathway = load_compound_to_pathway(paths["compound_to_pathway_index"])
    if verbose:
        print("- map-to-ath index", flush=True)
    map_to_ath = load_map_to_ath(paths["map_to_ath_index"])
    if verbose:
        print("- pathway annotation index", flush=True)
    pathway_annotations = load_pathway_annotations(paths["pathway_annotation_index"])
    if verbose:
        print("- plant evidence index", flush=True)
    plant_evidence = load_plant_evidence(paths["plant_evidence_index"])
    if verbose:
        print("Preprocessed indexes loaded.", flush=True)
    return {
        "metadata": metadata,
        "name_indexes": name_indexes,
        "formula_indexes": None,
        "formula_index_path": paths["name_to_formula_index"],
        "verbose": verbose,
        "mappings_by_name": mappings_by_name,
        "best_mapping_score_by_compound": best_mapping_score_by_compound,
        "compound_to_pathway": compound_to_pathway,
        "map_to_ath": map_to_ath,
        "pathway_annotations": pathway_annotations,
        "plant_evidence": plant_evidence,
    }


def get_formula_indexes(state):
    """Load the formula index lazily because only typo correction needs it."""

    if state["formula_indexes"] is None:
        if state.get("verbose"):
            print("- name-to-formula index (on demand for typo correction)", flush=True)
        state["formula_indexes"] = load_formula_indexes(state["formula_index_path"])
    return state["formula_indexes"]


def run_query(query: str, state, top_k: int) -> tuple[ResolvedName | None, list[MappingRecord], list[RankedPathway]]:
    """Run the full query pipeline for one free-text name."""

    standard_indexes, alias_indexes, primary_records = state["name_indexes"]
    resolved = resolve_name(
        query,
        standard_indexes,
        alias_indexes,
        primary_records,
        lambda: get_formula_indexes(state),
        state["best_mapping_score_by_compound"],
    )
    if resolved is None:
        return None, [], []
    mappings = lookup_mappings_for_standard_name(resolved.canonical_name, state["mappings_by_name"])
    pathways = rank_pathways(
        resolved,
        mappings,
        state["compound_to_pathway"],
        state["map_to_ath"],
        state["pathway_annotations"],
        state["plant_evidence"],
        top_k,
    )
    return resolved, mappings, pathways


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
        resolved, mappings, pathways = run_query(query, state, top_k)
        if resolved is None:
            print("No matching standard name found.")
            continue
        print_result(query, resolved, mappings, pathways, top_k)


def main() -> None:
    """Load the preprocessed indexes once, then run one-shot or interactive queries."""

    args = parse_args()
    workdir = Path(args.workdir).resolve()
    state = load_preprocessed_state(workdir)
    if args.name:
        resolved, mappings, pathways = run_query(args.name, state, args.top_k)
        if resolved is None:
            print("No matching standard name found.")
            return
        print_result(args.name, resolved, mappings, pathways, args.top_k)
        return
    repl(state, args.top_k)


if __name__ == "__main__":
    main()
