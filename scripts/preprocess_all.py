#!/usr/bin/env python3
"""Build preprocessed indexes for the name-first metabolite query pipeline."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from datetime import datetime, UTC
from pathlib import Path

from process_chebi_to_pathways_v2 import (
    XrefInfo,
    build_candidate_mappings,
    build_compound_context,
    build_name_formula_index,
    build_kegg_structure_indexes,
    build_variants,
    load_ath_gene_counts,
    load_ath_pathways,
    load_base_aliases,
    load_comments_profile,
    load_compounds,
    load_formula_info,
    load_kegg_compounds,
    load_kegg_pathway_links,
    load_lipidmaps_records,
    load_map_pathways,
    load_pathway_categories,
    load_plantcyc_compounds,
    load_plantcyc_pathway_stats,
    load_pubchem_synonyms,
    load_reactome_pathways,
    load_structures,
    load_xrefs,
    mapping_confidence_label,
    mapping_method_label,
    normalize_inchi_key,
    normalize_name,
    select_candidates,
)


PREPROCESSED_DIRNAME = "preprocessed"
VERSION = "preprocess-query-v3"
PROGRESS_EVERY = 10000


def inchi_key_prefix(value: str) -> str:
    """Return the connectivity block of an InChIKey when available."""

    text = (value or "").strip().upper()
    if not text:
        return ""
    return text.split("-", 1)[0]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for preprocess execution."""

    parser = argparse.ArgumentParser(description="Build preprocessed indexes for the name-first pathway query flow.")
    parser.add_argument("--workdir", default=".", help="Workspace containing compounds.tsv, comments.tsv, refs/, and outputs/.")
    return parser.parse_args()


def ensure_exists(path: Path) -> None:
    """Raise a clear error when a required input file is missing."""

    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")


def required_paths(workdir: Path) -> list[Path]:
    """Return all input files needed to build the preprocessed indexes."""

    refs = workdir / "refs"
    return [
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


def build_standard_name_indexes(kegg_compounds):
    """Build KEGG primary-name lookup tables."""

    indexes = {
        "exact": defaultdict(set),
        "compact": defaultdict(set),
        "singular": defaultdict(set),
        "stereo_stripped": defaultdict(set),
    }
    for kegg_id, kegg in kegg_compounds.items():
        indexes["exact"][kegg.primary_exact].add(kegg_id)
        indexes["compact"][kegg.primary_compact].add(kegg_id)
        indexes["singular"][kegg.primary_singular].add(kegg_id)
        indexes["stereo_stripped"][kegg.primary_stereo_stripped].add(kegg_id)
    return indexes


def write_name_normalization_index(
    path: Path,
    compounds,
    compound_contexts,
) -> int:
    """Write the standard-name plus alias index consumed by query step 1."""

    row_count = 0
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "compound_id",
                "chebi_accession",
                "canonical_name",
                "alias",
                "source_type",
                "is_primary_name",
                "exact_name",
                "compact_name",
                "singular_name",
                "stereo_stripped_name",
            ],
            delimiter="\t",
        )
        writer.writeheader()
        for compound_id in sorted(compounds, key=int):
            compound = compounds[compound_id]
            compound_context = compound_contexts[compound_id]
            for alias in sorted(compound_context.all_aliases, key=lambda item: (item.source_type, item.raw_name)):
                writer.writerow(
                    {
                        "compound_id": compound_id,
                        "chebi_accession": compound.chebi_accession,
                        "canonical_name": compound.name,
                        "alias": alias.raw_name,
                        "source_type": alias.source_type,
                        "is_primary_name": str(alias.source_type == "compound_name").lower(),
                        "exact_name": alias.exact,
                        "compact_name": alias.compact,
                        "singular_name": alias.singular,
                        "stereo_stripped_name": alias.stereo_stripped,
                    }
                )
                row_count += 1
    return row_count


def aggregate_name_formula_rows(base_aliases, structures, plantcyc_records, lipidmaps_records):
    """Aggregate normalized names to lightweight structure signatures.

    The query layer uses this table as the lazy-loaded safety net for weak name
    matches. It carries coarse structure identity signals:

    - formula_keys: first-pass chemistry compatibility gate
    - InChIKey full values: exact structure identity when available
    - InChIKey prefixes: connectivity-level gate when full structure is missing
    """

    rows = {}

    def add_name(name: str, formula_key_value: str, inchi_key_value: str, source: str) -> None:
        if not name:
            return
        variants = build_variants(name)
        key = (variants["exact"], variants["compact"])
        if not key[0] or not key[1]:
            return
        entry = rows.setdefault(
            key,
            {
                "exact_name": key[0],
                "compact_name": key[1],
                "formula_keys": set(),
                "inchi_key_prefixes": set(),
                "inchi_key_fulls": set(),
                "evidence_sources": set(),
            },
        )
        if formula_key_value:
            entry["formula_keys"].add(formula_key_value)
        normalized_inchi = (inchi_key_value or "").strip().upper()
        if normalized_inchi:
            entry["inchi_key_fulls"].add(normalized_inchi)
            entry["inchi_key_prefixes"].add(inchi_key_prefix(normalized_inchi))
        entry["evidence_sources"].add(source)

    for compound_id, aliases in base_aliases.items():
        structure = structures.get(compound_id)
        if not structure:
            continue
        for alias in aliases:
            add_name(alias.raw_name, structure.formula_key, structure.standard_inchi_key, "ChEBI")

    for record in plantcyc_records.values():
        add_name(record.common_name, record.formula_key, "", record.source_db)
        for synonym in sorted(record.synonyms):
            add_name(synonym, record.formula_key, "", record.source_db)

    for record in lipidmaps_records.values():
        add_name(record.common_name, record.formula_key, record.inchi_key, "LIPID MAPS")
        add_name(record.systematic_name, record.formula_key, record.inchi_key, "LIPID MAPS")
        for synonym in sorted(record.synonyms):
            add_name(synonym, record.formula_key, record.inchi_key, "LIPID MAPS")

    return rows


def write_name_to_formula_index(path: Path, rows) -> int:
    """Write the structure lookup used by query-side weak-match validation."""

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "exact_name",
                "compact_name",
                "formula_keys",
                "formula_count",
                "inchi_key_prefixes",
                "inchi_key_prefix_count",
                "inchi_key_fulls",
                "inchi_key_full_count",
                "evidence_sources",
            ],
            delimiter="\t",
        )
        writer.writeheader()
        for key in sorted(rows):
            entry = rows[key]
            writer.writerow(
                {
                    "exact_name": entry["exact_name"],
                    "compact_name": entry["compact_name"],
                    "formula_keys": ";".join(sorted(entry["formula_keys"])),
                    "formula_count": len(entry["formula_keys"]),
                    "inchi_key_prefixes": ";".join(sorted(value for value in entry["inchi_key_prefixes"] if value)),
                    "inchi_key_prefix_count": len([value for value in entry["inchi_key_prefixes"] if value]),
                    "inchi_key_fulls": ";".join(sorted(value for value in entry["inchi_key_fulls"] if value)),
                    "inchi_key_full_count": len([value for value in entry["inchi_key_fulls"] if value]),
                    "evidence_sources": ";".join(sorted(entry["evidence_sources"])),
                }
            )
    return len(rows)


def write_name_to_kegg_index(path: Path, compounds, selected_by_compound) -> int:
    """Write the standard-name -> KEGG mapping index consumed by query step 2.

    The query layer now uses the canonical standardized name returned by step 1
    to look up KEGG compound IDs. We still keep compound_id and accession as
    provenance columns, but the exported table is centered around the canonical
    name variants rather than a compound_id-only lookup.
    """

    row_count = 0
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "compound_id",
                "chebi_accession",
                "canonical_name",
                "canonical_exact_name",
                "canonical_compact_name",
                "canonical_singular_name",
                "canonical_stereo_stripped_name",
                "kegg_compound_id",
                "kegg_primary_name",
                "mapping_score",
                "mapping_confidence_level",
                "mapping_method",
                "direct_kegg_xref",
                "has_structure_evidence",
                "used_pubchem_synonym",
                "best_alias",
                "best_alias_source",
                "best_variant",
                "evidence_count",
                "external_sources",
                "mapping_reason",
            ],
            delimiter="\t",
        )
        writer.writeheader()
        for compound_id in sorted(compounds, key=int):
            compound = compounds[compound_id]
            canonical_variants = build_variants(compound.name)
            for mapping in selected_by_compound.get(compound_id, []):
                writer.writerow(
                    {
                        "compound_id": compound_id,
                        "chebi_accession": compound.chebi_accession,
                        "canonical_name": compound.name,
                        "canonical_exact_name": canonical_variants["exact"],
                        "canonical_compact_name": canonical_variants["compact"],
                        "canonical_singular_name": canonical_variants["singular"],
                        "canonical_stereo_stripped_name": canonical_variants["stereo_stripped"],
                        "kegg_compound_id": mapping.kegg_compound_id,
                        "kegg_primary_name": mapping.kegg_primary_name,
                        "mapping_score": f"{mapping.final_score:.3f}",
                        "mapping_confidence_level": mapping_confidence_label(mapping.final_score),
                        "mapping_method": mapping_method_label(mapping),
                        "direct_kegg_xref": str(mapping.direct_kegg_xref).lower(),
                        "has_structure_evidence": str(mapping.has_structure_evidence).lower(),
                        "used_pubchem_synonym": str(mapping.used_pubchem_synonym).lower(),
                        "best_alias": mapping.best_alias,
                        "best_alias_source": mapping.best_source_type,
                        "best_variant": mapping.best_variant,
                        "evidence_count": mapping.evidence_count,
                        "external_sources": ";".join(sorted(mapping.external_sources)),
                        "mapping_reason": "; ".join(mapping.reasons[:4]),
                    }
                )
                row_count += 1
    return row_count


def write_compound_structure_kegg_index(path: Path, compounds, structures, kegg_compounds, kegg_structure_indexes) -> int:
    """Write compound_id -> KEGG mappings recovered from exact InChIKey identity.

    This index is cheap to build and lets query-time step 2 truly prefer exact
    structure identity over name-based mapping.
    """

    row_count = 0
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "compound_id",
                "chebi_accession",
                "canonical_name",
                "canonical_exact_name",
                "canonical_compact_name",
                "canonical_singular_name",
                "canonical_stereo_stripped_name",
                "inchi_key",
                "kegg_compound_id",
                "kegg_primary_name",
                "mapping_score",
                "mapping_confidence_level",
                "mapping_method",
                "direct_kegg_xref",
                "has_structure_evidence",
                "used_pubchem_synonym",
                "best_alias",
                "best_alias_source",
                "best_variant",
                "evidence_count",
                "external_sources",
                "mapping_reason",
            ],
            delimiter="\t",
        )
        writer.writeheader()
        for compound_id in sorted(compounds, key=int):
            compound = compounds[compound_id]
            structure = structures.get(compound_id)
            if not structure or not structure.standard_inchi_key:
                continue
            inchi_key = normalize_inchi_key(structure.standard_inchi_key)
            hits = kegg_structure_indexes.get("by_inchi_key_full", {}).get(inchi_key, {})
            if not hits:
                continue
            canonical_variants = build_variants(compound.name)
            for kegg_compound_id, sources in sorted(hits.items()):
                if kegg_compound_id not in kegg_compounds:
                    continue
                source_set = set(sources)
                score = 0.99 if source_set >= {"ChEBI", "LIPID MAPS"} else 0.985 if "ChEBI" in source_set else 0.980
                writer.writerow(
                    {
                        "compound_id": compound_id,
                        "chebi_accession": compound.chebi_accession,
                        "canonical_name": compound.name,
                        "canonical_exact_name": canonical_variants["exact"],
                        "canonical_compact_name": canonical_variants["compact"],
                        "canonical_singular_name": canonical_variants["singular"],
                        "canonical_stereo_stripped_name": canonical_variants["stereo_stripped"],
                        "inchi_key": inchi_key,
                        "kegg_compound_id": kegg_compound_id,
                        "kegg_primary_name": kegg_compounds[kegg_compound_id].primary_name,
                        "mapping_score": f"{score:.3f}",
                        "mapping_confidence_level": "high",
                        "mapping_method": "inchi_key_exact",
                        "direct_kegg_xref": "false",
                        "has_structure_evidence": "true",
                        "used_pubchem_synonym": "false",
                        "best_alias": "",
                        "best_alias_source": "",
                        "best_variant": "inchi_key_exact",
                        "evidence_count": len(source_set),
                        "external_sources": ";".join(sorted(source_set)),
                        "mapping_reason": f"Exact InChIKey match to {kegg_compound_id} via {','.join(sorted(source_set))}",
                    }
                )
                row_count += 1
    return row_count


def write_compound_to_pathway_index(path: Path, kegg_to_pathways) -> int:
    """Write the KEGG compound -> map pathway index for query step 3."""

    row_count = 0
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["kegg_compound_id", "map_pathway_id"],
            delimiter="\t",
        )
        writer.writeheader()
        for kegg_compound_id in sorted(kegg_to_pathways):
            for map_pathway_id, _ath_pathway_id in sorted(kegg_to_pathways[kegg_compound_id]):
                writer.writerow(
                    {
                        "kegg_compound_id": kegg_compound_id,
                        "map_pathway_id": map_pathway_id,
                    }
                )
                row_count += 1
    return row_count


def write_map_to_ath_index(path: Path, map_to_ath) -> int:
    """Write the KEGG map -> ath conversion table."""

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["map_pathway_id", "ath_pathway_id"],
            delimiter="\t",
        )
        writer.writeheader()
        for map_pathway_id in sorted(map_to_ath):
            writer.writerow(
                {
                    "map_pathway_id": map_pathway_id,
                    "ath_pathway_id": map_to_ath[map_pathway_id],
                }
            )
    return len(map_to_ath)


def write_pathway_annotation_index(
    path: Path,
    map_pathways,
    ath_pathways,
    map_to_ath,
    pathway_categories,
    map_pathway_compound_counts,
    ath_gene_counts,
    reactome_pathways,
) -> int:
    """Write a unified pathway annotation table for both map and ath IDs."""

    row_count = 0
    map_ids = sorted(set(map_pathways) | set(map_to_ath))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "pathway_id",
                "pathway_target_type",
                "map_pathway_id",
                "ath_pathway_id",
                "pathway_name",
                "pathway_group",
                "pathway_category",
                "map_pathway_compound_count",
                "ath_gene_count",
                "reactome_matches",
            ],
            delimiter="\t",
        )
        writer.writeheader()
        for map_pathway_id in map_ids:
            group, category, _pathway_name = pathway_categories.get(map_pathway_id, ("", "", ""))
            map_name = map_pathways.get(map_pathway_id, map_pathway_id)
            writer.writerow(
                {
                    "pathway_id": map_pathway_id,
                    "pathway_target_type": "map_fallback",
                    "map_pathway_id": map_pathway_id,
                    "ath_pathway_id": "",
                    "pathway_name": map_name,
                    "pathway_group": group,
                    "pathway_category": category,
                    "map_pathway_compound_count": map_pathway_compound_counts.get(map_pathway_id, 0),
                    "ath_gene_count": 0,
                    "reactome_matches": ";".join(
                        f"{pathway_id}|{species}"
                        for pathway_id, species in reactome_pathways.get(normalize_name(map_name), [])[:3]
                    ),
                }
            )
            row_count += 1

            ath_pathway_id = map_to_ath.get(map_pathway_id, "")
            if not ath_pathway_id:
                continue
            ath_name = ath_pathways.get(ath_pathway_id, ath_pathway_id)
            writer.writerow(
                {
                    "pathway_id": ath_pathway_id,
                    "pathway_target_type": "ath",
                    "map_pathway_id": map_pathway_id,
                    "ath_pathway_id": ath_pathway_id,
                    "pathway_name": ath_name,
                    "pathway_group": group,
                    "pathway_category": category,
                    "map_pathway_compound_count": map_pathway_compound_counts.get(map_pathway_id, 0),
                    "ath_gene_count": ath_gene_counts.get(ath_pathway_id, 0),
                    "reactome_matches": ";".join(
                        f"{pathway_id}|{species}"
                        for pathway_id, species in reactome_pathways.get(normalize_name(ath_name), [])[:3]
                    ),
                }
            )
            row_count += 1
    return row_count


def write_plant_evidence_index(path: Path, compounds, compound_contexts, plantcyc_pathway_stats) -> int:
    """Write precomputed compound -> PMN pathway evidence.

    This table serves two roles:

    1. KEGG-backed ranking boost: if a KEGG pathway name matches an AraCyc or
       PlantCyc pathway name already seen for the same compound, query-side
       scoring adds a plant-specific support bonus.
    2. Direct PMN fallback: when KEGG has no usable compound/pathway mapping,
       query-side logic can still return plant pathways supported by AraCyc or
       PlantCyc for the resolved compound.
    """

    row_count = 0
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "compound_id",
                "chebi_accession",
                "canonical_name",
                "pathway_name_normalized",
                "plant_support_source",
                "plant_pathway_ids",
                "plant_support_examples",
                "plant_support_bonus",
                "plant_support_gene_count",
            ],
            delimiter="\t",
        )
        writer.writeheader()
        for compound_id in sorted(compounds, key=int):
            compound = compounds[compound_id]
            compound_context = compound_contexts[compound_id]

            for normalized_name, names in sorted(compound_context.aracyc_pathways.items()):
                pathway_stat = plantcyc_pathway_stats["AraCyc"].get(normalized_name, {})
                gene_count = len(pathway_stat.get("gene_ids", set()))
                writer.writerow(
                    {
                        "compound_id": compound_id,
                        "chebi_accession": compound.chebi_accession,
                        "canonical_name": compound.name,
                        "pathway_name_normalized": normalized_name,
                        "plant_support_source": "AraCyc",
                        "plant_pathway_ids": ";".join(sorted(pathway_stat.get("pathway_ids", set()))),
                        "plant_support_examples": ";".join(sorted(names)[:3]),
                        "plant_support_bonus": "0.060",
                        "plant_support_gene_count": gene_count,
                    }
                )
                row_count += 1

            for normalized_name, names in sorted(compound_context.plantcyc_pathways.items()):
                pathway_stat = plantcyc_pathway_stats["PlantCyc"].get(normalized_name, {})
                gene_count = len(pathway_stat.get("gene_ids", set()))
                writer.writerow(
                    {
                        "compound_id": compound_id,
                        "chebi_accession": compound.chebi_accession,
                        "canonical_name": compound.name,
                        "pathway_name_normalized": normalized_name,
                        "plant_support_source": "PlantCyc",
                        "plant_pathway_ids": ";".join(sorted(pathway_stat.get("pathway_ids", set()))),
                        "plant_support_examples": ";".join(sorted(names)[:3]),
                        "plant_support_bonus": "0.040",
                        "plant_support_gene_count": gene_count,
                    }
                )
                row_count += 1
    return row_count


def build_metadata(workdir: Path, output_dir: Path, counts: dict[str, int]) -> dict[str, object]:
    """Create a compact metadata/audit payload for the preprocessed indexes."""

    input_mtimes = {}
    for path in required_paths(workdir):
        input_mtimes[str(path.relative_to(workdir))] = datetime.fromtimestamp(path.stat().st_mtime, tz=UTC).isoformat()
    return {
        "version": VERSION,
        "generated_at": datetime.now(tz=UTC).isoformat(),
        "workdir": str(workdir),
        "output_dir": str(output_dir),
        "counts": counts,
        "inputs": input_mtimes,
        "notes": [
            "The query pipeline consumes only outputs/preprocessed/* and does not rescan refs/ during interactive or one-shot queries.",
            "Weak step-1 matches are guarded by formula compatibility, InChIKey connectivity prefixes, and chemistry-sensitive name semantics.",
            "Step 2 prefers exact InChIKey-based KEGG structure matches before falling back to name-based KEGG resolution.",
            "AraCyc and PlantCyc direct pathway support is exported so the query layer can fall back to PMN when KEGG coverage is missing.",
            "Step 8 similarity fallback and step 9 extra PlantCyc reranking are intentionally excluded from this first query-oriented refactor.",
        ],
    }


def main() -> None:
    """Build all preprocessed indexes required by the name-first query flow."""

    args = parse_args()
    workdir = Path(args.workdir).resolve()
    refs = workdir / "refs"
    output_dir = workdir / "outputs" / PREPROCESSED_DIRNAME
    output_dir.mkdir(parents=True, exist_ok=True)

    for path in required_paths(workdir):
        ensure_exists(path)

    print("Loading primary datasets...", flush=True)
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
    kegg_standard_indexes = build_standard_name_indexes(kegg_compounds)
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
    kegg_structure_indexes = build_kegg_structure_indexes(structures, xrefs, lipidmaps_records)
    target_pubchem_cids = set(plantcyc_pubchem_cids) | set(lipidmaps_pubchem_cids)
    for info in xrefs.values():
        target_pubchem_cids.update(info.pubchem_cids)
    pubchem_synonyms, pubchem_stats = load_pubchem_synonyms(refs / "pubchem_cid_synonym_filtered.gz", target_pubchem_cids)
    reactome_pathways = load_reactome_pathways(refs / "ReactomePathways.txt")

    print("Building per-compound contexts and KEGG selections...", flush=True)
    compound_contexts = {}
    selected_by_compound = {}
    mapping_status = Counter()
    sorted_compound_ids = sorted(compounds, key=int)
    for index, compound_id in enumerate(sorted_compound_ids, start=1):
        compound = compounds[compound_id]
        compound_context = build_compound_context(
            compound=compound,
            structure=structures.get(compound_id),
            xrefs=xrefs.get(compound_id, XrefInfo()),
            base_aliases=base_aliases.get(compound_id, []),
            plantcyc_records=plantcyc_records,
            plantcyc_indexes=plantcyc_indexes,
            lipidmaps_records=lipidmaps_records,
            lipidmaps_indexes=lipidmaps_indexes,
            pubchem_synonyms=pubchem_synonyms,
        )
        compound_contexts[compound_id] = compound_context
        ranked = build_candidate_mappings(
            compound=compound,
            structure=structures.get(compound_id),
            xrefs=xrefs.get(compound_id, XrefInfo()),
            context=compound_context,
            plantcyc_records=plantcyc_records,
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
            mapping_status["selected"] += 1
        elif ranked:
            mapping_status["ambiguous"] += 1
        else:
            mapping_status["unmapped"] += 1
        if index % PROGRESS_EVERY == 0 or index == len(sorted_compound_ids):
            print(
                f"  processed {index}/{len(sorted_compound_ids)} compounds "
                f"(selected={mapping_status['selected']}, ambiguous={mapping_status['ambiguous']}, "
                f"unmapped={mapping_status['unmapped']})",
                flush=True,
            )

    print("Writing preprocessed indexes...", flush=True)
    counts = {}
    counts["compounds_total"] = len(compounds)
    counts["comments_profile_keys"] = len(comments_profile)
    counts["name_normalization_index"] = write_name_normalization_index(
        output_dir / "name_normalization_index.tsv",
        compounds,
        compound_contexts,
    )
    counts["name_to_formula_index"] = write_name_to_formula_index(
        output_dir / "name_to_formula_index.tsv",
        aggregate_name_formula_rows(base_aliases, structures, plantcyc_records, lipidmaps_records),
    )
    counts["name_to_kegg_index"] = write_name_to_kegg_index(
        output_dir / "name_to_kegg_index.tsv",
        compounds,
        selected_by_compound,
    )
    counts["compound_structure_kegg_index"] = write_compound_structure_kegg_index(
        output_dir / "compound_structure_kegg_index.tsv",
        compounds,
        structures,
        kegg_compounds,
        kegg_structure_indexes,
    )
    counts["compound_to_pathway_index"] = write_compound_to_pathway_index(
        output_dir / "compound_to_pathway_index.tsv",
        kegg_to_pathways,
    )
    counts["map_to_ath_index"] = write_map_to_ath_index(
        output_dir / "map_to_ath_index.tsv",
        map_to_ath,
    )
    counts["pathway_annotation_index"] = write_pathway_annotation_index(
        output_dir / "pathway_annotation_index.tsv",
        map_pathways,
        ath_pathways,
        map_to_ath,
        pathway_categories,
        map_pathway_compound_counts,
        ath_gene_counts,
        reactome_pathways,
    )
    counts["plant_evidence_index"] = write_plant_evidence_index(
        output_dir / "plant_evidence_index.tsv",
        compounds,
        compound_contexts,
        plantcyc_pathway_stats,
    )
    counts["mapping_status_selected"] = mapping_status["selected"]
    counts["mapping_status_ambiguous"] = mapping_status["ambiguous"]
    counts["mapping_status_unmapped"] = mapping_status["unmapped"]
    counts["pubchem_target_cids"] = pubchem_stats.get("target_cids", 0)

    metadata = build_metadata(workdir, output_dir, counts)
    with (output_dir / "preprocess_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)

    print("Preprocess completed.", flush=True)
    print(f"- Output directory: {output_dir}", flush=True)
    for key in sorted(counts):
        print(f"- {key}: {counts[key]}", flush=True)


if __name__ == "__main__":
    main()
