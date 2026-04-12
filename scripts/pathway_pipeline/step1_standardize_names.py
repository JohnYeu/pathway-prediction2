"""Step 1: alias normalization and alias-table materialization.

This module is responsible for all name-side preparation before pathway matching.
It does four things in order:

1. Load the core ChEBI compound table and profile comments.tsv.
2. Load alias / structure / cross-reference sources from ChEBI and external
   support databases.
3. Expand each ChEBI compound into a full per-compound alias context.
4. Write the fully standardized alias audit table used downstream.

The output of this step is intentionally verbose. Later steps only need part of
the information, but keeping the full alias table makes debugging and manual
review much easier.
"""

from __future__ import annotations

import argparse
import csv
import gzip
from collections import Counter, defaultdict
import sys
from pathlib import Path
import zipfile

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from process_chebi_to_pathways_v2 import (
    ALLOWED_CHEBI_ALIAS_TYPES,
    EXTERNAL_NAME_MATCH_MAX_RECORDS,
    KEGG_ID_RE,
    MAX_EXTRA_SYNONYMS_PER_RECORD,
    MAX_PUBCHEM_SYNONYMS_PER_CID,
    PUBCHEM_CID_RE,
    AliasRecord,
    ChEBICompound,
    CompoundContext,
    LipidMapsRecord,
    PlantCycCompound,
    StructureInfo,
    XrefInfo,
    build_variants,
    formula_key,
    keep_pubchem_synonym,
    normalize_chebi_id,
    normalize_name,
    record_alias,
    split_multi_value,
)

from pathway_pipeline.cli_utils import build_context, build_parser, print_summary
from pathway_pipeline.context import PipelineContext


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
    """Build the initial alias table from ChEBI primary names and names.tsv."""

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
    """Create a compact-name -> formula-set lookup used by fuzzy validation."""

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


def _inchi_key_prefix(value: str) -> str:
    """Return the connectivity block of an InChIKey when available."""

    text = (value or "").strip().upper()
    if not text:
        return ""
    return text.split("-", 1)[0]


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
        elif prefix in {"LIGAND-CPD", "KEGG", "CPD"} and KEGG_ID_RE.fullmatch(value):
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
    """Collect how this compound touched external resources."""

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


def write_name_normalization_index(context: PipelineContext) -> int:
    """Write the canonical name + alias table consumed by query step 1."""

    row_count = 0
    with context.paths.name_normalization_index_path.open("w", newline="", encoding="utf-8") as handle:
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
        for compound_id in sorted(context.compounds, key=int):
            compound = context.compounds[compound_id]
            compound_context = context.compound_contexts[compound_id]
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


def aggregate_name_formula_rows(context: PipelineContext) -> dict[tuple[str, str], dict[str, object]]:
    """Aggregate normalized names to structure signatures for weak-match validation."""

    rows: dict[tuple[str, str], dict[str, object]] = {}

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
            prefix = _inchi_key_prefix(normalized_inchi)
            if prefix:
                entry["inchi_key_prefixes"].add(prefix)
        entry["evidence_sources"].add(source)

    for compound_id, aliases in context.base_aliases.items():
        structure = context.structures.get(compound_id)
        if not structure:
            continue
        for alias in aliases:
            add_name(alias.raw_name, structure.formula_key, structure.standard_inchi_key, "ChEBI")

    for record in context.plantcyc_records.values():
        add_name(record.common_name, record.formula_key, "", record.source_db)
        for synonym in sorted(record.synonyms):
            add_name(synonym, record.formula_key, "", record.source_db)

    for record in context.lipidmaps_records.values():
        add_name(record.common_name, record.formula_key, record.inchi_key, "LIPID MAPS")
        add_name(record.systematic_name, record.formula_key, record.inchi_key, "LIPID MAPS")
        for synonym in sorted(record.synonyms):
            add_name(synonym, record.formula_key, record.inchi_key, "LIPID MAPS")

    return rows


def write_name_to_formula_index(context: PipelineContext, rows: dict[tuple[str, str], dict[str, object]]) -> int:
    """Write the step-1 structure lookup used by weak query-time corrections."""

    with context.paths.name_to_formula_index_path.open("w", newline="", encoding="utf-8") as handle:
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
                    "inchi_key_prefixes": ";".join(sorted(entry["inchi_key_prefixes"])),
                    "inchi_key_prefix_count": len(entry["inchi_key_prefixes"]),
                    "inchi_key_fulls": ";".join(sorted(entry["inchi_key_fulls"])),
                    "inchi_key_full_count": len(entry["inchi_key_fulls"]),
                    "evidence_sources": ";".join(sorted(entry["evidence_sources"])),
                }
            )
    return len(rows)


def run(context: PipelineContext) -> PipelineContext:
    """Load alias sources and write the normalized alias table.

    The returned context contains:
    - core ChEBI compounds
    - base ChEBI aliases
    - structure / formula / cross-reference evidence
    - PlantCyc / AraCyc / LIPID MAPS / PubChem support resources
    - a per-compound CompoundContext with all merged aliases
    - the step-1 alias output TSV
    """

    paths = context.paths

    # Fail early if any required input is missing. Step 1 depends on nearly all
    # reference inputs because alias expansion already pulls in external naming
    # sources such as PlantCyc, LIPID MAPS, and PubChem.
    paths.ensure_required_inputs()

    # Load the starting ChEBI compound table. This is the master list of
    # compounds that the whole pipeline iterates over.
    context.compounds = load_compounds(paths.compounds_path)

    # comments.tsv is not used as an alias source, but we still profile it so
    # the final summary can explain why it was excluded.
    context.comments_profile = load_comments_profile(paths.comments_path)

    # Load the standard ChEBI names and approved ChEBI alias types from
    # names.tsv.gz. This is the primary alias source before external expansion.
    context.base_aliases = load_base_aliases(context.compounds, paths.chebi_names_path)

    # Load curated ChEBI cross-references such as KEGG and PubChem IDs. These
    # IDs are later used both for strong mapping evidence and for synonym pulls.
    context.xrefs = load_xrefs(paths.chebi_database_accession_path)

    # Load formula information first, because structure loading enriches each
    # structure record with the preferred formula / formula_key when available.
    context.formulas = load_formula_info(paths.chebi_chemical_data_path)
    context.structures = load_structures(paths.chebi_structures_path, context.formulas)

    # Load PlantCyc and AraCyc compound tables. These provide additional names,
    # pathway hints, and external identifiers that may strengthen a compound's
    # support context even before KEGG matching begins.
    (
        context.plantcyc_records,
        context.plantcyc_indexes,
        plantcyc_pubchem_cids,
    ) = load_plantcyc_compounds(
        [
            ("AraCyc", paths.aracyc_compounds_path),
            ("PlantCyc", paths.plantcyc_compounds_path),
        ]
    )

    # Load LIPID MAPS records. This is especially important for lipid-like
    # compounds where names alone are often too ambiguous.
    (
        context.lipidmaps_records,
        context.lipidmaps_indexes,
        lipidmaps_pubchem_cids,
    ) = load_lipidmaps_records(paths.lipidmaps_sdf_path)

    # Build a normalized name -> formula index spanning ChEBI and selected
    # external sources. Step 2 uses this to guard typo-like fuzzy corrections:
    # if two near-identical names have incompatible formulas, they should not be
    # auto-corrected into the same compound.
    context.name_formula_index = build_name_formula_index(
        context.base_aliases,
        context.structures,
        context.plantcyc_records,
        context.lipidmaps_records,
    )

    # Collect the set of PubChem CIDs that are relevant to the current corpus.
    # We only load synonyms for CIDs that we can actually reach through ChEBI,
    # PlantCyc, or LIPID MAPS, which keeps the synonym expansion bounded.
    target_pubchem_cids = set(plantcyc_pubchem_cids) | set(lipidmaps_pubchem_cids)
    for info in context.xrefs.values():
        target_pubchem_cids.update(info.pubchem_cids)
    context.pubchem_synonyms, context.pubchem_stats = load_pubchem_synonyms(
        paths.pubchem_synonyms_path,
        target_pubchem_cids,
    )

    # Step 1 writes the alias audit table directly, because this table is useful
    # on its own and does not need to wait for KEGG matching.
    alias_rows = 0
    compound_contexts = {}
    with paths.alias_output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
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
        writer.writeheader()

        # Build a full alias/support context for every ChEBI compound. This is
        # where base ChEBI aliases are merged with external names from
        # PlantCyc/AraCyc, LIPID MAPS, and PubChem.
        for compound_id in sorted(context.compounds, key=int):
            compound = context.compounds[compound_id]
            compound_context = build_compound_context(
                compound=compound,
                structure=context.structures.get(compound_id),
                xrefs=context.xrefs.get(compound_id, XrefInfo()),
                base_aliases=context.base_aliases.get(compound_id, []),
                plantcyc_records=context.plantcyc_records,
                plantcyc_indexes=context.plantcyc_indexes,
                lipidmaps_records=context.lipidmaps_records,
                lipidmaps_indexes=context.lipidmaps_indexes,
                pubchem_synonyms=context.pubchem_synonyms,
            )

            # Cache the merged context in memory so step 2 can reuse it without
            # having to rebuild aliases or re-scan external tables.
            compound_contexts[compound_id] = compound_context

            # Write one row per alias variant source. The output table keeps the
            # raw alias plus several normalized forms used in the matcher.
            for alias in sorted(compound_context.all_aliases, key=lambda item: (item.source_type, item.raw_name)):
                writer.writerow(
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

    # Persist the per-compound merged contexts and alias row count for later
    # steps and for the final JSON summary.
    context.compound_contexts = compound_contexts
    context.alias_rows = alias_rows
    context.preprocess_counts["compounds_total"] = len(context.compounds)
    context.preprocess_counts["comments_profile_keys"] = len(context.comments_profile)
    context.preprocess_counts["name_normalization_index"] = write_name_normalization_index(context)
    formula_rows = aggregate_name_formula_rows(context)
    context.preprocess_counts["name_to_formula_index"] = write_name_to_formula_index(context, formula_rows)
    context.preprocess_counts["pubchem_target_cids"] = context.pubchem_stats.get("target_cids", 0)
    context.add_note("Step 1 wrote the standardized alias table before downstream compound matching.")
    return context


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for standalone step-1 execution."""

    return build_parser(
        description="Run pathway pipeline step 1 only: alias normalization and alias-table writing.",
        default_output_tag="step1_cli",
    ).parse_args()


def main() -> None:
    """Run step 1 as a standalone terminal command."""

    args = parse_args()
    context = build_context(workdir=args.workdir, output_tag=args.output_tag)
    run(context)
    print_summary(
        "Step 1 completed.",
        [
            f"Alias table: {context.paths.alias_output_path}",
            f"Name normalization index: {context.paths.name_normalization_index_path}",
            f"Name-to-formula index: {context.paths.name_to_formula_index_path}",
            f"Compounds loaded: {len(context.compounds)}",
            f"Alias rows written: {context.alias_rows}",
            f"PubChem synonym CIDs loaded: {context.pubchem_stats.get('target_cids', 0)}",
        ],
    )


if __name__ == "__main__":
    main()
