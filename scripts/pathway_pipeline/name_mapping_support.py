"""Step 2: map normalized names to KEGG compound IDs.

This module owns the step-2 matching strategy. It takes the per-compound alias
contexts prepared in step 1, resolves them to KEGG compound IDs, and writes the
step-2 audit/index files consumed by later steps and query-time lookups.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from process_chebi_to_pathways_v2 import (
    ALIAS_SOURCE_WEIGHTS,
    CandidateMapping,
    CHAR_EDIT2_BASE_SCORE,
    Chem,
    ChEBICompound,
    CompoundContext,
    KEGG_ID_RE,
    KeggCompound,
    LipidMapsRecord,
    PLANT_BRIDGE_BASE_SCORES,
    XrefInfo,
    PlantCycCompound,
    PlantToKeggBridge,
    StructureInfo,
    VARIANT_BASE_SCORES,
    VARIANT_ORDER,
    build_variants,
    canonicalize_smiles,
    char_edit_distance_at_most_two,
    formula_sets_compatible,
    load_kegg_compounds,
    lookup_compact_fuzzy_candidates,
    normalize_inchi_key,
    rdkit_smiles_available,
    serialize_support_context,
)

from pathway_pipeline.cli_utils import build_context, build_parser, format_counter, print_summary
from pathway_pipeline.context import PipelineContext
from pathway_pipeline.step1_standardize_names import run as run_step1


PROGRESS_EVERY = 10000


def build_standard_name_indexes(kegg_compounds):
    """Materialize primary-name indexes from loaded KEGG rows.

    The KEGG loader already returns a full-name index and an alias-only index.
    For the refactored step flow we also build a dedicated standard-name index
    keyed only by primary_name variants. This keeps the "standard name first,
    alias second" rule explicit in the orchestration layer.
    """

    indexes = {
        "exact": defaultdict(set),
        "compact": defaultdict(set),
        "singular": defaultdict(set),
        "stereo_stripped": defaultdict(set),
    }

    # Each KEGG compound contributes its canonical label in several normalized
    # forms. Downstream matching can then try strict to relaxed variants.
    for kegg_id, kegg in kegg_compounds.items():
        indexes["exact"][kegg.primary_exact].add(kegg_id)
        indexes["compact"][kegg.primary_compact].add(kegg_id)
        indexes["singular"][kegg.primary_singular].add(kegg_id)
        indexes["stereo_stripped"][kegg.primary_stereo_stripped].add(kegg_id)
    return indexes


def bridge_confidence_label(score: float) -> str:
    """Bucket PMN bridge scores into user-facing confidence bands."""

    if score >= 0.93:
        return "high"
    if score >= 0.88:
        return "medium"
    return "low"


def build_kegg_structure_indexes(
    structures: dict[str, StructureInfo],
    xrefs: dict[str, XrefInfo],
    lipidmaps_records: dict[str, LipidMapsRecord],
) -> dict[str, dict[str, dict[str, set[str]]]]:
    """Build exact InChIKey/SMILES -> KEGG compound evidence for step 2."""

    by_inchi_key_full: defaultdict[str, defaultdict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    by_smiles_norm: defaultdict[str, defaultdict[str, set[str]]] = defaultdict(lambda: defaultdict(set))

    for compound_id, info in xrefs.items():
        structure = structures.get(compound_id)
        if not structure:
            continue
        inchi_key = normalize_inchi_key(structure.standard_inchi_key)
        smiles_norm = canonicalize_smiles(structure.smiles)
        for kegg_id in info.kegg_ids:
            if not KEGG_ID_RE.fullmatch(kegg_id):
                continue
            if inchi_key:
                by_inchi_key_full[inchi_key][kegg_id].add("ChEBI")
            if smiles_norm:
                by_smiles_norm[smiles_norm][kegg_id].add("ChEBI")

    for record in lipidmaps_records.values():
        inchi_key = normalize_inchi_key(record.inchi_key)
        smiles_norm = canonicalize_smiles(record.smiles)
        for kegg_id in record.kegg_ids:
            if not KEGG_ID_RE.fullmatch(kegg_id):
                continue
            if inchi_key:
                by_inchi_key_full[inchi_key][kegg_id].add("LIPID MAPS")
            if smiles_norm:
                by_smiles_norm[smiles_norm][kegg_id].add("LIPID MAPS")

    return {
        "by_inchi_key_full": {
            inchi_key: {kegg_id: set(sources) for kegg_id, sources in hits.items()}
            for inchi_key, hits in by_inchi_key_full.items()
        },
        "by_smiles_norm": {
            smiles_norm: {kegg_id: set(sources) for kegg_id, sources in hits.items()}
            for smiles_norm, hits in by_smiles_norm.items()
        },
    }


def build_plantcyc_compound_index(
    plantcyc_records: dict[str, PlantCycCompound],
) -> list[dict[str, object]]:
    """Flatten PMN compound rows into an audit/index table for step 2."""

    rows: list[dict[str, object]] = []
    for record_id in sorted(plantcyc_records):
        record = plantcyc_records[record_id]
        variants = build_variants(record.common_name)
        rows.append(
            {
                "plant_db": record.source_db,
                "plant_compound_id": record.compound_id,
                "canonical_name": record.common_name,
                "name_norm_exact": variants["exact"],
                "name_norm_compact": variants["compact"],
                "name_norm_singular": variants["singular"],
                "synonym_list": ";".join(sorted(record.synonyms)),
                "formula": record.formula,
                "formula_key": record.formula_key,
                "smiles_raw": record.smiles,
                "smiles_norm": canonicalize_smiles(record.smiles),
                "inchi_key_full": "",
                "chebi_ids": ";".join(sorted(record.chebi_ids)),
                "pubchem_cids": ";".join(sorted(record.pubchem_cids)),
                "kegg_ids": ";".join(sorted(record.kegg_ids)),
                "hmdb_ids": ";".join(sorted(record.hmdb_ids)),
                "pathway_count": len(record.pathways),
                "pathway_examples": ";".join(sorted(record.pathways)[:3]),
                "arabidopsis_supported": str(record.source_db == "AraCyc").lower(),
            }
        )
    return rows


def build_kegg_structure_index(
    structures: dict[str, StructureInfo],
    xrefs: dict[str, XrefInfo],
    lipidmaps_records: dict[str, LipidMapsRecord],
) -> list[dict[str, object]]:
    """Flatten the KEGG structure bridge table used by PMN structure rescue."""

    rows: list[dict[str, object]] = []
    indexes = build_kegg_structure_indexes(structures, xrefs, lipidmaps_records)
    merged: dict[tuple[str, str, str], set[str]] = {}

    for inchi_key, hits in indexes.get("by_inchi_key_full", {}).items():
        for kegg_cid, sources in hits.items():
            key = (kegg_cid, inchi_key, "")
            merged.setdefault(key, set()).update(sources)

    for smiles_norm, hits in indexes.get("by_smiles_norm", {}).items():
        for kegg_cid, sources in hits.items():
            matched_key = None
            for key in merged:
                if key[0] == kegg_cid and key[2] == smiles_norm:
                    matched_key = key
                    break
            if matched_key is None:
                matched_key = (kegg_cid, "", smiles_norm)
            merged.setdefault(matched_key, set()).update(sources)

    for (kegg_cid, inchi_key, smiles_norm), sources in sorted(merged.items()):
        rows.append(
            {
                "kegg_cid": kegg_cid,
                "inchi_key_full": inchi_key,
                "smiles_norm": smiles_norm,
                "source_dbs": ";".join(sorted(sources)),
                "source_count": len(sources),
            }
        )
    return rows


def build_plant_to_kegg_bridge(
    *,
    compound: ChEBICompound,
    context: CompoundContext,
    plantcyc_records: dict[str, PlantCycCompound],
    chebi_to_kegg_index: dict[str, set[str]],
    pubchem_to_kegg_index: dict[str, set[str]],
    kegg_structure_indexes: dict[str, dict[str, dict[str, set[str]]]],
) -> list[PlantToKeggBridge]:
    """Bridge matched PMN records back to KEGG using curated and structural evidence."""

    bridges: dict[tuple[str, str], dict[str, object]] = {}

    for record_id, matched_methods in sorted(context.matched_plantcyc_methods.items()):
        record = plantcyc_records[record_id]
        arabidopsis_supported = record.source_db == "AraCyc"

        def add_bridge(
            *,
            kegg_cid: str,
            bridge_method: str,
            supporting_ids: list[str],
            has_structure_evidence: bool,
            bridge_reason: str,
        ) -> None:
            if not KEGG_ID_RE.fullmatch(kegg_cid):
                return
            key = (record.record_id, kegg_cid)
            base_score = PLANT_BRIDGE_BASE_SCORES[(bridge_method, record.source_db)]
            extra_sources = 0
            if bridge_method != "plant_direct_kegg_xref" and record.kegg_ids:
                extra_sources += 1
            if bridge_method != "plant_via_chebi" and record.chebi_ids:
                extra_sources += 1
            if bridge_method != "plant_via_pubchem" and record.pubchem_cids:
                extra_sources += 1
            bridge_score = min(base_score + min(extra_sources, 3) * 0.01, base_score + 0.03)
            existing = bridges.get(key)
            if existing and float(existing["bridge_score"]) >= bridge_score:
                existing["bridge_record_ids"].add(record.record_id)
                existing["supporting_ids"].update(supporting_ids)
                existing["bridge_reason"].add(bridge_reason)
                return
            bridges[key] = {
                "compound_id": compound.compound_id,
                "chebi_accession": compound.chebi_accession,
                "canonical_name": compound.name,
                "plant_db": record.source_db,
                "plant_compound_id": record.compound_id,
                "kegg_cid": kegg_cid,
                "bridge_method": bridge_method,
                "bridge_score": bridge_score,
                "bridge_record_ids": {record.record_id},
                "supporting_ids": set(supporting_ids),
                "arabidopsis_supported": arabidopsis_supported,
                "has_structure_evidence": has_structure_evidence,
                "bridge_reason": {bridge_reason},
            }

        for kegg_cid in sorted(record.kegg_ids):
            add_bridge(
                kegg_cid=kegg_cid,
                bridge_method="plant_direct_kegg_xref",
                supporting_ids=[f"KEGG:{kegg_cid}"],
                has_structure_evidence=False,
                bridge_reason=f"{record.source_db} record {record.compound_id} carries direct KEGG link {kegg_cid}",
            )

        for chebi_id in sorted(record.chebi_ids):
            for kegg_cid in sorted(chebi_to_kegg_index.get(chebi_id, set())):
                add_bridge(
                    kegg_cid=kegg_cid,
                    bridge_method="plant_via_chebi",
                    supporting_ids=[chebi_id],
                    has_structure_evidence=False,
                    bridge_reason=f"{record.source_db} record {record.compound_id} bridges to {kegg_cid} via {chebi_id}",
                )

        record_inchi_key = ""
        if record.smiles and Chem is not None:
            try:
                mol = Chem.MolFromSmiles(record.smiles)
                if mol is not None:
                    record_inchi_key = normalize_inchi_key(Chem.MolToInchiKey(mol))
            except Exception:
                record_inchi_key = ""
        if record_inchi_key:
            for kegg_cid, sources in sorted(kegg_structure_indexes.get("by_inchi_key_full", {}).get(record_inchi_key, {}).items()):
                add_bridge(
                    kegg_cid=kegg_cid,
                    bridge_method="plant_inchikey_exact",
                    supporting_ids=[f"InChIKey:{record_inchi_key}"],
                    has_structure_evidence=True,
                    bridge_reason=f"{record.source_db} record {record.compound_id} exact InChIKey matches {kegg_cid} via {','.join(sorted(sources))}",
                )

        for cid in sorted(record.pubchem_cids):
            for kegg_cid in sorted(pubchem_to_kegg_index.get(cid, set())):
                add_bridge(
                    kegg_cid=kegg_cid,
                    bridge_method="plant_via_pubchem",
                    supporting_ids=[f"PUBCHEM:{cid}"],
                    has_structure_evidence=False,
                    bridge_reason=f"{record.source_db} record {record.compound_id} bridges to {kegg_cid} via PUBCHEM:{cid}",
                )

        smiles_norm = canonicalize_smiles(record.smiles)
        if smiles_norm:
            for kegg_cid, sources in sorted(kegg_structure_indexes.get("by_smiles_norm", {}).get(smiles_norm, {}).items()):
                add_bridge(
                    kegg_cid=kegg_cid,
                    bridge_method="plant_smiles_exact",
                    supporting_ids=[f"SMILES:{smiles_norm}"],
                    has_structure_evidence=True,
                    bridge_reason=f"{record.source_db} record {record.compound_id} canonical SMILES matches {kegg_cid} via {','.join(sorted(sources))}",
                )

    rows: list[PlantToKeggBridge] = []
    for (_record_id, _kegg_cid), row in sorted(bridges.items(), key=lambda item: (item[0][1], -item[1]["bridge_score"], item[0][0])):
        rows.append(
            PlantToKeggBridge(
                compound_id=row["compound_id"],
                chebi_accession=row["chebi_accession"],
                canonical_name=row["canonical_name"],
                plant_db=row["plant_db"],
                plant_compound_id=row["plant_compound_id"],
                kegg_cid=row["kegg_cid"],
                bridge_method=row["bridge_method"],
                bridge_score=float(row["bridge_score"]),
                bridge_confidence=bridge_confidence_label(float(row["bridge_score"])),
                bridge_record_ids=tuple(sorted(row["bridge_record_ids"])),
                supporting_ids=tuple(sorted(row["supporting_ids"])),
                arabidopsis_supported=bool(row["arabidopsis_supported"]),
                has_structure_evidence=bool(row["has_structure_evidence"]),
                bridge_reason="; ".join(sorted(row["bridge_reason"])),
            )
        )
    return rows


def reason_summary(candidate: CandidateMapping) -> str:
    """Collapse the most relevant evidence strings into a short explanation."""

    return "; ".join(candidate.reasons[:4])


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
    kegg_structure_indexes: dict[str, dict[str, dict[str, set[str]]]],
) -> list[CandidateMapping]:
    """Build and score all KEGG candidates for one ChEBI compound."""

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

    if structure and structure.standard_inchi_key:
        inchi_key = normalize_inchi_key(structure.standard_inchi_key)
        for kegg_id, sources in sorted(kegg_structure_indexes.get("by_inchi_key_full", {}).get(inchi_key, {}).items()):
            if kegg_id not in kegg_compounds:
                continue
            for source_label in sorted(sources):
                source_score = 0.985 if source_label == "ChEBI" else 0.980
                candidate_for(kegg_id).add_evidence(
                    score=source_score,
                    method="inchi_key_exact",
                    reason=f"Exact InChIKey match links {compound.chebi_accession} to {kegg_id} via {source_label}",
                    external_source=source_label,
                    has_structure_evidence=True,
                )

    for bridge in plant_bridge_rows:
        if bridge.kegg_cid not in kegg_compounds:
            continue
        candidate_for(bridge.kegg_cid).add_evidence(
            score=bridge.bridge_score,
            method=bridge.bridge_method,
            reason=bridge.bridge_reason,
            external_source=bridge.plant_db,
            has_structure_evidence=bridge.has_structure_evidence,
            plant_bridge_method=bridge.bridge_method,
            plant_bridge_source=bridge.plant_db,
            arabidopsis_supported=bridge.arabidopsis_supported,
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
    """Choose final mappings after evidence aggregation and ranking."""

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
    if "inchi_key_exact" in mapping.methods:
        return "inchi_key_exact"
    for method in (
        "plant_direct_kegg_xref",
        "plant_via_chebi",
        "plant_inchikey_exact",
        "plant_via_pubchem",
        "plant_smiles_exact",
    ):
        if method in mapping.methods:
            return method
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


def write_name_to_kegg_index(context: PipelineContext) -> int:
    """Write the canonical-name -> KEGG mapping index used by query step 2."""

    row_count = 0
    with context.paths.name_to_kegg_index_path.open("w", newline="", encoding="utf-8") as handle:
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
                "bridge_method",
                "bridge_confidence",
                "bridge_source_db",
                "arabidopsis_supported",
                "mapping_reason",
            ],
            delimiter="\t",
        )
        writer.writeheader()
        for compound_id in sorted(context.compounds, key=int):
            compound = context.compounds[compound_id]
            canonical_variants = build_variants(compound.name)
            for mapping in context.selected_by_compound.get(compound_id, []):
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
                        "bridge_method": next(iter(sorted(mapping.plant_bridge_methods)), ""),
                        "bridge_confidence": mapping_confidence_label(mapping.final_score)
                        if mapping.plant_bridge_methods
                        else "",
                        "bridge_source_db": next(iter(sorted(mapping.plant_bridge_sources)), ""),
                        "arabidopsis_supported": str(mapping.arabidopsis_supported).lower(),
                        "mapping_reason": "; ".join(mapping.reasons[:4]),
                    }
                )
                row_count += 1
    return row_count


def write_compound_structure_kegg_index(context: PipelineContext) -> int:
    """Write exact InChIKey-based structure mappings for structure-first step 2."""

    row_count = 0
    with context.paths.compound_structure_kegg_index_path.open("w", newline="", encoding="utf-8") as handle:
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
        for compound_id in sorted(context.compounds, key=int):
            compound = context.compounds[compound_id]
            structure = context.structures.get(compound_id)
            if not structure or not structure.standard_inchi_key:
                continue
            inchi_key = normalize_inchi_key(structure.standard_inchi_key)
            hits = context.kegg_structure_indexes.get("by_inchi_key_full", {}).get(inchi_key, {})
            if not hits:
                continue
            canonical_variants = build_variants(compound.name)
            for kegg_compound_id, sources in sorted(hits.items()):
                if kegg_compound_id not in context.kegg_compounds:
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
                        "kegg_primary_name": context.kegg_compounds[kegg_compound_id].primary_name,
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


def build_bridge_lookup_indexes(context: PipelineContext) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    """Build lightweight ChEBI/PubChem -> KEGG bridge lookup tables."""

    chebi_to_kegg = defaultdict(set)
    pubchem_to_kegg = defaultdict(set)
    for compound_id, compound in context.compounds.items():
        info = context.xrefs.get(compound_id, XrefInfo())
        valid_kegg_ids = {kegg_id for kegg_id in info.kegg_ids if kegg_id in context.kegg_compounds}
        if not valid_kegg_ids:
            continue
        chebi_to_kegg[compound.chebi_accession].update(valid_kegg_ids)
        for cid in info.pubchem_cids:
            pubchem_to_kegg[cid].update(valid_kegg_ids)
    for record in context.lipidmaps_records.values():
        valid_kegg_ids = {kegg_id for kegg_id in record.kegg_ids if kegg_id in context.kegg_compounds}
        if not valid_kegg_ids:
            continue
        for cid in record.pubchem_cids:
            pubchem_to_kegg[cid].update(valid_kegg_ids)
    return (
        {key: set(values) for key, values in chebi_to_kegg.items()},
        {key: set(values) for key, values in pubchem_to_kegg.items()},
    )


def write_plantcyc_compound_index(context: PipelineContext) -> int:
    """Write the PMN plant-first recall pool used for bridge auditing."""

    rows = build_plantcyc_compound_index(context.plantcyc_records)
    with context.paths.plantcyc_compound_index_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "plant_db",
                "plant_compound_id",
                "canonical_name",
                "name_norm_exact",
                "name_norm_compact",
                "name_norm_singular",
                "synonym_list",
                "formula",
                "formula_key",
                "smiles_raw",
                "smiles_norm",
                "inchi_key_full",
                "chebi_ids",
                "pubchem_cids",
                "kegg_ids",
                "hmdb_ids",
                "pathway_count",
                "pathway_examples",
                "arabidopsis_supported",
            ],
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def write_kegg_structure_index(context: PipelineContext) -> int:
    """Write the KEGG-side structure bridge table used by PMN rescue."""

    rows = build_kegg_structure_index(context.structures, context.xrefs, context.lipidmaps_records)
    with context.paths.kegg_structure_index_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["kegg_cid", "inchi_key_full", "smiles_norm", "source_dbs", "source_count"],
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def write_plant_to_kegg_bridge_index(context: PipelineContext) -> int:
    """Write precomputed PMN -> KEGG bridge rows for query-time reuse."""

    rows = [
        bridge
        for compound_id in sorted(context.plant_to_kegg_bridge_rows_by_compound, key=int)
        for bridge in context.plant_to_kegg_bridge_rows_by_compound[compound_id]
    ]
    with context.paths.plant_to_kegg_bridge_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "compound_id",
                "chebi_accession",
                "canonical_name",
                "plant_db",
                "plant_compound_id",
                "kegg_cid",
                "kegg_primary_name",
                "bridge_method",
                "bridge_score",
                "bridge_confidence",
                "bridge_record_ids",
                "supporting_ids",
                "arabidopsis_supported",
                "has_structure_evidence",
                "bridge_reason",
            ],
            delimiter="\t",
        )
        writer.writeheader()
        for bridge in rows:
            writer.writerow(
                {
                    "compound_id": bridge.compound_id,
                    "chebi_accession": bridge.chebi_accession,
                    "canonical_name": bridge.canonical_name,
                    "plant_db": bridge.plant_db,
                    "plant_compound_id": bridge.plant_compound_id,
                    "kegg_cid": bridge.kegg_cid,
                    "kegg_primary_name": context.kegg_compounds.get(bridge.kegg_cid).primary_name
                    if bridge.kegg_cid in context.kegg_compounds
                    else "",
                    "bridge_method": bridge.bridge_method,
                    "bridge_score": f"{bridge.bridge_score:.3f}",
                    "bridge_confidence": bridge.bridge_confidence,
                    "bridge_record_ids": ";".join(bridge.bridge_record_ids),
                    "supporting_ids": ";".join(bridge.supporting_ids),
                    "arabidopsis_supported": str(bridge.arabidopsis_supported).lower(),
                    "has_structure_evidence": str(bridge.has_structure_evidence).lower(),
                    "bridge_reason": bridge.bridge_reason,
                }
            )
    return len(rows)


def run(context: PipelineContext) -> PipelineContext:
    """Build KEGG candidates, select final mappings, and write mapping tables.

    Outputs written here:
    - mapping summary table for every ChEBI compound
    - selected KEGG mapping table for trusted final mappings

    State captured for later steps:
    - full ranked candidate list per compound
    - final selected mappings per compound
    - compact support context retained only for selected compounds
    """

    # Load KEGG compound names and the supporting indexes used by the matching
    # logic. The v2 loader returns several specialized indexes; in this
    # refactored step we keep only the ones actually needed for the current
    # matching policy.
    (
        context.kegg_compounds,
        _all_indexes,
        _token_exact_indexes,
        _token_delete_indexes,
        _all_compact_delete_index,
        context.kegg_alias_indexes,
        context.kegg_primary_compact_delete_index,
    ) = load_kegg_compounds(context.paths.kegg_compound_list_path)
    context.kegg_structure_indexes = build_kegg_structure_indexes(
        context.structures,
        context.xrefs,
        context.lipidmaps_records,
    )

    # Build an explicit primary-name-only index so the matching order remains:
    # standard KEGG name -> KEGG alias -> guarded fuzzy correction.
    context.kegg_standard_indexes = build_standard_name_indexes(context.kegg_compounds)

    # These per-compound structures are both written to disk and carried forward
    # in memory for pathway expansion.
    selected_by_compound = {}
    ranked_candidates_by_compound = {}
    support_contexts = {}
    plant_to_kegg_bridge_rows_by_compound = {}
    mapping_status = defaultdict(int)
    chebi_to_kegg_index, pubchem_to_kegg_index = build_bridge_lookup_indexes(context)
    if not rdkit_smiles_available():
        context.add_note("RDKit is not available; Step 2 disabled plant_smiles_exact and PMN InChIKey derivation from SMILES.")

    # Step 2 produces two different audit views:
    # 1. a summary row for every ChEBI compound
    # 2. one row per trusted selected KEGG mapping
    with context.paths.mapping_summary_path.open("w", newline="", encoding="utf-8") as mapping_handle, context.paths.mapping_selected_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as selected_handle:
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
                "bridge_method",
                "bridge_confidence",
                "bridge_source_db",
                "arabidopsis_supported",
                "mapping_reason",
            ],
            delimiter="\t",
        )
        selected_writer.writeheader()

        # Process compounds one by one so the step output remains deterministic
        # and easy to diff across runs.
        total = len(context.compounds)
        for index, compound_id in enumerate(sorted(context.compounds, key=int), start=1):
            compound = context.compounds[compound_id]
            compound_context = context.compound_contexts[compound_id]
            plant_bridge_rows = build_plant_to_kegg_bridge(
                compound=compound,
                context=compound_context,
                plantcyc_records=context.plantcyc_records,
                chebi_to_kegg_index=chebi_to_kegg_index,
                pubchem_to_kegg_index=pubchem_to_kegg_index,
                kegg_structure_indexes=context.kegg_structure_indexes,
            )
            plant_to_kegg_bridge_rows_by_compound[compound_id] = plant_bridge_rows

            # Ask the shared v2 matching code to gather every plausible KEGG
            # candidate, accumulate evidence, and score the candidates.
            ranked = build_candidate_mappings(
                compound=compound,
                structure=context.structures.get(compound_id),
                xrefs=context.xrefs.get(compound_id, XrefInfo()),
                context=compound_context,
                plant_bridge_rows=plant_bridge_rows,
                lipidmaps_records=context.lipidmaps_records,
                kegg_compounds=context.kegg_compounds,
                kegg_standard_indexes=context.kegg_standard_indexes,
                kegg_alias_indexes=context.kegg_alias_indexes,
                kegg_primary_compact_delete_index=context.kegg_primary_compact_delete_index,
                name_formula_index=context.name_formula_index,
                kegg_structure_indexes=context.kegg_structure_indexes,
            )
            ranked_candidates_by_compound[compound_id] = ranked

            # Apply the conservative selection policy: direct curated mappings
            # are always kept; otherwise only clearly dominant high-scoring
            # candidates survive.
            selected = select_candidates(ranked)
            selected_by_compound[compound_id] = selected

            # Track whether the compound mapped cleanly, had unresolved
            # competition, or had no credible KEGG candidate at all.
            if selected:
                selection_status = "selected"
                support_contexts[compound_id] = serialize_support_context(compound_context)
            elif ranked:
                selection_status = "ambiguous"
            elif compound_context.matched_plantcyc_methods:
                selection_status = "no_kegg_bridge"
            else:
                selection_status = "unmapped"
            mapping_status[selection_status] += 1

            # The summary table always records the top candidate, even if it was
            # ultimately rejected, so users can inspect near-misses manually.
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
                    "alias_count": len(compound_context.all_aliases),
                    "matched_pubchem_cids": ";".join(sorted(compound_context.pubchem_cids)),
                    "mapping_reason": reason_summary(top) if top else "",
                }
            )

            # The selected table contains only the final trusted mappings and
            # exposes the strongest supporting evidence used to keep them.
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
                        "bridge_method": next(iter(sorted(mapping.plant_bridge_methods)), ""),
                        "bridge_confidence": mapping_confidence_label(mapping.final_score)
                        if mapping.plant_bridge_methods
                        else "",
                        "bridge_source_db": next(iter(sorted(mapping.plant_bridge_sources)), ""),
                        "arabidopsis_supported": str(mapping.arabidopsis_supported).lower(),
                        "mapping_reason": reason_summary(mapping),
                    }
                )

            if index % PROGRESS_EVERY == 0 or index == total:
                print(
                    f"  processed {index}/{total} compounds "
                    f"(selected={mapping_status['selected']}, ambiguous={mapping_status['ambiguous']}, "
                    f"no_kegg_bridge={mapping_status['no_kegg_bridge']}, "
                    f"unmapped={mapping_status['unmapped']})",
                    flush=True,
                )

    # Persist both the full ranked candidate space and the final selections for
    # downstream pathway expansion, ranking, and reporting.
    context.ranked_candidates_by_compound = ranked_candidates_by_compound
    context.selected_by_compound = selected_by_compound
    context.support_contexts = support_contexts
    context.plant_to_kegg_bridge_rows_by_compound = plant_to_kegg_bridge_rows_by_compound
    context.mapping_status.update(mapping_status)
    context.preprocess_counts["name_to_kegg_index"] = write_name_to_kegg_index(context)
    context.preprocess_counts["compound_structure_kegg_index"] = write_compound_structure_kegg_index(context)
    context.preprocess_counts["plantcyc_compound_index"] = write_plantcyc_compound_index(context)
    context.preprocess_counts["kegg_structure_index"] = write_kegg_structure_index(context)
    context.preprocess_counts["plant_to_kegg_bridge"] = write_plant_to_kegg_bridge_index(context)
    context.preprocess_counts["mapping_status_selected"] = mapping_status["selected"]
    context.preprocess_counts["mapping_status_ambiguous"] = mapping_status["ambiguous"]
    context.preprocess_counts["mapping_status_no_kegg_bridge"] = mapping_status["no_kegg_bridge"]
    context.preprocess_counts["mapping_status_unmapped"] = mapping_status["unmapped"]
    return context


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for standalone step-2 execution."""

    return build_parser(
        description="Run pathway pipeline steps 1-2: alias normalization plus KEGG compound mapping.",
        default_output_tag="step2_cli",
    ).parse_args()


def main() -> None:
    """Run steps 1-2 and write the step-2 outputs."""

    args = parse_args()
    context = build_context(workdir=args.workdir, output_tag=args.output_tag)
    run_step1(context)
    run(context)
    print_summary(
        "Step 2 completed.",
        [
            f"Alias table: {context.paths.alias_output_path}",
            f"Mapping summary: {context.paths.mapping_summary_path}",
            f"Selected mappings: {context.paths.mapping_selected_path}",
            f"Name-to-KEGG index: {context.paths.name_to_kegg_index_path}",
            f"Structure-to-KEGG index: {context.paths.compound_structure_kegg_index_path}",
            f"PlantCyc compound index: {context.paths.plantcyc_compound_index_path}",
            f"KEGG structure index: {context.paths.kegg_structure_index_path}",
            f"Plant-to-KEGG bridge: {context.paths.plant_to_kegg_bridge_path}",
            f"Mapping status counts: {format_counter(context.mapping_status)}",
        ],
    )


if __name__ == "__main__":
    main()
