"""Step 2 (AraCyc-first): map ChEBI compounds to AraCyc/PlantCyc compounds.

Matching hierarchy (highest priority first):
  1. Direct ChEBI xref in AraCyc Links field  (score 1.00)
  2. KEGG xref bridge via AraCyc LIGAND-CPD    (score 0.95)
  3. InChIKey exact match from SMILES           (score 0.90)
  3b. Tanimoto similarity >= 0.85               (score 0.85)
  4. Name matching (exact/compact/singular)      (score 0.70-0.80)
  5. PlantCyc fallback (same tiers, lower base)  (score * 0.85)
"""

from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from process_chebi_to_pathways_v2 import (
    ChEBICompound,
    PlantCycCompound,
    StructureInfo,
    XrefInfo,
    build_variants,
    canonicalize_smiles,
    normalize_inchi_key,
    normalize_name,
    rdkit_smiles_available,
)

from pathway_pipeline.context import AraCycCompoundMatch, PipelineContext

# ---------------------------------------------------------------------------
# Cofactor names (used to flag cofactor-like matches early)
# ---------------------------------------------------------------------------

COFACTOR_NAME_TOKENS = {
    "atp", "adp", "amp", "nadh", "nadph", "nadp", "nad", "water",
    "oxygen", "coenzyme a", "phosphate", "pyrophosphate", "carbon dioxide",
    "proton", "h+", "h2o", "o2", "co2",
}


# ---------------------------------------------------------------------------
# Matching logic
# ---------------------------------------------------------------------------


def _build_chebi_xref_index(
    records: dict[str, PlantCycCompound],
) -> dict[str, set[str]]:
    """Map ChEBI IDs -> set of PlantCycCompound record_ids that reference them."""
    idx: dict[str, set[str]] = defaultdict(set)
    for record_id, rec in records.items():
        for chebi_id in rec.chebi_ids:
            idx[chebi_id].add(record_id)
    return dict(idx)


def _build_kegg_xref_index(
    records: dict[str, PlantCycCompound],
) -> dict[str, set[str]]:
    """Map KEGG CIDs -> set of record_ids."""
    idx: dict[str, set[str]] = defaultdict(set)
    for record_id, rec in records.items():
        for kegg_id in rec.kegg_ids:
            idx[kegg_id].add(record_id)
    return dict(idx)


def _build_inchikey_index(
    records: dict[str, PlantCycCompound],
) -> dict[str, set[str]]:
    """Map InChIKey (from SMILES) -> set of record_ids."""
    idx: dict[str, set[str]] = defaultdict(set)
    try:
        from rdkit import Chem
        from rdkit.Chem.inchi import MolToInchi, InchiToInchiKey
    except ImportError:
        return dict(idx)
    for record_id, rec in records.items():
        if not rec.smiles:
            continue
        try:
            mol = Chem.MolFromSmiles(rec.smiles)
            if mol is None:
                continue
            inchi = MolToInchi(mol)
            if inchi is None:
                continue
            inchikey = InchiToInchiKey(inchi)
            if inchikey:
                idx[inchikey].add(record_id)
        except Exception:
            continue
    return dict(idx)


def _build_morgan_fp_index(
    records: dict[str, PlantCycCompound],
) -> dict[str, object]:
    """Build Morgan fingerprint index for Tanimoto similarity search.

    Returns dict mapping record_id -> fingerprint object.
    """
    fps: dict[str, object] = {}
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
    except ImportError:
        return fps
    for record_id, rec in records.items():
        if not rec.smiles:
            continue
        try:
            mol = Chem.MolFromSmiles(rec.smiles)
            if mol is None:
                continue
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
            fps[record_id] = fp
        except Exception:
            continue
    return fps


def _build_name_index(
    records: dict[str, PlantCycCompound],
) -> dict[str, dict[str, set[str]]]:
    """Build name variant indexes for AraCyc/PlantCyc records.

    Returns dict with keys: 'exact', 'compact', 'singular', 'stereo_stripped'
    each mapping normalized name -> set of record_ids.
    """
    indexes: dict[str, dict[str, set[str]]] = {
        "exact": defaultdict(set),
        "compact": defaultdict(set),
        "singular": defaultdict(set),
        "stereo_stripped": defaultdict(set),
    }
    for record_id, rec in records.items():
        all_names = [rec.common_name] + sorted(rec.synonyms)
        for name in all_names:
            variants = build_variants(name)
            for vtype, vtext in variants.items():
                if vtext and vtype in indexes:
                    indexes[vtype][vtext].add(record_id)
    return {k: dict(v) for k, v in indexes.items()}


def _reference_compound_key(record: PlantCycCompound) -> str:
    """Build a stable Arabidopsis reference key for one PMN compound."""

    if record.chebi_ids:
        return f"chebi:{sorted(record.chebi_ids)[0]}"
    if record.kegg_ids:
        return f"kegg:{sorted(record.kegg_ids)[0]}"
    canonical_smiles = canonicalize_smiles(record.smiles)
    if canonical_smiles:
        return f"smiles:{canonical_smiles}"
    normalized_name = normalize_name(record.common_name)
    if normalized_name:
        return f"name:{normalized_name}"
    return f"record:{record.record_id}"


def _build_match(
    record: PlantCycCompound,
    *,
    method: str,
    score: float,
    chebi_xref_direct: bool,
    structure_validated: bool,
    discount: float = 1.0,
) -> AraCycCompoundMatch:
    """Build one AraCyc/PlantCyc match row."""
    return AraCycCompoundMatch(
        aracyc_compound_id=record.compound_id,
        aracyc_common_name=record.common_name,
        source_db=record.source_db,
        match_method=method,
        match_score=round(score * discount, 4),
        chebi_xref_direct=chebi_xref_direct,
        structure_validated=structure_validated,
        pathways=tuple(sorted(record.pathways)),
        smiles=record.smiles,
        ec_numbers=(),
        plant_record_id=record.record_id,
        reference_compound_key=_reference_compound_key(record),
    )


def _chebi_inchikey(compound_id: str, structures: dict[str, StructureInfo]) -> str:
    """Get InChIKey for a ChEBI compound from preloaded structures."""
    si = structures.get(compound_id)
    if si and si.standard_inchi_key:
        return normalize_inchi_key(si.standard_inchi_key)
    return ""


def _chebi_morgan_fp(compound_id: str, structures: dict[str, StructureInfo]):
    """Get Morgan fingerprint for a ChEBI compound."""
    si = structures.get(compound_id)
    if not si or not si.smiles:
        return None
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
        mol = Chem.MolFromSmiles(si.smiles)
        if mol is None:
            return None
        return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
    except Exception:
        return None


def _tanimoto(fp1, fp2) -> float:
    """Compute Tanimoto similarity between two fingerprints."""
    from rdkit import DataStructs
    return DataStructs.TanimotoSimilarity(fp1, fp2)


def _match_one_compound(
    compound_id: str,
    compound: ChEBICompound,
    xrefs: dict[str, XrefInfo],
    structures: dict[str, StructureInfo],
    aracyc_records: dict[str, PlantCycCompound],
    plantcyc_records: dict[str, PlantCycCompound],
    aracyc_chebi_idx: dict[str, set[str]],
    aracyc_kegg_idx: dict[str, set[str]],
    aracyc_inchikey_idx: dict[str, set[str]],
    aracyc_fp_idx: dict[str, object],
    aracyc_name_idx: dict[str, dict[str, set[str]]],
    plantcyc_chebi_idx: dict[str, set[str]],
    plantcyc_kegg_idx: dict[str, set[str]],
    plantcyc_inchikey_idx: dict[str, set[str]],
    plantcyc_fp_idx: dict[str, object],
    plantcyc_name_idx: dict[str, dict[str, set[str]]],
) -> list[AraCycCompoundMatch]:
    """Try to match one ChEBI compound to AraCyc/PlantCyc records.

    Returns all matches found, sorted by score descending.
    """
    matches: list[AraCycCompoundMatch] = []

    seen_record_ids: set[str] = set()
    chebi_accession = f"CHEBI:{compound_id}"

    # --- Tier 1: Direct ChEBI xref (AraCyc) ---
    for record_id in aracyc_chebi_idx.get(chebi_accession, set()) | aracyc_chebi_idx.get(compound_id, set()):
        if record_id in seen_record_ids:
            continue
        seen_record_ids.add(record_id)
        record = aracyc_records[record_id]
        matches.append(_build_match(record, method="chebi_xref", score=1.00, chebi_xref_direct=True, structure_validated=False))

    # --- Tier 2: KEGG xref bridge (AraCyc) ---
    xref_info = xrefs.get(compound_id)
    if xref_info and xref_info.kegg_ids:
        for kegg_id in xref_info.kegg_ids:
            for record_id in aracyc_kegg_idx.get(kegg_id, set()):
                if record_id in seen_record_ids:
                    continue
                seen_record_ids.add(record_id)
                record = aracyc_records[record_id]
                matches.append(_build_match(record, method="kegg_xref", score=0.95, chebi_xref_direct=False, structure_validated=False))

    # --- Tier 3: InChIKey exact match (AraCyc) ---
    chebi_ik = _chebi_inchikey(compound_id, structures)
    if chebi_ik:
        for record_id in aracyc_inchikey_idx.get(chebi_ik, set()):
            if record_id in seen_record_ids:
                continue
            seen_record_ids.add(record_id)
            record = aracyc_records[record_id]
            matches.append(_build_match(record, method="inchikey", score=0.90, chebi_xref_direct=False, structure_validated=True))

    # --- Tier 3b: Tanimoto similarity >= 0.85 (AraCyc) ---
    # Disabled by default: O(N*M) brute-force is too slow for 200K compounds.
    # Uncomment or pass enable_tanimoto=True when a faster index is available.
    # if not matches and aracyc_fp_idx:
    #     chebi_fp = _chebi_morgan_fp(compound_id, structures)
    #     if chebi_fp is not None:
    #         best_sim, best_record_id = 0.0, None
    #         for record_id, aracyc_fp in aracyc_fp_idx.items():
    #             if record_id in seen_record_ids: continue
    #             sim = _tanimoto(chebi_fp, aracyc_fp)
    #             if sim >= 0.85 and sim > best_sim:
    #                 best_sim, best_record_id = sim, record_id
    #         if best_record_id is not None:
    #             seen_record_ids.add(best_record_id)
    #             record = aracyc_records[best_record_id]
    #             matches.append(_make_match(record, "tanimoto", round(0.85 * best_sim, 4), False, True))

    # --- Tier 4: Name matching (AraCyc) ---
    if not matches:
        all_names = [compound.name]
        compound_variants = {}
        for name in all_names:
            compound_variants.update(build_variants(name))

        name_match_scores = {"exact": 0.80, "compact": 0.76, "singular": 0.73, "stereo_stripped": 0.70}
        for vtype in ("exact", "compact", "singular", "stereo_stripped"):
            vtext = compound_variants.get(vtype, "")
            if not vtext:
                continue
            for record_id in aracyc_name_idx.get(vtype, {}).get(vtext, set()):
                if record_id in seen_record_ids:
                    continue
                seen_record_ids.add(record_id)
                record = aracyc_records[record_id]
                matches.append(_build_match(record, method=f"name_{vtype}", score=name_match_scores[vtype], chebi_xref_direct=False, structure_validated=False))

    # --- Tier 5: PlantCyc fallback (same tiers, discounted by 0.85) ---
    if not matches:
        plantcyc_matches = _match_plantcyc_fallback(
            compound_id, compound, xrefs, structures,
            plantcyc_records, plantcyc_chebi_idx, plantcyc_kegg_idx,
            plantcyc_inchikey_idx, plantcyc_fp_idx, plantcyc_name_idx,
            seen_record_ids,
        )
        matches.extend(plantcyc_matches)

    # Sort by score descending
    matches.sort(key=lambda m: m.match_score, reverse=True)
    return matches


def _match_plantcyc_fallback(
    compound_id: str,
    compound: ChEBICompound,
    xrefs: dict[str, XrefInfo],
    structures: dict[str, StructureInfo],
    plantcyc_records: dict[str, PlantCycCompound],
    plantcyc_chebi_idx: dict[str, set[str]],
    plantcyc_kegg_idx: dict[str, set[str]],
    plantcyc_inchikey_idx: dict[str, set[str]],
    plantcyc_fp_idx: dict[str, object],
    plantcyc_name_idx: dict[str, dict[str, set[str]]],
    seen_record_ids: set[str],
) -> list[AraCycCompoundMatch]:
    """Match against PlantCyc as a fallback source with discounted scores."""

    DISCOUNT = 0.85
    matches: list[AraCycCompoundMatch] = []

    chebi_accession = f"CHEBI:{compound_id}"

    # ChEBI xref
    for record_id in plantcyc_chebi_idx.get(chebi_accession, set()) | plantcyc_chebi_idx.get(compound_id, set()):
        if record_id in seen_record_ids:
            continue
        seen_record_ids.add(record_id)
        matches.append(_build_match(plantcyc_records[record_id], method="chebi_xref", score=1.00, chebi_xref_direct=True, structure_validated=False, discount=DISCOUNT))

    # KEGG xref bridge
    xref_info = xrefs.get(compound_id)
    if xref_info and xref_info.kegg_ids:
        for kegg_id in xref_info.kegg_ids:
            for record_id in plantcyc_kegg_idx.get(kegg_id, set()):
                if record_id in seen_record_ids:
                    continue
                seen_record_ids.add(record_id)
                matches.append(_build_match(plantcyc_records[record_id], method="kegg_xref", score=0.95, chebi_xref_direct=False, structure_validated=False, discount=DISCOUNT))

    # InChIKey
    chebi_ik = _chebi_inchikey(compound_id, structures)
    if chebi_ik:
        for record_id in plantcyc_inchikey_idx.get(chebi_ik, set()):
            if record_id in seen_record_ids:
                continue
            seen_record_ids.add(record_id)
            matches.append(_build_match(plantcyc_records[record_id], method="inchikey", score=0.90, chebi_xref_direct=False, structure_validated=True, discount=DISCOUNT))

    # Tanimoto — disabled (see AraCyc tier 3b comment above)

    # Name matching
    if not matches:
        compound_variants = build_variants(compound.name)
        name_scores = {"exact": 0.80, "compact": 0.76, "singular": 0.73, "stereo_stripped": 0.70}
        for vtype in ("exact", "compact", "singular", "stereo_stripped"):
            vtext = compound_variants.get(vtype, "")
            if not vtext:
                continue
            for record_id in plantcyc_name_idx.get(vtype, {}).get(vtext, set()):
                if record_id in seen_record_ids:
                    continue
                seen_record_ids.add(record_id)
                matches.append(_build_match(plantcyc_records[record_id], method=f"name_{vtype}", score=name_scores[vtype], chebi_xref_direct=False, structure_validated=False, discount=DISCOUNT))

    return matches


# ---------------------------------------------------------------------------
# Index writing
# ---------------------------------------------------------------------------


def write_name_to_aracyc_index(context: PipelineContext) -> int:
    """Write the ChEBI -> AraCyc mapping index for query-time lookup."""

    row_count = 0
    with context.paths.name_to_aracyc_index_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "compound_id",
                "chebi_accession",
                "chebi_name",
                "aracyc_compound_id",
                "aracyc_common_name",
                "source_db",
                "match_method",
                "match_score",
                "chebi_xref_direct",
                "structure_validated",
                "pathway_count",
                "pathways",
            ],
            delimiter="\t",
        )
        writer.writeheader()
        for compound_id in sorted(context.aracyc_matches_by_compound, key=int):
            compound = context.compounds[compound_id]
            for match in context.aracyc_matches_by_compound[compound_id]:
                writer.writerow({
                    "compound_id": compound_id,
                    "chebi_accession": f"CHEBI:{compound_id}",
                    "chebi_name": compound.name,
                    "aracyc_compound_id": match.aracyc_compound_id,
                    "aracyc_common_name": match.aracyc_common_name,
                    "source_db": match.source_db,
                    "match_method": match.match_method,
                    "match_score": f"{match.match_score:.4f}",
                    "chebi_xref_direct": str(match.chebi_xref_direct).lower(),
                    "structure_validated": str(match.structure_validated).lower(),
                    "pathway_count": len(match.pathways),
                    "pathways": ";".join(match.pathways),
                })
                row_count += 1
    return row_count


def write_structure_aracyc_index(context: PipelineContext) -> int:
    """Write the structure-based mapping index for compounds matched by InChIKey/Tanimoto."""

    row_count = 0
    with context.paths.compound_structure_aracyc_index_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "compound_id",
                "chebi_accession",
                "aracyc_compound_id",
                "source_db",
                "match_method",
                "match_score",
            ],
            delimiter="\t",
        )
        writer.writeheader()
        for compound_id in sorted(context.aracyc_matches_by_compound, key=int):
            for match in context.aracyc_matches_by_compound[compound_id]:
                if match.match_method in ("inchikey", "tanimoto"):
                    writer.writerow({
                        "compound_id": compound_id,
                        "chebi_accession": f"CHEBI:{compound_id}",
                        "aracyc_compound_id": match.aracyc_compound_id,
                        "source_db": match.source_db,
                        "match_method": match.match_method,
                        "match_score": f"{match.match_score:.4f}",
                    })
                    row_count += 1
    return row_count


# ---------------------------------------------------------------------------
# Step runner
# ---------------------------------------------------------------------------


def _filter_aracyc_records(records: dict[str, PlantCycCompound]) -> dict[str, PlantCycCompound]:
    """Filter to only AraCyc records."""
    return {k: v for k, v in records.items() if v.source_db == "AraCyc"}


def _filter_plantcyc_records(records: dict[str, PlantCycCompound]) -> dict[str, PlantCycCompound]:
    """Filter to only PlantCyc records (excluding AraCyc)."""
    return {k: v for k, v in records.items() if v.source_db == "PlantCyc"}


def run(context: PipelineContext) -> PipelineContext:
    """Match all ChEBI compounds to AraCyc/PlantCyc and write indexes."""

    print("  Step 2a: Building AraCyc/PlantCyc matching indexes...", flush=True)

    # Separate AraCyc and PlantCyc records (already loaded in step1)
    aracyc_records = _filter_aracyc_records(context.plantcyc_records)
    plantcyc_records = _filter_plantcyc_records(context.plantcyc_records)
    context.aracyc_reference_compound_keys = {
        _reference_compound_key(record) for record in aracyc_records.values()
    }

    print(f"    AraCyc: {len(aracyc_records)} compounds, PlantCyc: {len(plantcyc_records)} compounds", flush=True)
    print(
        f"    Arabidopsis reference compounds (AraCyc deduplicated): {context.aracyc_reference_total()}",
        flush=True,
    )

    # Build AraCyc indexes
    aracyc_chebi_idx = _build_chebi_xref_index(aracyc_records)
    aracyc_kegg_idx = _build_kegg_xref_index(aracyc_records)
    aracyc_name_idx = _build_name_index(aracyc_records)

    print("    Building InChIKey and fingerprint indexes...", flush=True)
    aracyc_inchikey_idx = _build_inchikey_index(aracyc_records)
    aracyc_fp_idx = _build_morgan_fp_index(aracyc_records)

    # Build PlantCyc indexes
    plantcyc_chebi_idx = _build_chebi_xref_index(plantcyc_records)
    plantcyc_kegg_idx = _build_kegg_xref_index(plantcyc_records)
    plantcyc_name_idx = _build_name_index(plantcyc_records)
    plantcyc_inchikey_idx = _build_inchikey_index(plantcyc_records)
    plantcyc_fp_idx = _build_morgan_fp_index(plantcyc_records)

    print(f"    AraCyc InChIKey index: {len(aracyc_inchikey_idx)} entries", flush=True)
    print(f"    AraCyc FP index: {len(aracyc_fp_idx)} entries", flush=True)

    # Match all compounds
    print("  Step 2a: Matching all compounds against AraCyc/PlantCyc...", flush=True)
    match_count = 0
    method_counts: dict[str, int] = defaultdict(int)

    for compound_id in sorted(context.compounds, key=int):
        compound = context.compounds[compound_id]
        matches = _match_one_compound(
            compound_id, compound,
            context.xrefs, context.structures,
            aracyc_records, plantcyc_records,
            aracyc_chebi_idx, aracyc_kegg_idx,
            aracyc_inchikey_idx, aracyc_fp_idx, aracyc_name_idx,
            plantcyc_chebi_idx, plantcyc_kegg_idx,
            plantcyc_inchikey_idx, plantcyc_fp_idx, plantcyc_name_idx,
        )

        if matches:
            # Keep only the best match (highest score)
            best = matches[0]
            context.aracyc_matches_by_compound[compound_id] = [best]
            match_count += 1
            method_counts[best.match_method] += 1
            context.aracyc_mapping_status[f"matched_{best.source_db.lower()}"] += 1
        else:
            context.aracyc_mapping_status["no_match"] += 1

    matched_aracyc_reference = {
        matches[0].reference_compound_key
        for matches in context.aracyc_matches_by_compound.values()
        if matches and matches[0].source_db == "AraCyc"
    }
    total = context.aracyc_reference_total()
    print(
        f"    Matched Arabidopsis reference compounds: {len(matched_aracyc_reference)}/{total} "
        f"({100*len(matched_aracyc_reference)/max(total, 1):.1f}%)",
        flush=True,
    )
    print(f"    By method: {dict(sorted(method_counts.items(), key=lambda x: -x[1]))}", flush=True)

    # Write indexes
    n2a = write_name_to_aracyc_index(context)
    print(f"    name_to_aracyc_index: {n2a} rows", flush=True)

    n_struct = write_structure_aracyc_index(context)
    print(f"    compound_structure_aracyc_index: {n_struct} rows", flush=True)

    context.preprocess_counts["step2a_aracyc_matched"] = match_count
    context.preprocess_counts["step2a_reference_compounds"] = total
    context.preprocess_counts["step2a_reference_matched"] = len(matched_aracyc_reference)
    return context
