"""Step 3 (AraCyc-first): extract compound-pathway links directly from AraCyc data.

Unlike the KEGG step 3 which requires REST API calls for compound→reaction→pathway
chains, AraCyc compound records already contain pathway annotations inline.
This step joins compound matches with the aracyc_pathways reference for
reaction/gene/EC metadata.
"""

from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from process_chebi_to_pathways_v2 import PlantCycCompound, normalize_name

from pathway_pipeline.context import (
    AraCycPathwayHit,
    AraCycPathwayInfo,
    PipelineContext,
)


# ---------------------------------------------------------------------------
# AraCyc pathway reference loading
# ---------------------------------------------------------------------------


def load_aracyc_pathway_info(path: Path, source_db: str) -> dict[str, AraCycPathwayInfo]:
    """Parse aracyc_pathways reference into pathway metadata.

    The file has one row per (pathway, reaction, gene) tuple. We aggregate
    to get per-pathway gene sets, reaction sets, EC sets.
    """
    pathways: dict[str, AraCycPathwayInfo] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            pid = row["Pathway-id"]
            pname = row["Pathway-name"]
            info = pathways.get(pid)
            if info is None:
                info = AraCycPathwayInfo(
                    pathway_id=pid,
                    pathway_name=pname,
                    source_dbs={source_db},
                    reaction_ids=set(),
                    ec_numbers=set(),
                    gene_ids=set(),
                    gene_names=set(),
                    protein_ids=set(),
                    compound_ids=set(),
                )
                pathways[pid] = info
            rid = row.get("Reaction-id", "")
            if rid:
                info.reaction_ids.add(rid)
            ec = row.get("EC", "")
            if ec:
                info.ec_numbers.add(ec)
            gene_id = row.get("Gene-id", "")
            if gene_id:
                info.gene_ids.add(gene_id)
            gene_name = row.get("Gene-name", "")
            if gene_name:
                info.gene_names.add(gene_name)
            protein_id = row.get("Protein-id", "")
            if protein_id:
                info.protein_ids.add(protein_id)
    return pathways


def _build_pathway_name_indexes(
    pathway_info: dict[str, AraCycPathwayInfo],
) -> tuple[dict[str, str], dict[str, str]]:
    """Map pathway names to IDs using exact and normalized forms."""
    exact_name_to_id: dict[str, str] = {}
    normalized_name_to_id: dict[str, str] = {}
    for pid, info in pathway_info.items():
        exact_name_to_id[info.pathway_name] = pid
        normalized = normalize_name(info.pathway_name)
        if normalized:
            normalized_name_to_id.setdefault(normalized, pid)
    return exact_name_to_id, normalized_name_to_id


def _merge_pathway_info(target: AraCycPathwayInfo, incoming: AraCycPathwayInfo) -> None:
    # Only source_dbs is merged. Gene/EC/compound/reaction fields feed step 5
    # scoring and must stay AraCyc-only so PlantCyc's multi-species content
    # does not inflate gene_support, ec_support, or pathway_specificity.
    target.source_dbs.update(incoming.source_dbs)


def _resolve_pathway_id(
    pathway_name: str,
    exact_name_to_id: dict[str, str],
    normalized_name_to_id: dict[str, str],
) -> str:
    """Resolve a raw pathway name to the best-known pathway ID."""
    pathway_id = exact_name_to_id.get(pathway_name, "")
    if pathway_id:
        return pathway_id
    return normalized_name_to_id.get(normalize_name(pathway_name), "")


def _read_compound_pathway_detail(
    path: Path,
) -> dict[str, dict[str, list[dict[str, str]]]]:
    """Re-read compound file to get per-row EC and reaction_equation per (compound, pathway).

    Returns: compound_id -> pathway_name -> [{"ec": ..., "reaction_equation": ...}, ...]
    """
    result: dict[str, dict[str, list[dict[str, str]]]] = defaultdict(lambda: defaultdict(list))
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            cid = row["Compound_id"]
            pathway = row.get("Pathway", "")
            if not pathway:
                continue
            ec = row.get("EC", "")
            rxn = row.get("Reaction_equation", "")
            result[cid][pathway].append({"ec": ec, "reaction_equation": rxn})
    return dict(result)


# ---------------------------------------------------------------------------
# Index writing
# ---------------------------------------------------------------------------


def write_compound_pathway_index(context: PipelineContext) -> int:
    """Write the AraCyc compound-to-pathway index.

    This index is the baseline bridge from resolved compounds to candidate
    pathways and is reused by both online enrichment and downstream training.
    """
    row_count = 0
    with context.paths.aracyc_compound_pathway_index_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "compound_id",
                "chebi_accession",
                "aracyc_compound_id",
                "source_db",
                "pathway_id",
                "pathway_name",
                "ec_numbers",
                "reaction_count",
                "match_method",
                "match_score",
            ],
            delimiter="\t",
        )
        writer.writeheader()
        for compound_id in sorted(context.aracyc_pathway_hits, key=int):
            for hit in context.aracyc_pathway_hits[compound_id]:
                writer.writerow({
                    "compound_id": compound_id,
                    "chebi_accession": f"CHEBI:{compound_id}",
                    "aracyc_compound_id": hit.match.aracyc_compound_id,
                    "source_db": hit.source_db,
                    "pathway_id": hit.pathway_id,
                    "pathway_name": hit.pathway_name,
                    "ec_numbers": ";".join(hit.ec_numbers),
                    "reaction_count": len(hit.reaction_ids),
                    "match_method": hit.match.match_method,
                    "match_score": f"{hit.match.match_score:.4f}",
                })
                row_count += 1
    return row_count


def write_raw_pathway_hits_audit(context: PipelineContext) -> int:
    """Write raw AraCyc pathway hits before annotation and scoring.

    The audit table preserves the uncollapsed pathway hit evidence so later
    review can distinguish linking issues from annotation or scoring issues.
    """

    row_count = 0
    with context.paths.aracyc_raw_pathway_hits_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "compound_id",
                "chebi_accession",
                "chebi_name",
                "hit_rank",
                "aracyc_compound_id",
                "aracyc_common_name",
                "source_db",
                "match_method",
                "match_score",
                "chebi_xref_direct",
                "structure_validated",
                "plant_record_id",
                "reference_compound_key",
                "pathway_id",
                "pathway_name",
                "ec_numbers",
                "reaction_ids",
                "reaction_equations",
            ],
            delimiter="\t",
        )
        writer.writeheader()
        for compound_id in sorted(context.aracyc_pathway_hits, key=int):
            chebi_name = context.compounds.get(compound_id).name if compound_id in context.compounds else ""
            for rank, hit in enumerate(context.aracyc_pathway_hits[compound_id], start=1):
                writer.writerow({
                    "compound_id": compound_id,
                    "chebi_accession": f"CHEBI:{compound_id}",
                    "chebi_name": chebi_name,
                    "hit_rank": rank,
                    "aracyc_compound_id": hit.match.aracyc_compound_id,
                    "aracyc_common_name": hit.match.aracyc_common_name,
                    "source_db": hit.source_db,
                    "match_method": hit.match.match_method,
                    "match_score": f"{hit.match.match_score:.4f}",
                    "chebi_xref_direct": str(hit.match.chebi_xref_direct).lower(),
                    "structure_validated": str(hit.match.structure_validated).lower(),
                    "plant_record_id": hit.match.plant_record_id,
                    "reference_compound_key": hit.match.reference_compound_key,
                    "pathway_id": hit.pathway_id,
                    "pathway_name": hit.pathway_name,
                    "ec_numbers": ";".join(hit.ec_numbers),
                    "reaction_ids": ";".join(hit.reaction_ids),
                    "reaction_equations": " ; ".join(hit.reaction_equations),
                })
                row_count += 1
    return row_count


# ---------------------------------------------------------------------------
# Step runner
# ---------------------------------------------------------------------------


def run(context: PipelineContext) -> PipelineContext:
    """Link matched AraCyc compounds to pathways and build pathway metadata."""

    print("  Step 3a: Loading AraCyc pathway reference...", flush=True)

    # Load pathway metadata from aracyc_pathways file
    pathway_info = load_aracyc_pathway_info(context.paths.aracyc_pathways_path, "AraCyc")
    context.aracyc_pathway_info = pathway_info
    pathway_name_to_id, normalized_pathway_name_to_id = _build_pathway_name_indexes(pathway_info)

    print(f"    AraCyc pathways loaded: {len(pathway_info)}", flush=True)

    # PlantCyc pathway metadata is only used as corroborating source metadata.
    # AraCyc remains the authoritative source for gene / EC / reaction details.
    # Also load PlantCyc pathway info if available
    plantcyc_pathway_info: dict[str, AraCycPathwayInfo] = {}
    if context.paths.plantcyc_pathways_path.exists():
        plantcyc_pathway_info = load_aracyc_pathway_info(context.paths.plantcyc_pathways_path, "PlantCyc")
        # Add PlantCyc pathways to the name indexes, merging exact/normalized
        # name matches into the existing AraCyc metadata record so downstream
        # corroboration checks can see both sources on the same pathway object.
        for pid, info in plantcyc_pathway_info.items():
            existing_pid = _resolve_pathway_id(
                info.pathway_name,
                pathway_name_to_id,
                normalized_pathway_name_to_id,
            )
            if existing_pid:
                _merge_pathway_info(context.aracyc_pathway_info[existing_pid], info)
                pathway_name_to_id.setdefault(info.pathway_name, existing_pid)
                normalized_name = normalize_name(info.pathway_name)
                if normalized_name:
                    normalized_pathway_name_to_id.setdefault(normalized_name, existing_pid)
                continue

            pathway_name_to_id[info.pathway_name] = pid
            normalized_name = normalize_name(info.pathway_name)
            if normalized_name:
                normalized_pathway_name_to_id.setdefault(normalized_name, pid)
            context.aracyc_pathway_info[pid] = info
        print(f"    PlantCyc pathways loaded: {len(plantcyc_pathway_info)}", flush=True)
    else:
        print("    Optional input missing: PlantCyc pathways file; skipping PlantCyc pathway corroboration.", flush=True)

    # Read per-row compound-pathway detail from raw files
    aracyc_detail = _read_compound_pathway_detail(context.paths.aracyc_compounds_path)
    plantcyc_detail = _read_compound_pathway_detail(context.paths.plantcyc_compounds_path)

    # Build compound-to-pathway counts (how many compounds per pathway)
    pathway_compound_counter: dict[str, set[str]] = defaultdict(set)

    # Link matched compounds to their pathways
    print("  Step 3a: Linking compounds to pathways...", flush=True)
    total_hits = 0
    compounds_with_pathways = 0

    for compound_id in sorted(context.aracyc_matches_by_compound, key=int):
        matches = context.aracyc_matches_by_compound[compound_id]
        hits: list[AraCycPathwayHit] = []

        for match in matches:
            # Get the pathway names from the match (already stored)
            for pathway_name in match.pathways:
                pathway_id = _resolve_pathway_id(
                    pathway_name,
                    pathway_name_to_id,
                    normalized_pathway_name_to_id,
                )

                # Get EC and reaction detail from raw file
                detail_source = aracyc_detail if match.source_db == "AraCyc" else plantcyc_detail
                row_details = detail_source.get(match.aracyc_compound_id, {}).get(pathway_name, [])

                ec_set: set[str] = set()
                rxn_eqs: list[str] = []
                rxn_ids: set[str] = set()

                for d in row_details:
                    if d["ec"]:
                        ec_set.add(d["ec"])
                    if d["reaction_equation"]:
                        rxn_eqs.append(d["reaction_equation"])

                # Also get reaction IDs from pathway info
                if pathway_id and pathway_id in context.aracyc_pathway_info:
                    pinfo = context.aracyc_pathway_info[pathway_id]
                    rxn_ids = set(pinfo.reaction_ids)
                    ec_set.update(pinfo.ec_numbers)
                    pinfo.compound_ids.add(match.aracyc_compound_id)

                pathway_compound_counter[pathway_id or pathway_name].add(compound_id)

                hits.append(AraCycPathwayHit(
                    compound_id=compound_id,
                    match=match,
                    pathway_id=pathway_id,
                    pathway_name=pathway_name,
                    ec_numbers=tuple(sorted(ec_set)),
                    reaction_ids=tuple(sorted(rxn_ids)),
                    reaction_equations=tuple(rxn_eqs[:5]),  # limit stored
                    source_db=match.source_db,
                ))

        if hits:
            context.aracyc_pathway_hits[compound_id] = hits
            total_hits += len(hits)
            compounds_with_pathways += 1

    # Store pathway compound counts for scoring
    context.aracyc_pathway_compound_counts = {
        k: len(v) for k, v in pathway_compound_counter.items()
    }

    total_matched = len(context.aracyc_matches_by_compound)
    aracyc_reference_with_pathways = {
        hit.match.reference_compound_key
        for hits in context.aracyc_pathway_hits.values()
        for hit in hits
        if hit.match.source_db == "AraCyc"
    }
    total_compounds = context.aracyc_reference_total()
    print(
        f"    Arabidopsis reference compounds with pathways: "
        f"{len(aracyc_reference_with_pathways)}/{total_compounds}",
        flush=True,
    )
    print(f"    Matched ChEBI compounds with pathways: {compounds_with_pathways}/{total_matched}", flush=True)
    print(f"    Total compound-pathway hits: {total_hits}", flush=True)
    print(
        f"    Coverage: {100*len(aracyc_reference_with_pathways)/max(total_compounds, 1):.2f}%",
        flush=True,
    )

    # Write index
    n_idx = write_compound_pathway_index(context)
    print(f"    aracyc_compound_pathway_index: {n_idx} rows", flush=True)
    n_raw_hits = write_raw_pathway_hits_audit(context)
    print(f"    aracyc_raw_pathway_hits audit: {n_raw_hits} rows", flush=True)

    context.preprocess_counts["step3a_compounds_with_pathways"] = compounds_with_pathways
    context.preprocess_counts["step3a_total_pathway_hits"] = total_hits
    context.preprocess_counts["step3a_raw_hit_rows"] = n_raw_hits
    return context
