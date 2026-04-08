"""Step 3: build versioned KEGG relation snapshots and reaction-role evidence.

This step upgrades the previous "compound -> pathway direct link" expansion into
two layers:

1. L1 snapshotting
   - freeze KEGG direct relation edges into a versioned local snapshot
   - record release text, fetch timestamps, and stable edge hashes
   - keep query-time behavior strictly offline and reproducible
2. L2 reaction-role support
   - add compound -> reaction -> pathway evidence
   - parse KEGG reaction equations to infer substrate/product/both roles
   - flag ubiquitous cofactors so later ranking can down-weight them

The downstream interface intentionally stays simple: later steps still receive
one row per selected KEGG compound and linked map pathway, but each row now
carries relation provenance and reaction-support summaries.
"""

from __future__ import annotations

import argparse
import csv
from datetime import UTC, datetime
import hashlib
import math
import re
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable
from urllib.error import URLError
from urllib.request import urlopen

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from process_chebi_to_pathways_v2 import load_kegg_pathway_links

from pathway_pipeline.cli_utils import build_context, build_parser, print_summary
from pathway_pipeline.context import PipelineContext, Step3PathwayHit
from pathway_pipeline.step1_alias_standardization import run as run_step1
from pathway_pipeline.step2_map_names_to_kegg import run as run_step2


KEGG_INFO_URLS = {
    "compound": "https://rest.kegg.jp/info/compound",
    "pathway": "https://rest.kegg.jp/info/pathway",
    "reaction": "https://rest.kegg.jp/info/reaction",
}
KEGG_LINK_URLS = {
    "compound_reaction": "https://rest.kegg.jp/link/reaction/compound",
    "reaction_pathway": "https://rest.kegg.jp/link/pathway/reaction",
}
KEGG_GET_URL = "https://rest.kegg.jp/get/{ids}"
REACTION_BATCH_SIZE = 10
REACTION_ARROW_RE = re.compile(r"\s*(<=>|=>|<=)\s*")
CID_WITH_STOICH_RE = re.compile(r"^\s*(?:(\d+(?:\.\d+)?)\s+)?(C\d{5})\b")
RID_RE = re.compile(r"^R\d{5}$")
PID_RE = re.compile(r"^map\d{5}$")
RELEASE_RE = re.compile(r"Release\s+(.+)")
COFACTOR_CIDS = {
    "C00001",  # H2O
    "C00002",  # ATP
    "C00003",  # NAD+
    "C00004",  # NADH
    "C00005",  # NADPH
    "C00006",  # NADP+
    "C00007",  # O2
    "C00008",  # ADP
    "C00009",  # orthophosphate
    "C00010",  # CoA
    "C00011",  # CO2
    "C00013",  # pyrophosphate
    "C00020",  # AMP
    "C00080",  # proton
}
COFACTOR_NAME_TOKENS = {
    "atp",
    "adp",
    "amp",
    "nadh",
    "nadph",
    "nadp",
    "nad",
    "water",
    "oxygen",
    "coenzyme a",
    "phosphate",
    "pyrophosphate",
    "carbon dioxide",
    "proton",
}


def fetch_text(url: str) -> str:
    """Fetch UTF-8 text from KEGG REST."""

    try:
        with urlopen(url, timeout=120) as response:  # noqa: S310 - KEGG REST is intentional here
            return response.read().decode("utf-8")
    except URLError:
        result = subprocess.run(
            ["curl", "-Lsf", url],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout


def normalize_release_tag(text: str) -> str:
    """Turn release text into a compact tag suitable for file/version stamping."""

    collapsed = re.sub(r"[^A-Za-z0-9._-]+", "-", text.strip()).strip("-")
    return collapsed[:80] if collapsed else "unknown"


def extract_release_line(info_text: str) -> str:
    """Extract the most informative KEGG release line from an info response."""

    for line in info_text.splitlines():
        match = RELEASE_RE.search(line)
        if match:
            return match.group(1).strip()
    for line in info_text.splitlines():
        if "Release" in line:
            return line.strip()
    return info_text.strip().splitlines()[0].strip() if info_text.strip() else "unknown"


def edge_hash(*parts: str) -> str:
    """Build a stable hash for an exported relation edge."""

    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()


def split_prefixed_id(value: str, expected_prefix: str) -> str:
    """Convert KEGG REST link values like ``cpd:C00031`` into ``C00031``."""

    if ":" not in value:
        return ""
    prefix, raw_id = value.split(":", 1)
    if prefix != expected_prefix:
        return ""
    return raw_id.strip()


def read_tsv_pairs(path: Path, left_prefix: str, right_prefix: str) -> list[tuple[str, str]]:
    """Read simple 2-column KEGG link tables with prefixed identifiers."""

    pairs = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            left, right = line.rstrip("\n").split("\t", 1)
            left_id = split_prefixed_id(left, left_prefix)
            right_id = split_prefixed_id(right, right_prefix)
            if left_id and right_id:
                pairs.append((left_id, right_id))
    return pairs


def parse_kegg_entry_blocks(raw_text: str) -> list[dict[str, list[str]]]:
    """Parse KEGG flat-file entries into field -> values dictionaries."""

    entries = []
    for block in raw_text.split("///"):
        block = block.strip()
        if not block:
            continue
        current_key = ""
        data: dict[str, list[str]] = defaultdict(list)
        for line in block.splitlines():
            key = line[:12].strip()
            value = line[12:].rstrip()
            if key:
                current_key = key
            if current_key:
                data[current_key].append(value.strip())
        entries.append(dict(data))
    return entries


def fetch_reaction_detail_rows(reaction_ids: Iterable[str]) -> list[dict[str, str]]:
    """Fetch structured KEGG reaction rows for the requested reaction IDs."""

    unique_ids = sorted({rid for rid in reaction_ids if RID_RE.match(rid)})
    rows: list[dict[str, str]] = []
    for start in range(0, len(unique_ids), REACTION_BATCH_SIZE):
        batch = unique_ids[start : start + REACTION_BATCH_SIZE]
        raw_text = fetch_text(KEGG_GET_URL.format(ids="+".join(batch)))
        for entry in parse_kegg_entry_blocks(raw_text):
            entry_field = entry.get("ENTRY", [""])[0]
            rid = entry_field.split()[0]
            if not RID_RE.match(rid):
                continue
            rows.append(
                {
                    "rid": rid,
                    "name": " ".join(entry.get("NAME", [])),
                    "definition": " ".join(entry.get("DEFINITION", [])),
                    "equation_text": " ".join(entry.get("EQUATION", [])),
                }
            )
    return rows


def refresh_step3_refs(context: PipelineContext) -> None:
    """Refresh the KEGG info/link inputs needed for versioned step-3 preprocessing."""

    refs = context.paths
    refs.kegg_info_compound_path.write_text(fetch_text(KEGG_INFO_URLS["compound"]), encoding="utf-8")
    refs.kegg_info_pathway_path.write_text(fetch_text(KEGG_INFO_URLS["pathway"]), encoding="utf-8")
    refs.kegg_info_reaction_path.write_text(fetch_text(KEGG_INFO_URLS["reaction"]), encoding="utf-8")
    refs.kegg_compound_reaction_path.write_text(fetch_text(KEGG_LINK_URLS["compound_reaction"]), encoding="utf-8")
    refs.kegg_reaction_pathway_path.write_text(fetch_text(KEGG_LINK_URLS["reaction_pathway"]), encoding="utf-8")


def ensure_step3_base_refs(context: PipelineContext) -> None:
    """Make sure the base step-3 KEGG refs exist, optionally refreshing them."""

    required = [
        context.paths.kegg_info_compound_path,
        context.paths.kegg_info_pathway_path,
        context.paths.kegg_info_reaction_path,
        context.paths.kegg_compound_reaction_path,
        context.paths.kegg_reaction_pathway_path,
    ]
    missing = [path for path in required if not path.exists()]
    if context.refresh_step3_kegg:
        try:
            refresh_step3_refs(context)
        except URLError as exc:  # pragma: no cover - network-dependent
            raise RuntimeError(f"Failed to refresh KEGG step-3 refs: {exc}") from exc
        return
    if missing:
        joined = ", ".join(path.name for path in missing)
        raise FileNotFoundError(
            "Missing Step 3 KEGG refs: "
            f"{joined}. Run preprocess_all.py with --refresh-step3-kegg to fetch them."
        )


def build_vstamp(info_texts: list[str]) -> tuple[str, str, str]:
    """Create the step-3 version stamp and release text summary."""

    release_lines = [extract_release_line(text) for text in info_texts if text.strip()]
    release_text = " | ".join(release_lines) if release_lines else "unknown"
    release_tag = normalize_release_tag(release_lines[0] if release_lines else "unknown")
    fetched_date = datetime.now(tz=UTC).date().isoformat()
    return f"kegg_step3_{fetched_date}__{release_tag}", release_text, fetched_date


def parse_reaction_details(path: Path) -> dict[str, dict[str, str]]:
    """Load structured reaction details exported during the refresh step."""

    details = {}
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            details[row["rid"]] = row
    return details


def write_reaction_details(path: Path, rows: list[dict[str, str]]) -> int:
    """Persist structured reaction details to the step-3 local ref file."""

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["rid", "name", "definition", "equation_text"],
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda item: item["rid"]))
    return len(rows)


def ensure_reaction_details(
    context: PipelineContext,
    required_reaction_ids: set[str],
) -> dict[str, dict[str, str]]:
    """Ensure reaction details exist for the reactions relevant to current mappings."""

    existing = parse_reaction_details(context.paths.kegg_reaction_details_path) if context.paths.kegg_reaction_details_path.exists() else {}
    missing = sorted(rid for rid in required_reaction_ids if rid not in existing)
    if missing and not context.refresh_step3_kegg:
        preview = ", ".join(missing[:10])
        raise FileNotFoundError(
            "Missing required KEGG reaction details for current Step 3 compounds. "
            f"Example missing reactions: {preview}. Run preprocess_all.py with --refresh-step3-kegg."
        )
    if missing:
        fetched_rows = fetch_reaction_detail_rows(missing)
        for row in fetched_rows:
            existing[row["rid"]] = row
        write_reaction_details(context.paths.kegg_reaction_details_path, list(existing.values()))
    return existing


def parse_equation_side(side_text: str) -> dict[str, float]:
    """Parse one equation side into compound stoichiometries."""

    compounds: dict[str, float] = defaultdict(float)
    for token in side_text.split("+"):
        token = token.strip()
        if not token:
            continue
        match = CID_WITH_STOICH_RE.match(token)
        if not match:
            continue
        stoich = float(match.group(1)) if match.group(1) else 1.0
        cid = match.group(2)
        compounds[cid] += stoich
    return dict(compounds)


def parse_equation_roles(equation_text: str) -> tuple[dict[str, float], dict[str, float]]:
    """Split a KEGG equation into left/right compound stoichiometries."""

    match = REACTION_ARROW_RE.search(equation_text)
    if not match:
        return {}, {}
    left = equation_text[: match.start()]
    right = equation_text[match.end() :]
    return parse_equation_side(left), parse_equation_side(right)


def is_cofactor_like(cid: str, kegg_primary_name: str, reaction_count: int, pathway_count: int) -> bool:
    """Flag ubiquitous helper molecules that should be down-weighted later."""

    normalized_name = kegg_primary_name.lower()
    if cid in COFACTOR_CIDS:
        return True
    if any(token in normalized_name for token in COFACTOR_NAME_TOKENS):
        return True
    return reaction_count >= 100 and pathway_count >= 30


def write_pathway_link_snapshot(
    context: PipelineContext,
    direct_rows: list[dict[str, str]],
) -> int:
    """Write the direct KEGG compound -> pathway snapshot with version metadata."""

    with context.paths.pathway_link_snapshot_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["cid", "pid", "source", "vstamp", "fetched_at", "release_text", "edge_hash"],
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(direct_rows)
    return len(direct_rows)


def write_snapshot_diff(
    path: Path,
    previous_rows: list[dict[str, str]],
    current_rows: list[dict[str, str]],
) -> int:
    """Diff the current pathway-link snapshot against the previous one."""

    previous = {row["edge_hash"] for row in previous_rows}
    current = {row["edge_hash"] for row in current_rows}
    row_count = 0
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["edge_hash", "change_type"],
            delimiter="\t",
        )
        writer.writeheader()
        for edge in sorted(current - previous):
            writer.writerow({"edge_hash": edge, "change_type": "added"})
            row_count += 1
        for edge in sorted(previous - current):
            writer.writerow({"edge_hash": edge, "change_type": "removed"})
            row_count += 1
        for edge in sorted(current & previous):
            writer.writerow({"edge_hash": edge, "change_type": "unchanged"})
            row_count += 1
    return row_count


def read_existing_snapshot(path: Path) -> list[dict[str, str]]:
    """Read the previous direct-link snapshot, if present."""

    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def copy_step3_history(context: PipelineContext) -> None:
    """Persist the latest step-3 outputs under a versioned history directory."""

    history_dir = context.paths.preprocessed_history_dir / context.step3_vstamp
    history_dir.mkdir(parents=True, exist_ok=True)
    for source in (
        context.paths.pathway_link_snapshot_path,
        context.paths.pathway_link_snapshot_diff_path,
        context.paths.compound_reaction_index_path,
        context.paths.reaction_pathway_index_path,
        context.paths.compound_to_pathway_index_path,
        context.paths.compound_to_pathway_role_index_path,
    ):
        if source.exists():
            shutil.copy2(source, history_dir / source.name)


def write_compound_reaction_index(
    context: PipelineContext,
    rows: list[dict[str, str]],
) -> int:
    """Write compound -> reaction role rows used by step-3 diagnostics and scoring."""

    with context.paths.compound_reaction_index_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "cid",
                "rid",
                "side",
                "stoich",
                "equation_text",
                "role_mask",
                "is_ubiquitous_candidate",
                "vstamp",
                "edge_hash",
            ],
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def write_reaction_pathway_index(
    context: PipelineContext,
    rows: list[dict[str, str]],
) -> int:
    """Write reaction -> pathway rows with version/provenance metadata."""

    with context.paths.reaction_pathway_index_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["rid", "pid", "source", "vstamp", "fetched_at", "release_text", "edge_hash"],
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def write_compound_to_pathway_index(
    context: PipelineContext,
    rows: list[dict[str, str]],
) -> int:
    """Write the union of direct and reaction-supported KEGG compound -> pathway rows."""

    with context.paths.compound_to_pathway_index_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "kegg_compound_id",
                "map_pathway_id",
                "relation_vstamp",
                "direct_link_source",
                "direct_link_edge_hash",
                "release_text_short",
            ],
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def write_compound_to_pathway_role_index(
    context: PipelineContext,
    rows: list[dict[str, str]],
) -> int:
    """Write aggregated reaction-role evidence per KEGG compound/pathway pair."""

    with context.paths.compound_to_pathway_role_index_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "kegg_compound_id",
                "map_pathway_id",
                "relation_vstamp",
                "direct_link",
                "support_reaction_count",
                "support_rids",
                "has_substrate_role",
                "has_product_role",
                "has_both_role",
                "cofactor_like",
                "role_summary",
                "reaction_role_score",
            ],
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def run(context: PipelineContext) -> PipelineContext:
    """Build versioned pathway snapshots and reaction-role evidence for Step 3."""

    ensure_step3_base_refs(context)

    info_texts = [
        context.paths.kegg_info_compound_path.read_text(encoding="utf-8"),
        context.paths.kegg_info_pathway_path.read_text(encoding="utf-8"),
        context.paths.kegg_info_reaction_path.read_text(encoding="utf-8"),
    ]
    context.step3_vstamp, context.step3_release_text, fetched_date = build_vstamp(info_texts)
    release_text_short = context.step3_release_text[:160]
    fetched_at = datetime.now(tz=UTC).isoformat()

    # Direct KEGG compound -> pathway links stay the canonical primary relation
    # table. Later reaction-derived support augments this table but does not
    # replace it.
    context.kegg_to_pathways, context.map_pathway_compound_counts = load_kegg_pathway_links(
        context.paths.kegg_compound_pathway_path,
        {},
    )

    previous_snapshot_rows = read_existing_snapshot(context.paths.pathway_link_snapshot_path)
    direct_snapshot_rows = []
    direct_link_edges = {}
    for kegg_compound_id in sorted(context.kegg_to_pathways):
        for map_pathway_id, _unused_ath in sorted(context.kegg_to_pathways[kegg_compound_id]):
            hashed = edge_hash(kegg_compound_id, map_pathway_id, "kegg_link", context.step3_vstamp)
            direct_snapshot_rows.append(
                {
                    "cid": kegg_compound_id,
                    "pid": map_pathway_id,
                    "source": "kegg_link",
                    "vstamp": context.step3_vstamp,
                    "fetched_at": fetched_at,
                    "release_text": context.step3_release_text,
                    "edge_hash": hashed,
                }
            )
            direct_link_edges[(kegg_compound_id, map_pathway_id)] = hashed

    context.preprocess_counts["pathway_link_snapshot"] = write_pathway_link_snapshot(context, direct_snapshot_rows)
    context.preprocess_counts["pathway_link_snapshot_diff"] = write_snapshot_diff(
        context.paths.pathway_link_snapshot_diff_path,
        previous_snapshot_rows,
        direct_snapshot_rows,
    )

    # Load raw reaction links and structured equation text so we can infer the
    # role a compound plays in each supporting reaction.
    compound_reaction_pairs = read_tsv_pairs(context.paths.kegg_compound_reaction_path, "cpd", "rn")
    reaction_pathway_pairs = [
        (rid, pid)
        for rid, pid in read_tsv_pairs(context.paths.kegg_reaction_pathway_path, "rn", "path")
        if PID_RE.match(pid)
    ]
    selected_kegg_ids = {
        mapping.kegg_compound_id
        for mappings in context.selected_by_compound.values()
        for mapping in mappings
    }
    required_reaction_ids = {
        rid
        for cid, rid in compound_reaction_pairs
        if cid in selected_kegg_ids
    }
    reaction_details = ensure_reaction_details(context, required_reaction_ids)

    rid_to_pathways: dict[str, set[str]] = defaultdict(set)
    cid_to_reactions: dict[str, set[str]] = defaultdict(set)
    reaction_rows = []
    for rid, pid in reaction_pathway_pairs:
        rid_to_pathways[rid].add(pid)
        reaction_rows.append(
            {
                "rid": rid,
                "pid": pid,
                "source": "kegg_link_reaction_pathway",
                "vstamp": context.step3_vstamp,
                "fetched_at": fetched_at,
                "release_text": context.step3_release_text,
                "edge_hash": edge_hash(rid, pid, "kegg_link_reaction_pathway", context.step3_vstamp),
            }
        )
    context.preprocess_counts["reaction_pathway_index"] = write_reaction_pathway_index(context, reaction_rows)

    reaction_role_rows = []
    role_by_cid_rid: dict[tuple[str, str], dict[str, object]] = {}
    for cid, rid in compound_reaction_pairs:
        cid_to_reactions[cid].add(rid)
        detail = reaction_details.get(rid, {})
        equation_text = detail.get("equation_text", "")
        left_side, right_side = parse_equation_roles(equation_text)
        on_left = cid in left_side
        on_right = cid in right_side
        if on_left and on_right:
            side = "both"
            role_mask = "both"
        elif on_left:
            side = "left"
            role_mask = "substrate"
        elif on_right:
            side = "right"
            role_mask = "product"
        else:
            side = "unknown"
            role_mask = "unknown"
        stoich = left_side.get(cid, 0.0) + right_side.get(cid, 0.0)
        pathway_count = len({pid for pid in rid_to_pathways.get(rid, set())})
        cofactor = is_cofactor_like(
            cid,
            context.kegg_compounds.get(cid).primary_name if cid in context.kegg_compounds else "",
            reaction_count=1,
            pathway_count=pathway_count,
        )
        row = {
            "cid": cid,
            "rid": rid,
            "side": side,
            "stoich": f"{stoich:.3f}".rstrip("0").rstrip(".") if stoich else "1",
            "equation_text": equation_text,
            "role_mask": role_mask,
            "is_ubiquitous_candidate": str(cofactor).lower(),
            "vstamp": context.step3_vstamp,
            "edge_hash": edge_hash(cid, rid, role_mask, context.step3_vstamp),
        }
        reaction_role_rows.append(row)
        role_by_cid_rid[(cid, rid)] = {
            "role_mask": role_mask,
            "equation_text": equation_text,
            "side": side,
            "stoich": stoich or 1.0,
        }

    # Final cofactor flag uses both the static whitelist and global frequency
    # thresholds across reactions and pathways.
    cid_pathway_counts = {
        cid: len({pid for rid in rid_set for pid in rid_to_pathways.get(rid, set())})
        for cid, rid_set in cid_to_reactions.items()
    }
    cofactor_flags = {}
    for cid, rid_set in cid_to_reactions.items():
        primary_name = context.kegg_compounds.get(cid).primary_name if cid in context.kegg_compounds else ""
        cofactor_flags[cid] = is_cofactor_like(cid, primary_name, len(rid_set), cid_pathway_counts.get(cid, 0))

    for row in reaction_role_rows:
        row["is_ubiquitous_candidate"] = str(cofactor_flags.get(row["cid"], False)).lower()
    context.preprocess_counts["compound_reaction_index"] = write_compound_reaction_index(context, reaction_role_rows)

    aggregated_roles: dict[tuple[str, str], dict[str, object]] = {}
    for cid, pathways in context.kegg_to_pathways.items():
        for map_pathway_id, _unused_ath in pathways:
            aggregated_roles.setdefault(
                (cid, map_pathway_id),
                {
                    "direct_link": True,
                    "support_rids": set(),
                    "has_substrate_role": False,
                    "has_product_role": False,
                    "has_both_role": False,
                    "cofactor_like": cofactor_flags.get(cid, False),
                },
            )

    for cid, rid in compound_reaction_pairs:
        role_info = role_by_cid_rid.get((cid, rid))
        if role_info is None:
            continue
        for pid in rid_to_pathways.get(rid, set()):
            entry = aggregated_roles.setdefault(
                (cid, pid),
                {
                    "direct_link": False,
                    "support_rids": set(),
                    "has_substrate_role": False,
                    "has_product_role": False,
                    "has_both_role": False,
                    "cofactor_like": cofactor_flags.get(cid, False),
                },
            )
            entry["support_rids"].add(rid)
            if role_info["role_mask"] == "substrate":
                entry["has_substrate_role"] = True
            elif role_info["role_mask"] == "product":
                entry["has_product_role"] = True
            elif role_info["role_mask"] == "both":
                entry["has_both_role"] = True
            entry["cofactor_like"] = entry["cofactor_like"] or cofactor_flags.get(cid, False)

    compound_to_pathway_rows = []
    compound_to_pathway_role_rows = []
    for (cid, pid), entry in sorted(aggregated_roles.items()):
        support_rids = tuple(sorted(entry["support_rids"]))
        support_reaction_count = len(support_rids)
        score = 1.0 if entry["direct_link"] else 0.0
        if entry["has_substrate_role"]:
            score += 0.35
        if entry["has_product_role"]:
            score += 0.20
        if entry["has_both_role"]:
            score += 0.10
        if entry["cofactor_like"]:
            score -= 0.40
        score += 0.10 * math.log1p(support_reaction_count)
        reaction_role_score = max(0.0, round(score, 3))
        role_tokens = []
        if entry["has_substrate_role"]:
            role_tokens.append("substrate")
        if entry["has_product_role"]:
            role_tokens.append("product")
        if entry["has_both_role"]:
            role_tokens.append("both")
        if entry["cofactor_like"]:
            role_tokens.append("cofactor_like")
        if support_reaction_count:
            role_tokens.append(f"support_reactions={support_reaction_count}")
        role_summary = ";".join(role_tokens) if role_tokens else "direct_link_only"

        compound_to_pathway_rows.append(
            {
                "kegg_compound_id": cid,
                "map_pathway_id": pid,
                "relation_vstamp": context.step3_vstamp,
                "direct_link_source": "kegg_link" if entry["direct_link"] else "",
                "direct_link_edge_hash": direct_link_edges.get((cid, pid), ""),
                "release_text_short": release_text_short,
            }
        )
        compound_to_pathway_role_rows.append(
            {
                "kegg_compound_id": cid,
                "map_pathway_id": pid,
                "relation_vstamp": context.step3_vstamp,
                "direct_link": str(entry["direct_link"]).lower(),
                "support_reaction_count": support_reaction_count,
                "support_rids": ";".join(support_rids),
                "has_substrate_role": str(entry["has_substrate_role"]).lower(),
                "has_product_role": str(entry["has_product_role"]).lower(),
                "has_both_role": str(entry["has_both_role"]).lower(),
                "cofactor_like": str(entry["cofactor_like"]).lower(),
                "role_summary": role_summary,
                "reaction_role_score": f"{reaction_role_score:.3f}",
            }
        )

    context.preprocess_counts["compound_to_pathway_index"] = write_compound_to_pathway_index(context, compound_to_pathway_rows)
    context.preprocess_counts["compound_to_pathway_role_index"] = write_compound_to_pathway_role_index(
        context,
        compound_to_pathway_role_rows,
    )
    copy_step3_history(context)

    # Expand selected mappings to all map pathways supported either by the
    # canonical direct KEGG link table or by the reaction layer.
    role_by_cid_pid = {
        (row["kegg_compound_id"], row["map_pathway_id"]): row
        for row in compound_to_pathway_role_rows
    }
    raw_hits = {}
    for compound_id, selected_mappings in context.selected_by_compound.items():
        hits = []
        for mapping in selected_mappings:
            supported_paths = sorted(
                {
                    pid
                    for (cid, pid) in role_by_cid_pid
                    if cid == mapping.kegg_compound_id
                }
            )
            for map_pathway_id in supported_paths:
                role_row = role_by_cid_pid[(mapping.kegg_compound_id, map_pathway_id)]
                hits.append(
                    Step3PathwayHit(
                        compound_id=compound_id,
                        mapping=mapping,
                        map_pathway_id=map_pathway_id,
                        relation_vstamp=role_row["relation_vstamp"],
                        direct_link=role_row["direct_link"] == "true",
                        support_reaction_count=int(role_row["support_reaction_count"]),
                        support_rids=tuple(value for value in role_row["support_rids"].split(";") if value),
                        has_substrate_role=role_row["has_substrate_role"] == "true",
                        has_product_role=role_row["has_product_role"] == "true",
                        has_both_role=role_row["has_both_role"] == "true",
                        cofactor_like=role_row["cofactor_like"] == "true",
                        role_summary=role_row["role_summary"],
                        reaction_role_score=float(role_row["reaction_role_score"]),
                    )
                )
        raw_hits[compound_id] = hits

    context.raw_pathway_hits = raw_hits
    return context


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for standalone step-3 execution."""

    parser = build_parser(
        description="Run pathway pipeline steps 1-3: build versioned compound-to-pathway and reaction-role indexes.",
        default_output_tag="step3_cli",
    )
    parser.add_argument(
        "--refresh-step3-kegg",
        action="store_true",
        help="Refresh KEGG step-3 refs (info, compound->reaction, reaction->pathway, reaction details).",
    )
    return parser.parse_args()


def main() -> None:
    """Run steps 1-3 and report versioned pathway-link counts."""

    args = parse_args()
    context = build_context(workdir=args.workdir, output_tag=args.output_tag)
    context.refresh_step3_kegg = args.refresh_step3_kegg
    run_step1(context)
    run_step2(context)
    run(context)
    raw_hit_count = sum(len(hits) for hits in context.raw_pathway_hits.values())
    print_summary(
        "Step 3 completed.",
        [
            f"Step 3 vstamp: {context.step3_vstamp}",
            f"Pathway link snapshot: {context.paths.pathway_link_snapshot_path}",
            f"Compound-reaction index: {context.paths.compound_reaction_index_path}",
            f"Reaction-pathway index: {context.paths.reaction_pathway_index_path}",
            f"Compound-to-pathway role index: {context.paths.compound_to_pathway_role_index_path}",
            f"Raw map-pathway hits: {raw_hit_count}",
        ],
    )


if __name__ == "__main__":
    main()
