#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import gzip
import html
import json
import math
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path


TAG_RE = re.compile(r"<[^>]+>")
WS_RE = re.compile(r"\s+")
NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")
KEGG_ID_RE = re.compile(r"^C\d{5}$")
PATHWAY_LINE_RE = re.compile(r"^C\s+(\d{5})\s+(.+)$")

ALLOWED_ALIAS_TYPES = {"SYNONYM", "IUPAC NAME", "INN"}
ALIAS_SOURCE_WEIGHTS = {
    "compound_name": 0.08,
    "ascii_name": 0.06,
    "SYNONYM": 0.05,
    "IUPAC NAME": 0.04,
    "INN": 0.03,
}
VARIANT_BASE_SCORES = {
    "exact": 0.86,
    "compact": 0.83,
    "singular": 0.81,
    "stereo_stripped": 0.79,
}
VARIANT_ORDER = ("exact", "compact", "singular", "stereo_stripped")
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
    raw_name: str
    source_type: str
    language_code: str
    exact: str
    compact: str
    singular: str
    stereo_stripped: str


@dataclass(slots=True)
class ChEBICompound:
    compound_id: str
    chebi_accession: str
    name: str
    ascii_name: str
    definition: str
    stars: int
    status_id: str


@dataclass(slots=True)
class KeggCompound:
    compound_id: str
    primary_name: str
    aliases: list[str]
    primary_exact: str
    primary_compact: str
    primary_singular: str
    primary_stereo_stripped: str


@dataclass
class CandidateMapping:
    kegg_compound_id: str
    kegg_primary_name: str
    best_score: float = 0.0
    final_score: float = 0.0
    direct_xref: bool = False
    primary_name_match: bool = False
    evidence_count: int = 0
    best_alias: str = ""
    best_source_type: str = ""
    best_variant: str = ""
    reasons: list[str] = field(default_factory=list)
    methods: set[str] = field(default_factory=set)
    matched_aliases: set[str] = field(default_factory=set)

    def add_evidence(
        self,
        *,
        score: float,
        method: str,
        reason: str,
        alias: str = "",
        source_type: str = "",
        variant: str = "",
        direct_xref: bool = False,
        primary_name_match: bool = False,
    ) -> None:
        self.evidence_count += 1
        self.methods.add(method)
        if reason and reason not in self.reasons and len(self.reasons) < 6:
            self.reasons.append(reason)
        if alias:
            self.matched_aliases.add(alias)
        if score > self.best_score:
            self.best_score = score
            self.best_alias = alias or self.best_alias
            self.best_source_type = source_type or self.best_source_type
            self.best_variant = variant or self.best_variant
        self.direct_xref = self.direct_xref or direct_xref
        self.primary_name_match = self.primary_name_match or primary_name_match


def clean_markup(text: str) -> str:
    text = html.unescape(text or "")
    text = text.translate(GREEK_MAP)
    text = TAG_RE.sub(" ", text)
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return text


def normalize_name(text: str) -> str:
    text = clean_markup(text).lower()
    text = text.replace("'", "")
    text = NON_ALNUM_RE.sub(" ", text)
    text = WS_RE.sub(" ", text).strip()
    return text


def singularize_token(token: str) -> str:
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
    tokens = normalized.split()
    while tokens and tokens[0] in LEADING_STEREO_TOKENS:
        tokens.pop(0)
    return " ".join(tokens)


def build_variants(text: str) -> dict[str, str]:
    exact = normalize_name(text)
    singular = " ".join(singularize_token(token) for token in exact.split())
    stereo_stripped = strip_stereo_tokens(singular)
    return {
        "exact": exact,
        "compact": exact.replace(" ", ""),
        "singular": singular,
        "stereo_stripped": stereo_stripped,
    }


def load_compounds(path: Path) -> dict[str, ChEBICompound]:
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
    counter: Counter[str] = Counter()
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            counter[row["datatype"] or "<empty>"] += 1
    return dict(counter.most_common())


def record_alias(
    alias_bucket: dict[str, AliasRecord],
    *,
    raw_name: str,
    source_type: str,
    language_code: str = "en",
) -> None:
    variants = build_variants(raw_name)
    if not variants["exact"]:
        return
    candidate = AliasRecord(
        raw_name=raw_name,
        source_type=source_type,
        language_code=language_code or "",
        exact=variants["exact"],
        compact=variants["compact"],
        singular=variants["singular"],
        stereo_stripped=variants["stereo_stripped"],
    )
    existing = alias_bucket.get(candidate.exact)
    if existing is None:
        alias_bucket[candidate.exact] = candidate
        return
    if ALIAS_SOURCE_WEIGHTS.get(candidate.source_type, 0.0) > ALIAS_SOURCE_WEIGHTS.get(
        existing.source_type,
        0.0,
    ):
        alias_bucket[candidate.exact] = candidate


def load_aliases(
    compounds: dict[str, ChEBICompound],
    names_path: Path,
) -> dict[str, list[AliasRecord]]:
    aliases: dict[str, dict[str, AliasRecord]] = {compound_id: {} for compound_id in compounds}
    for compound_id, compound in compounds.items():
        record_alias(aliases[compound_id], raw_name=compound.name, source_type="compound_name")
        if compound.ascii_name and compound.ascii_name != compound.name:
            record_alias(aliases[compound_id], raw_name=compound.ascii_name, source_type="ascii_name")

    with gzip.open(names_path, "rt", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            compound_id = row["compound_id"]
            if compound_id not in aliases:
                continue
            if row["type"] not in ALLOWED_ALIAS_TYPES:
                continue
            if row["language_code"] not in {"", "en"}:
                continue
            record_alias(
                aliases[compound_id],
                raw_name=row["name"],
                source_type=row["type"],
                language_code=row["language_code"],
            )
            if row["ascii_name"] and row["ascii_name"] != row["name"]:
                record_alias(
                    aliases[compound_id],
                    raw_name=row["ascii_name"],
                    source_type=row["type"],
                    language_code=row["language_code"],
                )

    return {compound_id: list(alias_map.values()) for compound_id, alias_map in aliases.items()}


def load_direct_kegg_xrefs(path: Path) -> dict[str, set[str]]:
    xrefs: defaultdict[str, set[str]] = defaultdict(set)
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            accession_number = row["accession_number"].strip()
            if KEGG_ID_RE.match(accession_number):
                xrefs[row["compound_id"]].add(accession_number)
    return dict(xrefs)


def load_kegg_compounds(
    path: Path,
) -> tuple[dict[str, KeggCompound], dict[str, defaultdict[str, set[str]]]]:
    compounds: dict[str, KeggCompound] = {}
    indexes = {
        "exact": defaultdict(set),
        "compact": defaultdict(set),
        "singular": defaultdict(set),
        "stereo_stripped": defaultdict(set),
    }

    with path.open(encoding="utf-8") as handle:
        for line in handle:
            compound_id, raw_names = line.rstrip("\n").split("\t", 1)
            aliases = [name.strip() for name in raw_names.split(";") if name.strip()]
            primary = aliases[0]
            primary_variants = build_variants(primary)
            compounds[compound_id] = KeggCompound(
                compound_id=compound_id,
                primary_name=primary,
                aliases=aliases,
                primary_exact=primary_variants["exact"],
                primary_compact=primary_variants["compact"],
                primary_singular=primary_variants["singular"],
                primary_stereo_stripped=primary_variants["stereo_stripped"],
            )
            for alias in aliases:
                variants = build_variants(alias)
                for variant_name, variant_value in variants.items():
                    if variant_value:
                        indexes[variant_name][variant_value].add(compound_id)
    return compounds, indexes


def load_ath_pathways(path: Path) -> tuple[dict[str, str], dict[str, str]]:
    ath_pathways: dict[str, str] = {}
    map_to_ath: dict[str, str] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            pathway_id, raw_name = line.rstrip("\n").split("\t", 1)
            pathway_name = raw_name.split(" - ", 1)[0]
            ath_pathways[pathway_id] = pathway_name
            map_to_ath[f"map{pathway_id[3:]}"] = pathway_id
    return ath_pathways, map_to_ath


def load_pathway_categories(path: Path) -> dict[str, tuple[str, str, str]]:
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
    links: defaultdict[str, list[tuple[str, str]]] = defaultdict(list)
    pathway_compound_counts: Counter[str] = Counter()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            compound_ref, pathway_ref = line.rstrip("\n").split("\t", 1)
            kegg_compound_id = compound_ref.replace("cpd:", "")
            map_pathway_id = pathway_ref.replace("path:", "")
            ath_pathway_id = map_to_ath.get(map_pathway_id)
            if not ath_pathway_id:
                continue
            links[kegg_compound_id].append((map_pathway_id, ath_pathway_id))
            pathway_compound_counts[ath_pathway_id] += 1
    return dict(links), dict(pathway_compound_counts)


def build_candidate_mappings(
    aliases: list[AliasRecord],
    direct_xrefs: set[str],
    kegg_compounds: dict[str, KeggCompound],
    kegg_indexes: dict[str, defaultdict[str, set[str]]],
) -> list[CandidateMapping]:
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

    for kegg_compound_id in sorted(direct_xrefs):
        if kegg_compound_id not in kegg_compounds:
            continue
        candidate_for(kegg_compound_id).add_evidence(
            score=0.99,
            method="chebi_database_accession",
            reason=f"Direct ChEBI database accession points to {kegg_compound_id}",
            direct_xref=True,
        )

    for alias in aliases:
        for variant_name in VARIANT_ORDER:
            variant_value = getattr(alias, variant_name)
            if not variant_value:
                continue
            hit_ids = kegg_indexes[variant_name].get(variant_value)
            if not hit_ids:
                continue
            for kegg_compound_id in hit_ids:
                kegg = kegg_compounds[kegg_compound_id]
                primary_match = variant_value == getattr(kegg, f"primary_{variant_name}")
                score = VARIANT_BASE_SCORES[variant_name] + ALIAS_SOURCE_WEIGHTS.get(
                    alias.source_type,
                    0.0,
                )
                if primary_match:
                    score += 0.03
                candidate_for(kegg_compound_id).add_evidence(
                    score=min(score, 0.98),
                    method=f"name_match_{variant_name}",
                    reason=f"{variant_name} match via {alias.source_type}: {alias.raw_name}",
                    alias=alias.raw_name,
                    source_type=alias.source_type,
                    variant=variant_name,
                    primary_name_match=primary_match,
                )
            break

    ranked = sorted(
        candidates.values(),
        key=lambda candidate: (
            int(candidate.direct_xref),
            candidate.best_score,
            int(candidate.primary_name_match),
            candidate.evidence_count,
            candidate.kegg_compound_id,
        ),
        reverse=True,
    )
    for candidate in ranked:
        bonus = min(0.05, 0.01 * max(candidate.evidence_count - 1, 0))
        if candidate.direct_xref:
            bonus += 0.01
        candidate.final_score = min(0.999, candidate.best_score + bonus)
    return ranked


def select_candidates(ranked: list[CandidateMapping]) -> list[CandidateMapping]:
    if not ranked:
        return []
    direct = [candidate for candidate in ranked if candidate.direct_xref]
    if direct:
        return direct

    top = ranked[0]
    if top.final_score < 0.87:
        return []
    if len(ranked) == 1:
        return [top]

    second = ranked[1]
    if top.final_score >= 0.95:
        return [top]
    if top.final_score - second.final_score >= 0.03:
        return [top]
    return []


def write_alias_table(
    path: Path,
    compounds: dict[str, ChEBICompound],
    aliases_by_compound: dict[str, list[AliasRecord]],
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "compound_id",
                "chebi_accession",
                "chebi_name",
                "alias",
                "source_type",
                "language_code",
                "normalized_name",
                "compact_name",
                "singular_name",
                "stereo_stripped_name",
            ],
            delimiter="\t",
        )
        writer.writeheader()
        for compound_id in sorted(compounds, key=int):
            compound = compounds[compound_id]
            for alias in sorted(aliases_by_compound[compound_id], key=lambda entry: (entry.source_type, entry.raw_name)):
                writer.writerow(
                    {
                        "compound_id": compound_id,
                        "chebi_accession": compound.chebi_accession,
                        "chebi_name": compound.name,
                        "alias": alias.raw_name,
                        "source_type": alias.source_type,
                        "language_code": alias.language_code,
                        "normalized_name": alias.exact,
                        "compact_name": alias.compact,
                        "singular_name": alias.singular,
                        "stereo_stripped_name": alias.stereo_stripped,
                    }
                )


def reason_summary(candidate: CandidateMapping) -> str:
    if not candidate.reasons:
        return ""
    return "; ".join(candidate.reasons[:3])


def write_mapping_tables(
    summary_path: Path,
    selected_path: Path,
    compounds: dict[str, ChEBICompound],
    aliases_by_compound: dict[str, list[AliasRecord]],
    direct_xrefs: dict[str, set[str]],
    kegg_compounds: dict[str, KeggCompound],
    kegg_indexes: dict[str, defaultdict[str, set[str]]],
) -> tuple[dict[str, list[CandidateMapping]], Counter[str]]:
    selected_by_compound: dict[str, list[CandidateMapping]] = {}
    status_counter: Counter[str] = Counter()

    with summary_path.open("w", newline="", encoding="utf-8") as summary_handle, selected_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as selected_handle:
        summary_writer = csv.DictWriter(
            summary_handle,
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
                "mapping_reason",
            ],
            delimiter="\t",
        )
        summary_writer.writeheader()

        selected_writer = csv.DictWriter(
            selected_handle,
            fieldnames=[
                "compound_id",
                "chebi_accession",
                "chebi_name",
                "kegg_compound_id",
                "kegg_primary_name",
                "mapping_score",
                "mapping_method",
                "best_alias",
                "best_alias_source",
                "best_variant",
                "evidence_count",
                "mapping_reason",
            ],
            delimiter="\t",
        )
        selected_writer.writeheader()

        for compound_id in sorted(compounds, key=int):
            compound = compounds[compound_id]
            ranked = build_candidate_mappings(
                aliases=aliases_by_compound[compound_id],
                direct_xrefs=direct_xrefs.get(compound_id, set()),
                kegg_compounds=kegg_compounds,
                kegg_indexes=kegg_indexes,
            )
            selected = select_candidates(ranked)
            selected_by_compound[compound_id] = selected

            if selected:
                status = "selected"
            elif ranked:
                status = "ambiguous"
            else:
                status = "unmapped"
            status_counter[status] += 1

            top = ranked[0] if ranked else None
            summary_writer.writerow(
                {
                    "compound_id": compound_id,
                    "chebi_accession": compound.chebi_accession,
                    "chebi_name": compound.name,
                    "selection_status": status,
                    "selected_kegg_compound_ids": ";".join(mapping.kegg_compound_id for mapping in selected),
                    "selected_kegg_names": ";".join(mapping.kegg_primary_name for mapping in selected),
                    "selected_scores": ";".join(f"{mapping.final_score:.3f}" for mapping in selected),
                    "top_candidate_kegg_compound_id": top.kegg_compound_id if top else "",
                    "top_candidate_name": top.kegg_primary_name if top else "",
                    "top_candidate_score": f"{top.final_score:.3f}" if top else "",
                    "candidate_count": len(ranked),
                    "alias_count": len(aliases_by_compound[compound_id]),
                    "mapping_reason": reason_summary(top) if top else "",
                }
            )

            for mapping in selected:
                mapping_method = "chebi_database_accession" if mapping.direct_xref else f"name_match_{mapping.best_variant}"
                selected_writer.writerow(
                    {
                        "compound_id": compound_id,
                        "chebi_accession": compound.chebi_accession,
                        "chebi_name": compound.name,
                        "kegg_compound_id": mapping.kegg_compound_id,
                        "kegg_primary_name": mapping.kegg_primary_name,
                        "mapping_score": f"{mapping.final_score:.3f}",
                        "mapping_method": mapping_method,
                        "best_alias": mapping.best_alias,
                        "best_alias_source": mapping.best_source_type,
                        "best_variant": mapping.best_variant,
                        "evidence_count": mapping.evidence_count,
                        "mapping_reason": reason_summary(mapping),
                    }
                )
    return selected_by_compound, status_counter


def write_pathway_table(
    path: Path,
    compounds: dict[str, ChEBICompound],
    selected_by_compound: dict[str, list[CandidateMapping]],
    kegg_to_pathways: dict[str, list[tuple[str, str]]],
    ath_pathways: dict[str, str],
    pathway_categories: dict[str, tuple[str, str, str]],
    pathway_compound_counts: dict[str, int],
) -> Counter[str]:
    pathway_status: Counter[str] = Counter()
    max_pathway_size = max(pathway_compound_counts.values(), default=1)

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "compound_id",
                "chebi_accession",
                "chebi_name",
                "pathway_rank",
                "score",
                "mapping_confidence",
                "support_kegg_compound_ids",
                "support_kegg_names",
                "ath_pathway_id",
                "map_pathway_id",
                "pathway_name",
                "pathway_group",
                "pathway_category",
                "pathway_compound_count",
                "supporting_mapping_count",
                "reason",
            ],
            delimiter="\t",
        )
        writer.writeheader()

        for compound_id in sorted(compounds, key=int):
            compound = compounds[compound_id]
            selected = selected_by_compound.get(compound_id, [])
            if not selected:
                pathway_status["without_mapping"] += 1
                continue

            aggregated: dict[str, dict[str, object]] = {}
            for mapping in selected:
                for map_pathway_id, ath_pathway_id in kegg_to_pathways.get(mapping.kegg_compound_id, []):
                    pathway_size = pathway_compound_counts[ath_pathway_id]
                    specificity = 1 - math.log1p(pathway_size) / math.log1p(max_pathway_size)
                    score = (
                        0.7 * mapping.final_score
                        + 0.25 * specificity
                        + 0.03 * int(mapping.direct_xref)
                        + 0.02 * min(mapping.evidence_count, 3)
                    )
                    entry = aggregated.setdefault(
                        ath_pathway_id,
                        {
                            "ath_pathway_id": ath_pathway_id,
                            "map_pathway_id": map_pathway_id,
                            "pathway_name": ath_pathways[ath_pathway_id],
                            "pathway_group": pathway_categories.get(map_pathway_id, ("", "", ""))[0],
                            "pathway_category": pathway_categories.get(map_pathway_id, ("", "", ""))[1],
                            "pathway_compound_count": pathway_size,
                            "score": 0.0,
                            "mapping_confidence": 0.0,
                            "support_kegg_ids": set(),
                            "support_kegg_names": set(),
                            "reasons": [],
                        },
                    )
                    entry["score"] = max(float(entry["score"]), min(score, 0.999))
                    entry["mapping_confidence"] = max(float(entry["mapping_confidence"]), mapping.final_score)
                    entry["support_kegg_ids"].add(mapping.kegg_compound_id)
                    entry["support_kegg_names"].add(mapping.kegg_primary_name)
                    pathway_reason = (
                        f"{mapping.kegg_compound_id} -> {ath_pathway_id} via {reason_summary(mapping)}; "
                        f"pathway specificity {specificity:.3f}"
                    )
                    if pathway_reason not in entry["reasons"] and len(entry["reasons"]) < 4:
                        entry["reasons"].append(pathway_reason)

            if not aggregated:
                pathway_status["mapped_without_ath_pathway"] += 1
                continue

            ranked_pathways = []
            for entry in aggregated.values():
                support_count = len(entry["support_kegg_ids"])
                entry["score"] = min(float(entry["score"]) + min(0.05, 0.02 * (support_count - 1)), 0.999)
                ranked_pathways.append(entry)

            ranked_pathways.sort(
                key=lambda entry: (
                    float(entry["score"]),
                    -int(entry["pathway_compound_count"]),
                    entry["ath_pathway_id"],
                ),
                reverse=True,
            )

            pathway_status["with_ath_pathway"] += 1
            pathway_status["pathway_rows"] += len(ranked_pathways)

            for rank, entry in enumerate(ranked_pathways, start=1):
                writer.writerow(
                    {
                        "compound_id": compound_id,
                        "chebi_accession": compound.chebi_accession,
                        "chebi_name": compound.name,
                        "pathway_rank": rank,
                        "score": f"{float(entry['score']):.3f}",
                        "mapping_confidence": f"{float(entry['mapping_confidence']):.3f}",
                        "support_kegg_compound_ids": ";".join(sorted(entry["support_kegg_ids"])),
                        "support_kegg_names": ";".join(sorted(entry["support_kegg_names"])),
                        "ath_pathway_id": entry["ath_pathway_id"],
                        "map_pathway_id": entry["map_pathway_id"],
                        "pathway_name": entry["pathway_name"],
                        "pathway_group": entry["pathway_group"],
                        "pathway_category": entry["pathway_category"],
                        "pathway_compound_count": entry["pathway_compound_count"],
                        "supporting_mapping_count": len(entry["support_kegg_ids"]),
                        "reason": " | ".join(entry["reasons"]),
                    }
                )
    return pathway_status


def build_summary(
    *,
    compounds: dict[str, ChEBICompound],
    aliases_by_compound: dict[str, list[AliasRecord]],
    direct_xrefs: dict[str, set[str]],
    mapping_status: Counter[str],
    pathway_status: Counter[str],
    comments_profile: dict[str, int],
) -> dict[str, object]:
    alias_count = sum(len(aliases) for aliases in aliases_by_compound.values())
    compounds_with_direct_xref = sum(1 for xrefs in direct_xrefs.values() if xrefs)
    return {
        "compounds_total": len(compounds),
        "standardized_alias_rows": alias_count,
        "compounds_with_direct_kegg_xref": compounds_with_direct_xref,
        "mapping_status": dict(mapping_status),
        "pathway_status": dict(pathway_status),
        "comments_profile": comments_profile,
        "notes": [
            "comments.tsv was profiled but not used as a synonym source because its CompoundName rows are mostly editorial notes rather than clean aliases.",
            "Pathway scores are rule-based features intended to support later ML ranking when labeled data is available.",
        ],
    }


def ensure_exists(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Process local ChEBI flat files into KEGG ath pathway mappings.")
    parser.add_argument("--workdir", default=".", help="Workspace containing compounds.tsv, comments.tsv, refs/, and outputs/.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    workdir = Path(args.workdir).resolve()
    refs = workdir / "refs"
    outputs = workdir / "outputs"
    outputs.mkdir(parents=True, exist_ok=True)

    compounds_path = workdir / "compounds.tsv"
    comments_path = workdir / "comments.tsv"
    names_path = refs / "names.tsv.gz"
    accession_path = refs / "database_accession.tsv.gz"
    kegg_compound_path = refs / "kegg_compound_list.tsv"
    kegg_link_path = refs / "kegg_compound_pathway.tsv"
    ath_pathway_path = refs / "kegg_pathway_ath.tsv"
    pathway_hierarchy_path = refs / "kegg_pathway_hierarchy.txt"

    for required_path in [
        compounds_path,
        comments_path,
        names_path,
        accession_path,
        kegg_compound_path,
        kegg_link_path,
        ath_pathway_path,
        pathway_hierarchy_path,
    ]:
        ensure_exists(required_path)

    compounds = load_compounds(compounds_path)
    comments_profile = load_comments_profile(comments_path)
    aliases_by_compound = load_aliases(compounds, names_path)
    direct_xrefs = load_direct_kegg_xrefs(accession_path)
    kegg_compounds, kegg_indexes = load_kegg_compounds(kegg_compound_path)
    ath_pathways, map_to_ath = load_ath_pathways(ath_pathway_path)
    pathway_categories = load_pathway_categories(pathway_hierarchy_path)
    kegg_to_pathways, pathway_compound_counts = load_kegg_pathway_links(kegg_link_path, map_to_ath)

    write_alias_table(outputs / "chebi_aliases_standardized.tsv", compounds, aliases_by_compound)
    selected_by_compound, mapping_status = write_mapping_tables(
        summary_path=outputs / "chebi_kegg_mapping.tsv",
        selected_path=outputs / "chebi_kegg_selected.tsv",
        compounds=compounds,
        aliases_by_compound=aliases_by_compound,
        direct_xrefs=direct_xrefs,
        kegg_compounds=kegg_compounds,
        kegg_indexes=kegg_indexes,
    )
    pathway_status = write_pathway_table(
        path=outputs / "chebi_ath_pathways.tsv",
        compounds=compounds,
        selected_by_compound=selected_by_compound,
        kegg_to_pathways=kegg_to_pathways,
        ath_pathways=ath_pathways,
        pathway_categories=pathway_categories,
        pathway_compound_counts=pathway_compound_counts,
    )

    summary = build_summary(
        compounds=compounds,
        aliases_by_compound=aliases_by_compound,
        direct_xrefs=direct_xrefs,
        mapping_status=mapping_status,
        pathway_status=pathway_status,
        comments_profile=comments_profile,
    )
    with (outputs / "processing_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
