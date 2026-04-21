"""Legacy KEGG pathway annotation support helpers.

These functions back shared loaders and utilities still reused by the active
AraCyc-first pipeline.
"""

from __future__ import annotations

import csv
from datetime import UTC, datetime
import gzip
import json
from functools import lru_cache
import math
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from process_chebi_to_pathways_v2 import (
    AGI_LOCUS_RE,
    DEVELOPMENT_KEYWORDS,
    GO_ID_RE,
    GO_TAIR_GAF_URLS,
    HORMONE_KEYWORDS,
    PATHWAY_LINE_RE,
    PLANT_CONTEXT_TAGS,
    PLANT_REACTOME_HIERARCHY_URL,
    PLANT_REACTOME_PARTICIPANTS_URL,
    PLANT_REACTOME_QUERY_URL,
    PRIMARY_BRITE_KEYWORDS,
    SECONDARY_BRITE_KEYWORDS,
    SECONDARY_NAME_KEYWORDS,
    STRESS_KEYWORDS,
    TRANSPORT_KEYWORDS,
    build_variants,
    clean_markup,
    normalize_name,
)

from pathway_pipeline.context import PipelineContext, Step5AnnotatedPathwayHit


def load_ath_pathways(path: Path) -> tuple[dict[str, str], dict[str, str]]:
    """Load ath pathway names and a mapXXXX -> athXXXX conversion table."""

    ath_pathways: dict[str, str] = {}
    map_to_ath: dict[str, str] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            pathway_id, raw_name = line.rstrip("\n").split("\t", 1)
            pathway_name = raw_name.split(" - ", 1)[0]
            ath_pathways[pathway_id] = pathway_name
            map_to_ath[f"map{pathway_id[3:]}"] = pathway_id
    return ath_pathways, map_to_ath


def load_map_pathways(path: Path) -> dict[str, str]:
    """Load KEGG reference pathway names keyed by map IDs."""

    map_pathways: dict[str, str] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            pathway_id, pathway_name = line.rstrip("\n").split("\t", 1)
            map_pathways[pathway_id] = pathway_name
    return map_pathways


def load_pathway_categories(path: Path) -> dict[str, tuple[str, str, str]]:
    """Parse KEGG pathway hierarchy into group/category/name triples."""

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


def load_ath_gene_counts(path: Path) -> dict[str, int]:
    """Count how many ath genes are linked to each ath pathway."""

    genes_by_pathway: defaultdict[str, set[str]] = defaultdict(set)
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            gene_ref, pathway_ref = line.rstrip("\n").split("\t", 1)
            pathway_id = pathway_ref.replace("path:", "")
            genes_by_pathway[pathway_id].add(gene_ref.replace("ath:", ""))
    return {pathway_id: len(genes) for pathway_id, genes in genes_by_pathway.items()}


def load_reactome_pathways(path: Path) -> dict[str, list[tuple[str, str]]]:
    """Index Reactome pathways by normalized name for explanation support."""

    index: defaultdict[str, list[tuple[str, str]]] = defaultdict(list)
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            pathway_id, pathway_name, species = line.rstrip("\n").split("\t", 2)
            normalized = normalize_name(pathway_name)
            if normalized:
                index[normalized].append((pathway_id, species))
    return dict(index)


def load_plantcyc_pathway_stats(files: list[tuple[str, Path]]) -> dict[str, dict[str, dict[str, object]]]:
    """Collect lightweight pathway support summaries from PMN exports."""

    stats = {"AraCyc": {}, "PlantCyc": {}}
    for source_db, path in files:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            for row in reader:
                pathway_name = row["Pathway-name"] or ""
                normalized = normalize_name(pathway_name)
                if not normalized:
                    continue
                entry = stats[source_db].setdefault(
                    normalized,
                    {"names": set(), "pathway_ids": set(), "gene_ids": set()},
                )
                entry["names"].add(pathway_name)
                if row["Pathway-id"]:
                    entry["pathway_ids"].add(row["Pathway-id"])
                if row["Gene-id"]:
                    entry["gene_ids"].add(row["Gene-id"])
    return stats


def fetch_text_with_fallback(url: str) -> str:
    """Fetch UTF-8 text with urllib first, then curl as a fallback."""

    try:
        with urlopen(url, timeout=120) as response:  # noqa: S310
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
        with urlopen(url, timeout=180) as response:  # noqa: S310
            destination.write_bytes(response.read())
            return
    except URLError:
        result = subprocess.run(
            ["curl", "-Lsf", url, "-o", str(destination)],
            check=True,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Failed to download {url} to {destination}")


def iso_mtime(path: Path) -> str:
    """Return a file's mtime in ISO-8601 UTC."""

    return datetime.fromtimestamp(path.stat().st_mtime, tz=UTC).isoformat()


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
    """Download Arabidopsis GO annotations when the local snapshot is missing."""

    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        errors = []
        for url in GO_TAIR_GAF_URLS:
            try:
                download_binary_with_fallback(url, path)
                break
            except Exception as exc:
                errors.append(f"{url}: {exc}")
        else:
            raise RuntimeError("Unable to download Arabidopsis GO annotations from any configured source:\n" + "\n".join(errors))
    return iso_mtime(path)


def load_gene_to_go_bp(gaf_path: Path, go_basic_path: Path) -> tuple[dict[str, set[tuple[str, str]]], dict[str, str], str]:
    """Load Arabidopsis gene -> GO BP mappings from a TAIR GAF snapshot."""

    go_names, go_namespaces = load_go_basic_terms(go_basic_path)
    gene_to_go: dict[str, set[tuple[str, str]]] = defaultdict(set)
    with gzip.open(gaf_path, "rt", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line or line.startswith("!"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 11:
                continue
            qualifier = fields[3]
            go_id = fields[4]
            aspect = fields[8]
            if aspect != "P" or not GO_ID_RE.fullmatch(go_id):
                continue
            qualifiers = {item.strip().upper() for item in qualifier.split("|") if item.strip()}
            if "NOT" in qualifiers:
                continue
            gene_ids = extract_agi_loci(fields[1], fields[2], fields[9], fields[10])
            if not gene_ids:
                continue
            go_name = go_names.get(go_id, go_id)
            for gene_id in gene_ids:
                gene_to_go[gene_id].add((go_id, go_name))
    return {gene: set(values) for gene, values in gene_to_go.items()}, go_namespaces, iso_mtime(gaf_path)


def build_go_term_gene_sets(gene_to_go: dict[str, set[tuple[str, str]]]) -> tuple[dict[str, set[str]], dict[str, str]]:
    """Invert a gene -> GO mapping into GO -> gene sets and GO names."""

    term_to_genes: dict[str, set[str]] = defaultdict(set)
    go_names: dict[str, str] = {}
    for gene_id, terms in gene_to_go.items():
        for go_id, go_name in terms:
            term_to_genes[go_id].add(gene_id)
            go_names[go_id] = go_name
    return {go_id: set(genes) for go_id, genes in term_to_genes.items()}, go_names


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
    """Compute GO BP enrichment for one pathway gene set."""

    pathway_genes = set(pathway_gene_ids) & background_genes
    gene_count = len(pathway_genes)
    background_gene_count = len(background_genes)
    if gene_count < 3 or background_gene_count == 0:
        return {
            "gene_count": gene_count,
            "background_gene_count": background_gene_count,
            "terms": [],
            "go_best_term": "",
            "go_best_fdr": "",
        }

    enrichment_rows = []
    for go_id, genes in term_to_genes.items():
        background_term_genes = genes & background_genes
        if not background_term_genes:
            continue
        hit_genes = pathway_genes & background_term_genes
        if not hit_genes:
            continue
        p_value = hypergeom_sf(
            population_size=background_gene_count,
            success_population=len(background_term_genes),
            draw_count=gene_count,
            observed_successes=len(hit_genes),
        )
        enrichment_rows.append(
            {
                "go_id": go_id,
                "go_name": go_names.get(go_id, go_id),
                "p_value": p_value,
                "n_genes_hit": len(hit_genes),
                "term_gene_count": len(background_term_genes),
            }
        )
    if not enrichment_rows:
        return {
            "gene_count": gene_count,
            "background_gene_count": background_gene_count,
            "terms": [],
            "go_best_term": "",
            "go_best_fdr": "",
        }

    adjusted = benjamini_hochberg([row["p_value"] for row in enrichment_rows])
    for row, fdr in zip(enrichment_rows, adjusted, strict=False):
        row["fdr"] = fdr
    enrichment_rows.sort(key=lambda row: (row["fdr"], -row["n_genes_hit"], row["go_name"]))
    top_terms = enrichment_rows[:top_k]
    return {
        "gene_count": gene_count,
        "background_gene_count": background_gene_count,
        "terms": top_terms,
        "go_best_term": top_terms[0]["go_name"] if top_terms else "",
        "go_best_fdr": top_terms[0]["fdr"] if top_terms else "",
    }


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
    """Download and materialize a local Plant Reactome snapshot if missing.

    Plant Reactome is an auxiliary evidence source in this project, not a
    primary label space. This helper snapshots just enough pathway metadata and
    pathway->gene links to support annotation-time alignment and reporting while
    keeping later steps fully local and reproducible.
    """

    if pathways_path.exists() and gene_path.exists() and version_path.exists():
        raw_text = version_path.read_text(encoding="utf-8").strip()
        try:
            payload = json.loads(raw_text)
            return payload.get("vstamp", raw_text)
        except json.JSONDecodeError:
            return raw_text

    hierarchy = fetch_json_with_fallback(PLANT_REACTOME_HIERARCHY_URL)
    pathway_rows = flatten_plant_reactome_hierarchy(hierarchy)
    gene_rows = []
    fetched_at = datetime.now(tz=UTC).isoformat()
    release_dates: set[str] = set()

    with pathways_path.open("w", newline="", encoding="utf-8") as pathway_handle, gene_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as gene_handle:
        pathway_writer = csv.DictWriter(
            pathway_handle,
            fieldnames=[
                "pathway_id",
                "pathway_name",
                "species",
                "top_level_category",
                "description",
                "release_date",
                "go_biological_process_id",
                "go_biological_process_name",
                "name_normalized",
                "name_compact",
            ],
            delimiter="\t",
        )
        gene_writer = csv.DictWriter(
            gene_handle,
            fieldnames=["pathway_id", "gene_id"],
            delimiter="\t",
        )
        pathway_writer.writeheader()
        gene_writer.writeheader()

        for pathway_id, base_row in sorted(pathway_rows.items()):
            detail = fetch_json_with_fallback(PLANT_REACTOME_QUERY_URL.format(st_id=pathway_id))
            participants = fetch_json_with_fallback(PLANT_REACTOME_PARTICIPANTS_URL.format(st_id=pathway_id))
            release_date = detail.get("releaseDate", "")
            if release_date:
                release_dates.add(release_date)
            go_process = detail.get("goBiologicalProcess") or {}
            summaries = detail.get("summation") or []
            description = ""
            if summaries:
                description = clean_markup(summaries[0].get("text", ""))
            if not description:
                description = clean_markup(go_process.get("definition", ""))
            name_value = detail.get("displayName") or base_row["pathway_name"]
            variants = build_variants(name_value)
            pathway_writer.writerow(
                {
                    "pathway_id": pathway_id,
                    "pathway_name": name_value,
                    "species": detail.get("speciesName") or base_row["species"],
                    "top_level_category": base_row["top_level_category"],
                    "description": description,
                    "release_date": release_date,
                    "go_biological_process_id": go_process.get("accession", ""),
                    "go_biological_process_name": go_process.get("displayName", ""),
                    "name_normalized": variants["exact"],
                    "name_compact": variants["compact"],
                }
            )
            for gene_id in sorted(extract_identifier_strings(participants)):
                gene_writer.writerow({"pathway_id": pathway_id, "gene_id": gene_id})
                gene_rows.append((pathway_id, gene_id))

    primary_release = sorted(release_dates)[-1] if release_dates else fetched_at[:10]
    vstamp = f"plant_reactome_{normalize_name(primary_release).replace(' ', '-') or fetched_at[:10]}"
    version_text = json.dumps(
        {
            "source": "plant_reactome_content_service",
            "fetched_at": fetched_at,
            "release_dates": sorted(release_dates),
            "pathway_count": len(pathway_rows),
            "gene_link_count": len(gene_rows),
            "vstamp": vstamp,
        },
        ensure_ascii=False,
    )
    version_path.write_text(version_text, encoding="utf-8")
    return vstamp


def load_plant_reactome_pathways(path: Path) -> dict[str, dict[str, str]]:
    """Load the normalized Plant Reactome pathway snapshot."""

    pathways = {}
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            pathways[row["pathway_id"]] = row
    return pathways


def load_plant_reactome_gene_index(path: Path) -> dict[str, set[str]]:
    """Load Plant Reactome pathway -> gene associations."""

    gene_index: dict[str, set[str]] = defaultdict(set)
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            if row["gene_id"]:
                gene_index[row["pathway_id"]].add(row["gene_id"].upper())
    return {pathway_id: set(genes) for pathway_id, genes in gene_index.items()}


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
    """Infer compact plant-context tags from BRITE/name/PMN/Plant Reactome text."""

    normalized_text = " ".join(
        normalize_name(text)
        for text in (brite_l1, brite_l2, brite_l3, pathway_name, *plant_reactome_texts, *go_texts)
        if text
    )
    tags: list[str] = []
    brite_l2_norm = normalize_name(brite_l2)
    if normalize_name(brite_l1) == "metabolism":
        if any(keyword in brite_l2_norm for keyword in PRIMARY_BRITE_KEYWORDS):
            tags.append("primary_metabolism")
        if any(keyword in brite_l2_norm for keyword in SECONDARY_BRITE_KEYWORDS) or any(keyword in normalized_text for keyword in SECONDARY_NAME_KEYWORDS):
            tags.append("secondary_metabolism")
    if any(keyword in normalized_text for keyword in HORMONE_KEYWORDS):
        tags.append("hormone_related")
    if any(keyword in normalized_text for keyword in STRESS_KEYWORDS):
        tags.append("stress_related")
    if any(keyword in normalized_text for keyword in DEVELOPMENT_KEYWORDS):
        tags.append("development_related")
    if any(keyword in normalized_text for keyword in TRANSPORT_KEYWORDS):
        tags.append("transport_related")
    if not tags and has_aracyc and normalize_name(brite_l1) == "metabolism":
        tags.append("primary_metabolism")
    elif not tags and has_plantcyc and normalize_name(brite_l1) == "metabolism":
        tags.append("primary_metabolism")
    return tuple(tag for tag in PLANT_CONTEXT_TAGS if tag in set(tags))


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
    """Match one KEGG pathway against Plant Reactome by name and gene overlap.

    The output is meant for external-support annotation, not strict equivalence.
    We surface the most plausible Plant Reactome companions so downstream
    reports can say whether a KEGG pathway is also supported by another
    Arabidopsis-focused pathway resource.
    """

    normalized_name = normalize_name(pathway_name)
    compact_name = build_variants(pathway_name)["compact"]
    ranked = []
    for pathway_id, row in plant_reactome_pathways.items():
        reactome_name = row["pathway_name"]
        reactome_name_norm = row.get("name_normalized", normalize_name(reactome_name))
        reactome_name_compact = row.get("name_compact", build_variants(reactome_name)["compact"])
        reactome_genes = plant_reactome_gene_sets.get(pathway_id, set())
        overlap_count = len(gene_ids & reactome_genes)
        gene_union = len(gene_ids | reactome_genes)
        gene_jaccard = overlap_count / gene_union if gene_union else 0.0
        overlap_ratio_kegg = overlap_count / len(gene_ids) if gene_ids else 0.0
        if normalized_name and normalized_name == reactome_name_norm:
            name_similarity = 1.0
        elif compact_name and compact_name == reactome_name_compact:
            name_similarity = 0.97
        else:
            name_similarity = token_sorensen_dice(pathway_name, reactome_name)
        arabidopsis_bonus = 1.0 if row.get("species", "") == "Arabidopsis thaliana" else 0.0
        category_bonus = plant_reactome_category_bonus(
            brite_l1=brite_l1,
            brite_l2=brite_l2,
            brite_l3=brite_l3,
            pathway_name=pathway_name,
            reactome_name=reactome_name,
            reactome_category=row.get("top_level_category", ""),
            reactome_description=row.get("description", ""),
        )
        if not (
            (normalized_name and normalized_name == reactome_name_norm)
            or (compact_name and compact_name == reactome_name_compact)
            or overlap_count >= 2
            or gene_jaccard >= 0.05
        ):
            continue
        # Gene overlap carries most of the score; name and category agreement
        # only refine otherwise plausible plant-context matches.
        alignment_score = (
            0.50 * gene_jaccard
            + 0.20 * overlap_ratio_kegg
            + 0.15 * name_similarity
            + 0.10 * arabidopsis_bonus
            + 0.05 * category_bonus
        )
        if alignment_score >= 0.55 and overlap_count >= 3:
            confidence = "high"
        elif alignment_score >= 0.35 and overlap_count >= 2:
            confidence = "medium"
        else:
            confidence = "low"
        ranked.append(
            {
                "plant_reactome_id": pathway_id,
                "pathway_name": reactome_name,
                "species": row.get("species", ""),
                "top_level_category": row.get("top_level_category", ""),
                "description": row.get("description", ""),
                "alignment_score": round(alignment_score, 4),
                "alignment_confidence": confidence,
                "gene_overlap_count": overlap_count,
                "gene_jaccard": round(gene_jaccard, 4),
                "overlap_ratio_kegg": round(overlap_ratio_kegg, 4),
                "name_similarity": round(name_similarity, 4),
            }
        )
    ranked.sort(
        key=lambda item: (
            float(item["alignment_score"]),
            int(item["gene_overlap_count"]),
            item["species"] == "Arabidopsis thaliana",
            item["pathway_name"],
        ),
        reverse=True,
    )
    return ranked[:top_k]


def build_annotation_confidence(
    *,
    brite_l1: str,
    go_best_term: str,
    aracyc_evidence_score: float,
    reactome_matches: tuple[tuple[str, str], ...],
    plant_evidence_sources: tuple[str, ...],
    plant_reactome_alignment_confidence: str,
) -> str:
    """Assign a simple confidence level to the final annotation bundle.

    This confidence describes how richly supported the annotation payload is,
    not whether the pathway itself is "correct". Independent evidence layers
    such as BRITE, GO BP, PMN, and Plant Reactome push the bundle upward.
    """

    if brite_l1 and (
        go_best_term
        or aracyc_evidence_score > 0
        or plant_reactome_alignment_confidence == "high"
    ):
        return "high"
    if brite_l1 and (
        reactome_matches
        or plant_evidence_sources
        or plant_reactome_alignment_confidence == "medium"
    ):
        return "medium"
    return "low"


def load_ath_gene_sets(path: Path) -> dict[str, set[str]]:
    """Load ath pathway -> AGI gene sets from the KEGG ath gene table."""

    gene_sets: dict[str, set[str]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            gene_ref, pathway_ref = line.rstrip("\n").split("\t", 1)
            pathway_id = pathway_ref.replace("path:", "")
            gene_id = gene_ref.replace("ath:", "").upper()
            gene_sets.setdefault(pathway_id, set()).add(gene_id)
    return gene_sets


def write_pathway_annotation_index(rows: list[dict[str, object]], context: PipelineContext) -> int:
    """Write the final Step 5 pathway annotation index."""

    fieldnames = [
        "pathway_id",
        "pathway_target_type",
        "map_pathway_id",
        "ath_pathway_id",
        "pathway_name",
        "map_id",
        "kegg_name",
        "brite_l1",
        "brite_l2",
        "brite_l3",
        "pathway_group",
        "pathway_category",
        "kegg_vstamp",
        "map_pathway_compound_count",
        "ath_gene_count",
        "go_top_terms_json",
        "go_best_term",
        "go_best_fdr",
        "go_vstamp",
        "plant_context_tags",
        "plant_evidence_sources",
        "aracyc_evidence_score",
        "reactome_matches",
        "plant_reactome_matches_json",
        "plant_reactome_best_id",
        "plant_reactome_best_name",
        "plant_reactome_best_category",
        "plant_reactome_best_description",
        "plant_reactome_alignment_score",
        "plant_reactome_alignment_confidence",
        "plant_reactome_tags",
        "plant_reactome_vstamp",
        "annotation_confidence",
    ]
    with context.paths.pathway_annotation_index_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def write_pathway_go_enrichment(context: PipelineContext) -> int:
    """Persist pathway-level GO enrichment results."""

    fieldnames = [
        "pathway_id",
        "map_id",
        "ath_id",
        "gene_count",
        "background_gene_count",
        "go_top_terms_json",
        "go_best_term",
        "go_best_fdr",
        "go_vstamp",
    ]
    rows = []
    for ath_id, payload in sorted(context.pathway_go_enrichment.items()):
        rows.append(
            {
                "pathway_id": ath_id,
                "map_id": f"map{ath_id[3:]}" if ath_id.startswith("ath") else "",
                "ath_id": ath_id,
                "gene_count": payload["gene_count"],
                "background_gene_count": payload["background_gene_count"],
                "go_top_terms_json": json.dumps(payload["terms"], ensure_ascii=False, sort_keys=True),
                "go_best_term": payload["go_best_term"],
                "go_best_fdr": f"{payload['go_best_fdr']:.6g}" if payload["go_best_fdr"] != "" else "",
                "go_vstamp": context.go_vstamp,
            }
        )
    with context.paths.pathway_go_enrichment_index_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def write_gene_to_go_index(context: PipelineContext) -> int:
    """Persist the preprocessed AGI -> GO BP index used in Step 5."""

    rows = []
    with context.paths.gene_to_go_index_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["gene_id", "go_id", "go_name", "namespace", "go_vstamp"],
            delimiter="\t",
        )
        writer.writeheader()
        for gene_id in sorted(context.go_gene_index):
            for go_id, go_name in sorted(context.go_gene_index[gene_id]):
                rows.append(
                    {
                        "gene_id": gene_id,
                        "go_id": go_id,
                        "go_name": go_name,
                        "namespace": context.go_term_namespace.get(go_id, ""),
                        "go_vstamp": context.go_vstamp,
                    }
                )
        writer.writerows(rows)
    return len(rows)


def write_plant_reactome_indexes(context: PipelineContext) -> tuple[int, int]:
    """Persist normalized Plant Reactome pathway and gene-set tables."""

    pathway_rows = []
    with context.paths.plant_reactome_pathway_index_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "pathway_id",
                "pathway_name",
                "species",
                "top_level_category",
                "description",
                "release_date",
                "go_biological_process_id",
                "go_biological_process_name",
                "name_normalized",
                "name_compact",
                "plant_reactome_vstamp",
            ],
            delimiter="\t",
        )
        writer.writeheader()
        for pathway_id, row in sorted(context.plant_reactome_pathways.items()):
            payload = dict(row)
            payload["plant_reactome_vstamp"] = context.plant_reactome_vstamp
            pathway_rows.append(payload)
        writer.writerows(pathway_rows)

    gene_rows = []
    with context.paths.plant_reactome_gene_index_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["pathway_id", "gene_id", "plant_reactome_vstamp"],
            delimiter="\t",
        )
        writer.writeheader()
        for pathway_id, genes in sorted(context.plant_reactome_gene_sets.items()):
            for gene_id in sorted(genes):
                gene_rows.append(
                    {
                        "pathway_id": pathway_id,
                        "gene_id": gene_id,
                        "plant_reactome_vstamp": context.plant_reactome_vstamp,
                    }
                )
        writer.writerows(gene_rows)
    return len(pathway_rows), len(gene_rows)


def write_plant_reactome_alignment(context: PipelineContext) -> int:
    """Persist KEGG pathway -> Plant Reactome alignment candidates."""

    rows = []
    with context.paths.plant_reactome_alignment_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "pathway_id",
                "map_id",
                "ath_id",
                "plant_reactome_id",
                "plant_reactome_name",
                "species",
                "top_level_category",
                "description",
                "alignment_score",
                "alignment_confidence",
                "gene_overlap_count",
                "gene_jaccard",
                "overlap_ratio_kegg",
                "name_similarity",
                "plant_reactome_vstamp",
            ],
            delimiter="\t",
        )
        writer.writeheader()
        for pathway_id, matches in sorted(context.plant_reactome_alignments.items()):
            ath_id = pathway_id if pathway_id.startswith("ath") else ""
            map_id = f"map{ath_id[3:]}" if ath_id else pathway_id
            for match in matches:
                rows.append(
                    {
                        "pathway_id": pathway_id,
                        "map_id": map_id,
                        "ath_id": ath_id,
                        "plant_reactome_id": match["plant_reactome_id"],
                        "plant_reactome_name": match["pathway_name"],
                        "species": match["species"],
                        "top_level_category": match["top_level_category"],
                        "description": match["description"],
                        "alignment_score": f"{match['alignment_score']:.4f}",
                        "alignment_confidence": match["alignment_confidence"],
                        "gene_overlap_count": match["gene_overlap_count"],
                        "gene_jaccard": f"{match['gene_jaccard']:.4f}",
                        "overlap_ratio_kegg": f"{match['overlap_ratio_kegg']:.4f}",
                        "name_similarity": f"{match['name_similarity']:.4f}",
                        "plant_reactome_vstamp": context.plant_reactome_vstamp,
                    }
                )
        writer.writerows(rows)
    return len(rows)


def write_plant_evidence_index(context: PipelineContext) -> int:
    """Write PMN support used for KEGG boosting and direct PMN fallback."""

    row_count = 0
    with context.paths.plant_evidence_index_path.open("w", newline="", encoding="utf-8") as handle:
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
        for compound_id in sorted(context.compounds, key=int):
            compound = context.compounds[compound_id]
            compound_context = context.compound_contexts[compound_id]

            for normalized_name, names in sorted(compound_context.aracyc_pathways.items()):
                pathway_stat = context.plantcyc_pathway_stats["AraCyc"].get(normalized_name, {})
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
                pathway_stat = context.plantcyc_pathway_stats["PlantCyc"].get(normalized_name, {})
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


def build_annotation_row(
    *,
    context: PipelineContext,
    pathway_id: str,
    pathway_target_type: str,
    map_pathway_id: str,
    ath_pathway_id: str,
    pathway_name: str,
    ath_gene_sets: dict[str, set[str]],
) -> tuple[dict[str, object], Step5AnnotatedPathwayHit]:
    """Build one fully annotated pathway record and its in-memory counterpart.

    Step 5 feeds two consumers:
    1. persisted pathway-level TSV indexes
    2. per-hit in-memory objects attached to resolved compound->pathway hits
    This helper builds both from a single annotation bundle so those views stay
    synchronized.
    """

    brite_l1, brite_l2, brite_l3 = context.pathway_categories.get(map_pathway_id, ("", "", ""))
    normalized_name = normalize_name(pathway_name)
    reactome_matches = tuple(context.reactome_pathways.get(normalized_name, [])[:3])
    has_aracyc = normalized_name in context.plantcyc_pathway_stats["AraCyc"]
    has_plantcyc = normalized_name in context.plantcyc_pathway_stats["PlantCyc"]
    plant_evidence_sources = tuple(
        source
        for source, enabled in (("AraCyc", has_aracyc), ("PlantCyc", has_plantcyc))
        if enabled
    )
    aracyc_evidence_score = 1.0 if has_aracyc else 0.5 if has_plantcyc else 0.0

    if ath_pathway_id:
        go_payload = context.pathway_go_enrichment.get(
            ath_pathway_id,
            {
                "terms": [],
                "go_best_term": "",
                "go_best_fdr": "",
                "gene_count": 0,
                "background_gene_count": 0,
            },
        )
        gene_ids = ath_gene_sets.get(ath_pathway_id, set())
    else:
        go_payload = {
            "terms": [],
            "go_best_term": "",
            "go_best_fdr": "",
            "gene_count": 0,
            "background_gene_count": 0,
        }
        gene_ids = set()

    # Plant Reactome remains an auxiliary support layer. We keep a small ranked
    # list so later reporting can explain cross-database agreement without
    # changing the upstream KEGG/AraCyc pathway identity.
    plant_reactome_matches = match_plant_reactome_context(
        pathway_name=pathway_name,
        gene_ids=gene_ids,
        brite_l1=brite_l1,
        brite_l2=brite_l2,
        brite_l3=brite_l3,
        plant_reactome_pathways=context.plant_reactome_pathways,
        plant_reactome_gene_sets=context.plant_reactome_gene_sets,
    )
    context.plant_reactome_alignments[pathway_id] = plant_reactome_matches
    top_reactome = plant_reactome_matches[0] if plant_reactome_matches else {}
    plant_reactome_tags = infer_plant_context_tags(
        brite_l1="",
        brite_l2=top_reactome.get("top_level_category", ""),
        brite_l3="",
        pathway_name=top_reactome.get("pathway_name", ""),
        plant_reactome_texts=(top_reactome.get("description", ""),),
    )
    go_texts = tuple(item["go_name"] for item in go_payload["terms"])
    plant_context_tags = infer_plant_context_tags(
        brite_l1=brite_l1,
        brite_l2=brite_l2,
        brite_l3=brite_l3,
        pathway_name=pathway_name,
        has_aracyc=has_aracyc,
        has_plantcyc=has_plantcyc,
        plant_reactome_texts=(
            top_reactome.get("pathway_name", ""),
            top_reactome.get("top_level_category", ""),
            top_reactome.get("description", ""),
        ),
        go_texts=go_texts,
    )
    annotation_confidence = build_annotation_confidence(
        brite_l1=brite_l1,
        go_best_term=go_payload["go_best_term"],
        aracyc_evidence_score=aracyc_evidence_score,
        reactome_matches=reactome_matches,
        plant_evidence_sources=plant_evidence_sources,
        plant_reactome_alignment_confidence=top_reactome.get("alignment_confidence", ""),
    )

    row = {
        "pathway_id": pathway_id,
        "pathway_target_type": pathway_target_type,
        "map_pathway_id": map_pathway_id,
        "ath_pathway_id": ath_pathway_id,
        "pathway_name": pathway_name,
        "map_id": map_pathway_id,
        "kegg_name": pathway_name,
        "brite_l1": brite_l1,
        "brite_l2": brite_l2,
        "brite_l3": brite_l3 or pathway_name,
        "pathway_group": brite_l1,
        "pathway_category": brite_l2,
        "kegg_vstamp": context.step3_vstamp or "unknown",
        "map_pathway_compound_count": context.map_pathway_compound_counts.get(map_pathway_id, 0),
        "ath_gene_count": context.ath_gene_counts.get(ath_pathway_id, 0) if ath_pathway_id else 0,
        "go_top_terms_json": json.dumps(go_payload["terms"], ensure_ascii=False, sort_keys=True),
        "go_best_term": go_payload["go_best_term"],
        "go_best_fdr": f"{go_payload['go_best_fdr']:.6g}" if go_payload["go_best_fdr"] != "" else "",
        "go_vstamp": context.go_vstamp,
        "plant_context_tags": ";".join(plant_context_tags),
        "plant_evidence_sources": ";".join(plant_evidence_sources),
        "aracyc_evidence_score": f"{aracyc_evidence_score:.3f}",
        "reactome_matches": ";".join(f"{pathway_ref}|{species}" for pathway_ref, species in reactome_matches),
        "plant_reactome_matches_json": json.dumps(plant_reactome_matches, ensure_ascii=False, sort_keys=True),
        "plant_reactome_best_id": top_reactome.get("plant_reactome_id", ""),
        "plant_reactome_best_name": top_reactome.get("pathway_name", ""),
        "plant_reactome_best_category": top_reactome.get("top_level_category", ""),
        "plant_reactome_best_description": top_reactome.get("description", ""),
        "plant_reactome_alignment_score": f"{top_reactome.get('alignment_score', 0.0):.4f}" if top_reactome else "",
        "plant_reactome_alignment_confidence": top_reactome.get("alignment_confidence", ""),
        "plant_reactome_tags": ";".join(plant_reactome_tags),
        "plant_reactome_vstamp": context.plant_reactome_vstamp,
        "annotation_confidence": annotation_confidence,
    }
    return row, Step5AnnotatedPathwayHit(
        compound_id="",
        mapping=None,  # placeholder, replaced for per-hit rows below
        map_pathway_id=map_pathway_id,
        ath_pathway_id=ath_pathway_id,
        pathway_target_id=pathway_id,
        pathway_target_type=pathway_target_type,
        pathway_name=pathway_name,
        map_id=map_pathway_id,
        kegg_name=pathway_name,
        brite_l1=brite_l1,
        brite_l2=brite_l2,
        brite_l3=brite_l3 or pathway_name,
        kegg_vstamp=context.step3_vstamp or "unknown",
        pathway_group=brite_l1,
        pathway_category=brite_l2,
        map_pathway_compound_count=context.map_pathway_compound_counts.get(map_pathway_id, 0),
        ath_gene_count=context.ath_gene_counts.get(ath_pathway_id, 0) if ath_pathway_id else 0,
        go_top_terms_json=json.dumps(go_payload["terms"], ensure_ascii=False, sort_keys=True),
        go_best_term=go_payload["go_best_term"],
        go_best_fdr=float(go_payload["go_best_fdr"]) if go_payload["go_best_fdr"] != "" else 0.0,
        go_vstamp=context.go_vstamp,
        plant_context_tags=plant_context_tags,
        plant_evidence_sources=plant_evidence_sources,
        aracyc_evidence_score=aracyc_evidence_score,
        reactome_matches=reactome_matches,
        plant_reactome_matches_json=json.dumps(plant_reactome_matches, ensure_ascii=False, sort_keys=True),
        plant_reactome_best_id=top_reactome.get("plant_reactome_id", ""),
        plant_reactome_best_name=top_reactome.get("pathway_name", ""),
        plant_reactome_best_category=top_reactome.get("top_level_category", ""),
        plant_reactome_best_description=top_reactome.get("description", ""),
        plant_reactome_alignment_score=float(top_reactome.get("alignment_score", 0.0)) if top_reactome else 0.0,
        plant_reactome_alignment_confidence=top_reactome.get("alignment_confidence", ""),
        plant_reactome_tags=plant_reactome_tags,
        plant_reactome_vstamp=context.plant_reactome_vstamp,
        annotation_confidence=annotation_confidence,
        relation_vstamp="",
        direct_link=False,
        support_reaction_count=0,
        support_rids=(),
        has_substrate_role=False,
        has_product_role=False,
        has_both_role=False,
        cofactor_like=False,
        role_summary="",
        reaction_role_score=0.0,
    )


def run(context: PipelineContext) -> PipelineContext:
    """Add pathway metadata needed for scoring and reporting.

    This step does not choose pathways. It enriches already-resolved pathway
    hits with KEGG BRITE labels, Arabidopsis GO BP summaries, PMN plant
    evidence, and Plant Reactome alignments so later ranking, review, and
    presentation layers can explain why a pathway is relevant.
    """

    ensure_go_annotations(context.paths.gene_association_tair_path)
    context.go_vstamp = iso_mtime(context.paths.gene_association_tair_path)
    context.plant_reactome_vstamp = ensure_plant_reactome_refs(
        context.paths.plant_reactome_pathways_path,
        context.paths.plant_reactome_gene_pathway_path,
        context.paths.plant_reactome_version_path,
    )

    context.map_pathways = load_map_pathways(context.paths.kegg_pathway_map_path)
    context.pathway_categories = load_pathway_categories(context.paths.kegg_pathway_hierarchy_path)
    context.ath_gene_counts = load_ath_gene_counts(context.paths.kegg_ath_gene_pathway_path)
    context.reactome_pathways = load_reactome_pathways(context.paths.reactome_pathways_path)
    context.plantcyc_pathway_stats = load_plantcyc_pathway_stats(
        [
            ("AraCyc", context.paths.aracyc_pathways_path),
            ("PlantCyc", context.paths.plantcyc_pathways_path),
        ]
    )
    context.go_gene_index, context.go_term_namespace, context.go_vstamp = load_gene_to_go_bp(
        context.paths.gene_association_tair_path,
        context.paths.go_basic_obo_path,
    )
    term_to_genes, go_names = build_go_term_gene_sets(context.go_gene_index)
    context.plant_reactome_pathways = load_plant_reactome_pathways(context.paths.plant_reactome_pathways_path)
    context.plant_reactome_gene_sets = load_plant_reactome_gene_index(context.paths.plant_reactome_gene_pathway_path)

    # GO enrichment is only meaningful for ath pathways that actually have gene
    # membership, so we build a KEGG-ath-specific background before annotating
    # individual pathways.
    ath_gene_sets = load_ath_gene_sets(context.paths.kegg_ath_gene_pathway_path)
    kegg_ath_genes = set()
    for genes in ath_gene_sets.values():
        kegg_ath_genes.update(genes)
    background_genes = set(context.go_gene_index) & kegg_ath_genes

    pathway_go_enrichment = {}
    for ath_id, gene_ids in ath_gene_sets.items():
        pathway_go_enrichment[ath_id] = compute_pathway_go_enrichment(
            gene_ids,
            term_to_genes,
            go_names,
            background_genes,
        )
    context.pathway_go_enrichment = pathway_go_enrichment

    annotation_rows = []
    annotation_lookup: dict[str, Step5AnnotatedPathwayHit] = {}
    map_ids = sorted(set(context.map_pathways) | set(context.map_to_ath))
    for map_pathway_id in map_ids:
        # Every map pathway gets a fallback annotation row so generic KEGG map
        # IDs remain explainable even when no ath-specific projection exists.
        map_name = context.map_pathways.get(map_pathway_id, map_pathway_id)
        row, template = build_annotation_row(
            context=context,
            pathway_id=map_pathway_id,
            pathway_target_type="map_fallback",
            map_pathway_id=map_pathway_id,
            ath_pathway_id="",
            pathway_name=map_name,
            ath_gene_sets=ath_gene_sets,
        )
        annotation_rows.append(row)
        annotation_lookup[map_pathway_id] = template

        ath_pathway_id = context.map_to_ath.get(map_pathway_id, "")
        if not ath_pathway_id:
            continue
        ath_name = context.ath_pathways.get(ath_pathway_id, ath_pathway_id)
        row, template = build_annotation_row(
            context=context,
            pathway_id=ath_pathway_id,
            pathway_target_type="ath",
            map_pathway_id=map_pathway_id,
            ath_pathway_id=ath_pathway_id,
            pathway_name=ath_name,
            ath_gene_sets=ath_gene_sets,
        )
        annotation_rows.append(row)
        annotation_lookup[ath_pathway_id] = template

    # Copy the pathway-level annotation template onto each compound-level hit
    # while preserving step-4 relation evidence such as reaction roles.
    annotated_hits: dict[str, list[Step5AnnotatedPathwayHit]] = {}
    for compound_id, hits in context.resolved_pathway_hits.items():
        annotated = []
        for hit in hits:
            template = annotation_lookup[hit.pathway_target_id]
            annotated.append(
                Step5AnnotatedPathwayHit(
                    compound_id=compound_id,
                    mapping=hit.mapping,
                    map_pathway_id=hit.map_pathway_id,
                    ath_pathway_id=hit.ath_pathway_id,
                    pathway_target_id=hit.pathway_target_id,
                    pathway_target_type=hit.pathway_target_type,
                    pathway_name=template.pathway_name,
                    map_id=template.map_id,
                    kegg_name=template.kegg_name,
                    brite_l1=template.brite_l1,
                    brite_l2=template.brite_l2,
                    brite_l3=template.brite_l3,
                    kegg_vstamp=template.kegg_vstamp,
                    pathway_group=template.pathway_group,
                    pathway_category=template.pathway_category,
                    map_pathway_compound_count=template.map_pathway_compound_count,
                    ath_gene_count=template.ath_gene_count,
                    go_top_terms_json=template.go_top_terms_json,
                    go_best_term=template.go_best_term,
                    go_best_fdr=template.go_best_fdr,
                    go_vstamp=template.go_vstamp,
                    plant_context_tags=template.plant_context_tags,
                    plant_evidence_sources=template.plant_evidence_sources,
                    aracyc_evidence_score=template.aracyc_evidence_score,
                    reactome_matches=template.reactome_matches,
                    plant_reactome_matches_json=template.plant_reactome_matches_json,
                    plant_reactome_best_id=template.plant_reactome_best_id,
                    plant_reactome_best_name=template.plant_reactome_best_name,
                    plant_reactome_best_category=template.plant_reactome_best_category,
                    plant_reactome_best_description=template.plant_reactome_best_description,
                    plant_reactome_alignment_score=template.plant_reactome_alignment_score,
                    plant_reactome_alignment_confidence=template.plant_reactome_alignment_confidence,
                    plant_reactome_tags=template.plant_reactome_tags,
                    plant_reactome_vstamp=template.plant_reactome_vstamp,
                    annotation_confidence=template.annotation_confidence,
                    relation_vstamp=hit.relation_vstamp,
                    direct_link=hit.direct_link,
                    support_reaction_count=hit.support_reaction_count,
                    support_rids=hit.support_rids,
                    has_substrate_role=hit.has_substrate_role,
                    has_product_role=hit.has_product_role,
                    has_both_role=hit.has_both_role,
                    cofactor_like=hit.cofactor_like,
                    role_summary=hit.role_summary,
                    reaction_role_score=hit.reaction_role_score,
                )
            )
        annotated_hits[compound_id] = annotated

    context.annotated_pathway_hits = annotated_hits
    context.preprocess_counts["pathway_annotation_index"] = write_pathway_annotation_index(annotation_rows, context)
    context.preprocess_counts["pathway_go_enrichment"] = write_pathway_go_enrichment(context)
    context.preprocess_counts["gene_to_go_index"] = write_gene_to_go_index(context)
    plant_reactome_pathway_count, plant_reactome_gene_count = write_plant_reactome_indexes(context)
    context.preprocess_counts["plant_reactome_pathway_index"] = plant_reactome_pathway_count
    context.preprocess_counts["plant_reactome_gene_index"] = plant_reactome_gene_count
    context.preprocess_counts["plant_reactome_alignment"] = write_plant_reactome_alignment(context)
    if context.compounds and context.compound_contexts:
        context.preprocess_counts["plant_evidence_index"] = write_plant_evidence_index(context)
    elif context.paths.plant_evidence_index_path.exists():
        with context.paths.plant_evidence_index_path.open(encoding="utf-8", newline="") as handle:
            existing_rows = max(sum(1 for _ in handle) - 1, 0)
        context.preprocess_counts["plant_evidence_index"] = existing_rows
        context.add_note("Step 5 kept the existing plant_evidence_index because compound-level PMN contexts were not loaded.")
    else:
        context.preprocess_counts["plant_evidence_index"] = 0
        context.add_note("Step 5 skipped plant_evidence_index because compound-level PMN contexts were not loaded.")
    context.add_note(
        "Step 5 enriches pathways with KEGG BRITE, Arabidopsis GO BP enrichment, PMN plant evidence, and Plant Reactome local alignment context."
    )
    return context
