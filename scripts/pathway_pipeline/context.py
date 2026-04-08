"""Shared state containers for the refactored step-wise pathway pipeline."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from process_chebi_to_pathways_v2 import (
    AliasRecord,
    CandidateMapping,
    ChEBICompound,
    CompoundContext,
    KeggCompound,
    LipidMapsRecord,
    PlantToKeggBridge,
    PlantCycCompound,
    StoredSupportContext,
    StructureInfo,
    XrefInfo,
    ensure_exists,
)


@dataclass(slots=True)
class Step3PathwayHit:
    """Raw KEGG compound -> map pathway expansion before ath resolution."""

    compound_id: str
    mapping: CandidateMapping
    map_pathway_id: str
    relation_vstamp: str
    direct_link: bool
    support_reaction_count: int
    support_rids: tuple[str, ...]
    has_substrate_role: bool
    has_product_role: bool
    has_both_role: bool
    cofactor_like: bool
    role_summary: str
    reaction_role_score: float


@dataclass(slots=True)
class Step4ResolvedPathwayHit:
    """Pathway hit after ath fallback resolution has been applied."""

    compound_id: str
    mapping: CandidateMapping
    map_pathway_id: str
    ath_pathway_id: str
    pathway_target_id: str
    pathway_target_type: str
    relation_vstamp: str
    direct_link: bool
    support_reaction_count: int
    support_rids: tuple[str, ...]
    has_substrate_role: bool
    has_product_role: bool
    has_both_role: bool
    cofactor_like: bool
    role_summary: str
    reaction_role_score: float


@dataclass(slots=True)
class Step5AnnotatedPathwayHit:
    """Pathway hit with names, categories, gene support, and Reactome context."""

    compound_id: str
    mapping: CandidateMapping
    map_pathway_id: str
    ath_pathway_id: str
    pathway_target_id: str
    pathway_target_type: str
    pathway_name: str
    pathway_group: str
    pathway_category: str
    map_pathway_compound_count: int
    ath_gene_count: int
    reactome_matches: tuple[tuple[str, str], ...]
    relation_vstamp: str
    direct_link: bool
    support_reaction_count: int
    support_rids: tuple[str, ...]
    has_substrate_role: bool
    has_product_role: bool
    has_both_role: bool
    cofactor_like: bool
    role_summary: str
    reaction_role_score: float


@dataclass(slots=True)
class RankedPathwayRow:
    """Final row layout written by step 7."""

    compound_id: str
    chebi_accession: str
    chebi_name: str
    pathway_rank: int
    score: float
    confidence_level: str
    evidence_type: str
    mapping_confidence: float
    support_kegg_compound_ids: tuple[str, ...]
    support_kegg_names: tuple[str, ...]
    pathway_target_id: str
    pathway_target_type: str
    ath_pathway_id: str
    map_pathway_id: str
    relation_vstamp: str
    pathway_name: str
    pathway_group: str
    pathway_category: str
    map_pathway_compound_count: int
    ath_gene_count: int
    support_reaction_count: int
    role_summary: str
    plantcyc_support_source: str
    plantcyc_support_examples: str
    reactome_matches: str
    top_positive_features: str
    top_negative_features: str
    feature_contributions_json: str
    reason: str


@dataclass(slots=True)
class PipelinePaths:
    """All well-known input and output paths used by the refactored pipeline."""

    workdir: Path
    refs: Path
    outputs: Path
    compounds_path: Path
    comments_path: Path
    chebi_names_path: Path
    chebi_database_accession_path: Path
    chebi_chemical_data_path: Path
    chebi_structures_path: Path
    kegg_compound_list_path: Path
    kegg_compound_pathway_path: Path
    kegg_compound_reaction_path: Path
    kegg_reaction_pathway_path: Path
    kegg_reaction_details_path: Path
    kegg_info_compound_path: Path
    kegg_info_pathway_path: Path
    kegg_info_reaction_path: Path
    kegg_pathway_ath_path: Path
    kegg_pathway_map_path: Path
    kegg_pathway_hierarchy_path: Path
    kegg_ath_gene_pathway_path: Path
    pubchem_synonyms_path: Path
    lipidmaps_sdf_path: Path
    aracyc_compounds_path: Path
    aracyc_pathways_path: Path
    plantcyc_compounds_path: Path
    plantcyc_pathways_path: Path
    reactome_pathways_path: Path
    preprocessed_dir: Path
    alias_output_path: Path
    mapping_summary_path: Path
    mapping_selected_path: Path
    pathway_output_path: Path
    summary_output_path: Path
    name_normalization_index_path: Path
    name_to_formula_index_path: Path
    name_to_kegg_index_path: Path
    compound_structure_kegg_index_path: Path
    plantcyc_compound_index_path: Path
    kegg_structure_index_path: Path
    plant_to_kegg_bridge_path: Path
    pathway_link_snapshot_path: Path
    pathway_link_snapshot_diff_path: Path
    compound_reaction_index_path: Path
    reaction_pathway_index_path: Path
    compound_to_pathway_role_index_path: Path
    compound_to_pathway_index_path: Path
    map_to_ath_index_path: Path
    pathway_annotation_index_path: Path
    plant_evidence_index_path: Path
    preprocess_metadata_path: Path
    preprocessed_history_dir: Path

    @classmethod
    def from_workdir(cls, workdir: Path, output_tag: str = "refactored") -> "PipelinePaths":
        """Construct a path bundle from the workspace root."""

        workdir = workdir.resolve()
        refs = workdir / "refs"
        outputs = workdir / "outputs"
        outputs.mkdir(parents=True, exist_ok=True)
        preprocessed_dir = outputs / "preprocessed"
        preprocessed_dir.mkdir(parents=True, exist_ok=True)
        preprocessed_history_dir = preprocessed_dir / "history"
        preprocessed_history_dir.mkdir(parents=True, exist_ok=True)
        suffix = f"_{output_tag}" if output_tag else ""
        return cls(
            workdir=workdir,
            refs=refs,
            outputs=outputs,
            compounds_path=workdir / "compounds.tsv",
            comments_path=workdir / "comments.tsv",
            chebi_names_path=refs / "names.tsv.gz",
            chebi_database_accession_path=refs / "database_accession.tsv.gz",
            chebi_chemical_data_path=refs / "chemical_data.tsv.gz",
            chebi_structures_path=refs / "structures.tsv.gz",
            kegg_compound_list_path=refs / "kegg_compound_list.tsv",
            kegg_compound_pathway_path=refs / "kegg_compound_pathway.tsv",
            kegg_compound_reaction_path=refs / "kegg_compound_reaction.tsv",
            kegg_reaction_pathway_path=refs / "kegg_reaction_pathway.tsv",
            kegg_reaction_details_path=refs / "kegg_reaction_details.tsv",
            kegg_info_compound_path=refs / "kegg_info_compound.txt",
            kegg_info_pathway_path=refs / "kegg_info_pathway.txt",
            kegg_info_reaction_path=refs / "kegg_info_reaction.txt",
            kegg_pathway_ath_path=refs / "kegg_pathway_ath.tsv",
            kegg_pathway_map_path=refs / "kegg_pathway_map.tsv",
            kegg_pathway_hierarchy_path=refs / "kegg_pathway_hierarchy.txt",
            kegg_ath_gene_pathway_path=refs / "kegg_ath_gene_pathway.tsv",
            pubchem_synonyms_path=refs / "pubchem_cid_synonym_filtered.gz",
            lipidmaps_sdf_path=refs / "lmsd_extended.sdf.zip",
            aracyc_compounds_path=refs / "aracyc_compounds.20230103",
            aracyc_pathways_path=refs / "aracyc_pathways.20230103",
            plantcyc_compounds_path=refs / "plantcyc_compounds.20220103",
            plantcyc_pathways_path=refs / "plantcyc_pathways.20230103",
            reactome_pathways_path=refs / "ReactomePathways.txt",
            preprocessed_dir=preprocessed_dir,
            alias_output_path=outputs / f"chebi_aliases_standardized{suffix}.tsv",
            mapping_summary_path=outputs / f"chebi_kegg_mapping{suffix}.tsv",
            mapping_selected_path=outputs / f"chebi_kegg_selected{suffix}.tsv",
            pathway_output_path=outputs / f"chebi_pathways_ranked{suffix}.tsv",
            summary_output_path=outputs / f"processing_summary{suffix}.json",
            name_normalization_index_path=preprocessed_dir / "name_normalization_index.tsv",
            name_to_formula_index_path=preprocessed_dir / "name_to_formula_index.tsv",
            name_to_kegg_index_path=preprocessed_dir / "name_to_kegg_index.tsv",
            compound_structure_kegg_index_path=preprocessed_dir / "compound_structure_kegg_index.tsv",
            plantcyc_compound_index_path=preprocessed_dir / "plantcyc_compound_index.tsv",
            kegg_structure_index_path=preprocessed_dir / "kegg_structure_index.tsv",
            plant_to_kegg_bridge_path=preprocessed_dir / "plant_to_kegg_bridge.tsv",
            pathway_link_snapshot_path=preprocessed_dir / "pathway_link_snapshot.tsv",
            pathway_link_snapshot_diff_path=preprocessed_dir / "pathway_link_snapshot_diff.tsv",
            compound_reaction_index_path=preprocessed_dir / "compound_reaction_index.tsv",
            reaction_pathway_index_path=preprocessed_dir / "reaction_pathway_index.tsv",
            compound_to_pathway_role_index_path=preprocessed_dir / "compound_to_pathway_role_index.tsv",
            compound_to_pathway_index_path=preprocessed_dir / "compound_to_pathway_index.tsv",
            map_to_ath_index_path=preprocessed_dir / "map_to_ath_index.tsv",
            pathway_annotation_index_path=preprocessed_dir / "pathway_annotation_index.tsv",
            plant_evidence_index_path=preprocessed_dir / "plant_evidence_index.tsv",
            preprocess_metadata_path=preprocessed_dir / "preprocess_metadata.json",
            preprocessed_history_dir=preprocessed_history_dir,
        )

    def required_inputs(self) -> list[Path]:
        """Return the baseline inputs required before step 3 KEGG refresh logic."""

        return [
            self.compounds_path,
            self.comments_path,
            self.chebi_names_path,
            self.chebi_database_accession_path,
            self.chebi_chemical_data_path,
            self.chebi_structures_path,
            self.kegg_compound_list_path,
            self.kegg_compound_pathway_path,
            self.kegg_pathway_ath_path,
            self.kegg_pathway_map_path,
            self.kegg_pathway_hierarchy_path,
            self.kegg_ath_gene_pathway_path,
            self.pubchem_synonyms_path,
            self.lipidmaps_sdf_path,
            self.aracyc_compounds_path,
            self.aracyc_pathways_path,
            self.plantcyc_compounds_path,
            self.plantcyc_pathways_path,
            self.reactome_pathways_path,
        ]

    def step3_required_inputs(self) -> list[Path]:
        """Return the KEGG refs required specifically by the step-3 snapshot layer."""

        return [
            self.kegg_compound_reaction_path,
            self.kegg_reaction_pathway_path,
            self.kegg_reaction_details_path,
            self.kegg_info_compound_path,
            self.kegg_info_pathway_path,
            self.kegg_info_reaction_path,
        ]

    def ensure_required_inputs(self) -> None:
        """Fail fast when a required file is missing."""

        for path in self.required_inputs():
            ensure_exists(path)


@dataclass
class PipelineContext:
    """Mutable state passed from one step module to the next."""

    paths: PipelinePaths
    refresh_step3_kegg: bool = False
    compounds: dict[str, ChEBICompound] = field(default_factory=dict)
    comments_profile: dict[str, int] = field(default_factory=dict)
    base_aliases: dict[str, list[AliasRecord]] = field(default_factory=dict)
    xrefs: dict[str, XrefInfo] = field(default_factory=dict)
    formulas: dict[str, tuple[str, str]] = field(default_factory=dict)
    structures: dict[str, StructureInfo] = field(default_factory=dict)
    compound_contexts: dict[str, CompoundContext] = field(default_factory=dict)
    alias_rows: int = 0

    plantcyc_records: dict[str, PlantCycCompound] = field(default_factory=dict)
    plantcyc_indexes: dict[str, dict[str, set[str]]] = field(default_factory=dict)
    plantcyc_pathway_stats: dict[str, dict[str, dict[str, object]]] = field(default_factory=dict)
    lipidmaps_records: dict[str, LipidMapsRecord] = field(default_factory=dict)
    lipidmaps_indexes: dict[str, dict[str, set[str]]] = field(default_factory=dict)
    pubchem_synonyms: dict[str, list[str]] = field(default_factory=dict)
    pubchem_stats: dict[str, int] = field(default_factory=dict)
    name_formula_index: dict[str, set[str]] = field(default_factory=dict)
    kegg_structure_indexes: dict[str, dict[str, set[str]]] = field(default_factory=dict)

    kegg_compounds: dict[str, KeggCompound] = field(default_factory=dict)
    kegg_standard_indexes: dict[str, dict[str, set[str]]] = field(default_factory=dict)
    kegg_alias_indexes: dict[str, dict[str, set[str]]] = field(default_factory=dict)
    kegg_primary_compact_delete_index: dict[str, set[str]] = field(default_factory=dict)

    ranked_candidates_by_compound: dict[str, list[CandidateMapping]] = field(default_factory=dict)
    selected_by_compound: dict[str, list[CandidateMapping]] = field(default_factory=dict)
    plant_to_kegg_bridge_rows_by_compound: dict[str, list[PlantToKeggBridge]] = field(default_factory=dict)
    support_contexts: dict[str, StoredSupportContext] = field(default_factory=dict)
    mapping_status: Counter[str] = field(default_factory=Counter)

    raw_pathway_hits: dict[str, list[Step3PathwayHit]] = field(default_factory=dict)
    resolved_pathway_hits: dict[str, list[Step4ResolvedPathwayHit]] = field(default_factory=dict)
    annotated_pathway_hits: dict[str, list[Step5AnnotatedPathwayHit]] = field(default_factory=dict)
    ranked_pathway_rows: dict[str, list[RankedPathwayRow]] = field(default_factory=dict)
    pathway_status: Counter[str] = field(default_factory=Counter)

    ath_pathways: dict[str, str] = field(default_factory=dict)
    map_to_ath: dict[str, str] = field(default_factory=dict)
    map_pathways: dict[str, str] = field(default_factory=dict)
    pathway_categories: dict[str, tuple[str, str, str]] = field(default_factory=dict)
    kegg_to_pathways: dict[str, list[tuple[str, str]]] = field(default_factory=dict)
    map_pathway_compound_counts: dict[str, int] = field(default_factory=dict)
    step3_vstamp: str = ""
    step3_release_text: str = ""
    ath_gene_counts: dict[str, int] = field(default_factory=dict)
    reactome_pathways: dict[str, list[tuple[str, str]]] = field(default_factory=dict)
    preprocess_counts: Counter[str] = field(default_factory=Counter)
    preprocess_version: str = ""

    step_notes: list[str] = field(default_factory=list)

    def add_note(self, note: str) -> None:
        """Keep lightweight audit notes across optional steps."""

        if note not in self.step_notes:
            self.step_notes.append(note)
