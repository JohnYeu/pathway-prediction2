"""Shared state containers for the refactored step-wise pathway pipeline."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

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
    map_id: str
    kegg_name: str
    brite_l1: str
    brite_l2: str
    brite_l3: str
    kegg_vstamp: str
    pathway_group: str
    pathway_category: str
    map_pathway_compound_count: int
    ath_gene_count: int
    go_top_terms_json: str
    go_best_term: str
    go_best_fdr: float
    go_vstamp: str
    plant_context_tags: tuple[str, ...]
    plant_evidence_sources: tuple[str, ...]
    aracyc_evidence_score: float
    reactome_matches: tuple[tuple[str, str], ...]
    plant_reactome_matches_json: str
    plant_reactome_best_id: str
    plant_reactome_best_name: str
    plant_reactome_best_category: str
    plant_reactome_best_description: str
    plant_reactome_alignment_score: float
    plant_reactome_alignment_confidence: str
    plant_reactome_tags: tuple[str, ...]
    plant_reactome_vstamp: str
    annotation_confidence: str
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
    map_id: str
    kegg_name: str
    brite_l1: str
    brite_l2: str
    brite_l3: str
    pathway_group: str
    pathway_category: str
    map_pathway_compound_count: int
    ath_gene_count: int
    go_best_term: str
    go_best_fdr: float
    plant_context_tags: str
    plant_reactome_best_category: str
    plant_reactome_alignment_confidence: str
    annotation_confidence: str
    support_reaction_count: int
    role_summary: str
    plantcyc_support_source: str
    plantcyc_support_examples: str
    reactome_matches: str
    top_positive_features: str
    top_negative_features: str
    feature_contributions_json: str
    reason: str


# ---------------------------------------------------------------------------
# AraCyc-first pipeline data structures
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class AraCycCompoundMatch:
    """Result of matching one ChEBI compound to an AraCyc/PlantCyc compound."""

    aracyc_compound_id: str
    aracyc_common_name: str
    source_db: str  # "AraCyc" or "PlantCyc"
    match_method: str  # "chebi_xref" | "kegg_xref" | "inchikey" | "tanimoto" | "name_exact" | "name_variant"
    match_score: float  # 0.0-1.0, method-dependent
    chebi_xref_direct: bool
    structure_validated: bool  # True if InChIKey/SMILES confirmed
    pathways: tuple[str, ...]  # pathway names from the PMN compound file
    smiles: str
    ec_numbers: tuple[str, ...]
    plant_record_id: str  # e.g. "AraCyc:UDP-D-XYLOSE"
    reference_compound_key: str


@dataclass(slots=True)
class AraCycPathwayHit:
    """AraCyc compound-pathway link with reaction context."""

    compound_id: str  # ChEBI compound_id
    match: AraCycCompoundMatch
    pathway_id: str  # PWY-xxxx from aracyc_pathways
    pathway_name: str  # human-readable name
    ec_numbers: tuple[str, ...]
    reaction_ids: tuple[str, ...]
    reaction_equations: tuple[str, ...]
    source_db: str  # "AraCyc" or "PlantCyc"


@dataclass(slots=True)
class AnnotatedAraCycHit:
    """AraCyc pathway hit enriched with gene, EC, and classification context."""

    compound_id: str
    match: AraCycCompoundMatch
    pathway_id: str
    pathway_name: str
    pathway_category: str  # extracted from name: biosynthesis, degradation, etc.
    ec_numbers: tuple[str, ...]
    reaction_ids: tuple[str, ...]
    reaction_count: int  # total reactions in this pathway
    compound_count: int  # total compounds participating in this pathway
    gene_ids: tuple[str, ...]  # AT-locus IDs
    gene_names: tuple[str, ...]
    gene_count: int
    go_best_term: str
    go_best_fdr: float
    plant_reactome_best_id: str
    plant_reactome_best_name: str
    plant_reactome_alignment_score: float
    plant_reactome_alignment_confidence: str
    annotation_confidence: str
    source_db: str
    cofactor_like: bool


@dataclass(slots=True)
class AraCycRankedRow:
    """Final row layout for the AraCyc-first pipeline output."""

    compound_id: str
    chebi_accession: str
    chebi_name: str
    pathway_rank: int
    score: float
    confidence_level: str
    evidence_type: str
    match_method: str
    match_score: float
    aracyc_compound_id: str
    aracyc_compound_name: str
    source_db: str
    pathway_id: str
    pathway_name: str
    pathway_category: str
    gene_count: int
    gene_ids: str
    ec_numbers: str
    reaction_count: int
    compound_count: int
    go_best_term: str
    go_best_fdr: float
    plant_reactome_best_category: str
    plant_reactome_alignment_confidence: str
    annotation_confidence: str
    chebi_xref_direct: bool
    structure_validated: bool
    cofactor_like: bool
    top_positive_features: str
    top_negative_features: str
    feature_contributions_json: str
    reason: str


# ---------------------------------------------------------------------------
# AraCyc pathway metadata (from aracyc_pathways file)
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class AraCycPathwayInfo:
    """Aggregated metadata for one AraCyc pathway ID."""

    pathway_id: str
    pathway_name: str
    reaction_ids: set[str]
    ec_numbers: set[str]
    gene_ids: set[str]
    gene_names: set[str]
    protein_ids: set[str]
    compound_ids: set[str]  # AraCyc compound IDs participating


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
    go_basic_obo_path: Path
    gene_association_tair_path: Path
    pubchem_synonyms_path: Path
    lipidmaps_sdf_path: Path
    aracyc_compounds_path: Path
    aracyc_pathways_path: Path
    plantcyc_compounds_path: Path
    plantcyc_pathways_path: Path
    reactome_pathways_path: Path
    plant_reactome_pathways_path: Path
    plant_reactome_gene_pathway_path: Path
    plant_reactome_version_path: Path
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
    pathway_go_enrichment_index_path: Path
    gene_to_go_index_path: Path
    plant_reactome_pathway_index_path: Path
    plant_reactome_gene_index_path: Path
    plant_reactome_alignment_path: Path
    plant_evidence_index_path: Path
    # AraCyc-first pipeline preprocessed indexes
    name_to_aracyc_index_path: Path
    compound_structure_aracyc_index_path: Path
    aracyc_compound_pathway_index_path: Path
    aracyc_pathway_annotation_index_path: Path
    aracyc_pathway_output_path: Path

    expanded_candidates_path: Path
    ml_training_pairs_path: Path
    ml_pathway_predictions_path: Path
    ml_model_path: Path
    ml_model_metadata_path: Path
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
            go_basic_obo_path=refs / "go-basic.obo",
            gene_association_tair_path=refs / "gene_association.tair.gz",
            pubchem_synonyms_path=refs / "pubchem_cid_synonym_filtered.gz",
            lipidmaps_sdf_path=refs / "lmsd_extended.sdf.zip",
            aracyc_compounds_path=refs / "aracyc_compounds.20230103",
            aracyc_pathways_path=refs / "aracyc_pathways.20230103",
            plantcyc_compounds_path=refs / "plantcyc_compounds.20220103",
            plantcyc_pathways_path=refs / "plantcyc_pathways.20230103",
            reactome_pathways_path=refs / "ReactomePathways.txt",
            plant_reactome_pathways_path=refs / "plant_reactome_pathways.tsv",
            plant_reactome_gene_pathway_path=refs / "plant_reactome_gene_pathway.tsv",
            plant_reactome_version_path=refs / "plant_reactome_version.txt",
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
            pathway_go_enrichment_index_path=preprocessed_dir / "pathway_go_enrichment.tsv",
            gene_to_go_index_path=preprocessed_dir / "gene_to_go_index.tsv",
            plant_reactome_pathway_index_path=preprocessed_dir / "plant_reactome_pathway_index.tsv",
            plant_reactome_gene_index_path=preprocessed_dir / "plant_reactome_gene_index.tsv",
            plant_reactome_alignment_path=preprocessed_dir / "plant_reactome_alignment.tsv",
            plant_evidence_index_path=preprocessed_dir / "plant_evidence_index.tsv",
            name_to_aracyc_index_path=preprocessed_dir / "name_to_aracyc_index.tsv",
            compound_structure_aracyc_index_path=preprocessed_dir / "compound_structure_aracyc_index.tsv",
            aracyc_compound_pathway_index_path=preprocessed_dir / "aracyc_compound_pathway_index.tsv",
            aracyc_pathway_annotation_index_path=preprocessed_dir / "aracyc_pathway_annotation_index.tsv",
            aracyc_pathway_output_path=outputs / f"chebi_pathways_aracyc{suffix}.tsv",
            expanded_candidates_path=preprocessed_dir / "compound_to_external_pathway_candidates.tsv",
            ml_training_pairs_path=preprocessed_dir / "ml_training_pairs.tsv",
            ml_pathway_predictions_path=preprocessed_dir / "ml_pathway_predictions.tsv",
            ml_model_path=preprocessed_dir / "models" / "pathway_expansion_model.joblib",
            ml_model_metadata_path=preprocessed_dir / "ml_model_metadata.json",
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
            self.go_basic_obo_path,
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
    go_gene_index: dict[str, set[tuple[str, str]]] = field(default_factory=dict)
    go_term_namespace: dict[str, str] = field(default_factory=dict)
    pathway_go_enrichment: dict[str, dict[str, object]] = field(default_factory=dict)
    go_vstamp: str = ""
    plant_reactome_pathways: dict[str, dict[str, object]] = field(default_factory=dict)
    plant_reactome_gene_sets: dict[str, set[str]] = field(default_factory=dict)
    plant_reactome_alignments: dict[str, list[dict[str, object]]] = field(default_factory=dict)
    plant_reactome_vstamp: str = ""
    preprocess_counts: Counter[str] = field(default_factory=Counter)
    preprocess_version: str = ""

    # AraCyc-first pipeline state
    aracyc_pathway_info: dict[str, AraCycPathwayInfo] = field(default_factory=dict)
    aracyc_pathway_compound_counts: dict[str, int] = field(default_factory=dict)
    aracyc_matches_by_compound: dict[str, list[AraCycCompoundMatch]] = field(default_factory=dict)
    aracyc_pathway_hits: dict[str, list[AraCycPathwayHit]] = field(default_factory=dict)
    annotated_aracyc_hits: dict[str, list[AnnotatedAraCycHit]] = field(default_factory=dict)
    aracyc_ranked_rows: dict[str, list[AraCycRankedRow]] = field(default_factory=dict)
    aracyc_mapping_status: Counter[str] = field(default_factory=Counter)
    aracyc_reference_compound_keys: set[str] = field(default_factory=set)

    expanded_candidates_count: int = 0
    ml_predictions_count: int = 0

    step_notes: list[str] = field(default_factory=list)

    def add_note(self, note: str) -> None:
        """Keep lightweight audit notes across optional steps."""

        if note not in self.step_notes:
            self.step_notes.append(note)

    def aracyc_reference_total(self) -> int:
        """Return the Arabidopsis reference compound count for AraCyc-first coverage."""

        return len(self.aracyc_reference_compound_keys)
