"""
Model Registry for Single-Cell Foundation Models

Defines model capabilities, I/O contracts, and hardware requirements.
See pantheon/factory/templates/skills/omics/_scfm_docs/ for runbooks and checkpoint conventions.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class TaskType(str, Enum):
    """Supported foundation model tasks"""
    EMBED = "embed"
    ANNOTATE = "annotate"
    INTEGRATE = "integrate"
    PERTURB = "perturb"
    SPATIAL = "spatial"
    DRUG_RESPONSE = "drug_response"


class Modality(str, Enum):
    """Data modalities"""
    RNA = "RNA"
    ATAC = "ATAC"
    SPATIAL = "Spatial"
    PROTEIN = "Protein"
    MULTIOMICS = "Multi-omics"


class GeneIDScheme(str, Enum):
    """Gene identifier schemes"""
    SYMBOL = "symbol"  # HGNC gene symbols (e.g., TP53)
    ENSEMBL = "ensembl"  # Ensembl IDs (e.g., ENSG00000141510)
    CUSTOM = "custom"  # Model-specific gene set


class SkillReadyStatus(str, Enum):
    """Documentation/adapter readiness"""
    READY = "ready"  # Full Pantheon Adapter Spec
    PARTIAL = "partial"  # Partial spec, needs validation
    REFERENCE = "reference"  # Reference docs only, no spec


@dataclass
class HardwareRequirements:
    """Hardware requirements for model inference"""
    gpu_required: bool = True
    min_vram_gb: int = 8
    recommended_vram_gb: int = 16
    cpu_fallback: bool = False
    default_batch_size: int = 64


@dataclass
class OutputKeys:
    """Standard AnnData output keys for each task"""
    embedding_key: str = ""  # obsm key for embeddings
    annotation_key: str = ""  # obs key for predictions
    confidence_key: str = ""  # obs key for confidence scores
    integration_key: str = ""  # obsm key for integrated embeddings


@dataclass
class ModelSpec:
    """Complete specification for a foundation model"""
    # Identity
    name: str
    version: str
    skill_ready: SkillReadyStatus = SkillReadyStatus.REFERENCE

    # Capabilities
    tasks: list[TaskType] = field(default_factory=list)
    modalities: list[Modality] = field(default_factory=list)
    species: list[str] = field(default_factory=list)  # e.g., ["human", "mouse"]

    # Input requirements
    gene_id_scheme: GeneIDScheme = GeneIDScheme.SYMBOL
    requires_finetuning: bool = False  # For annotation/some tasks
    zero_shot_embedding: bool = True
    zero_shot_annotation: bool = False

    # Output contract
    output_keys: OutputKeys = field(default_factory=OutputKeys)
    embedding_dim: int = 512

    # Hardware
    hardware: HardwareRequirements = field(default_factory=HardwareRequirements)

    # Routing hints
    differentiator: str = ""  # Unique feature that distinguishes this model
    prefer_when: str = ""  # When to specifically choose this model

    # Resources
    checkpoint_url: str = ""
    documentation_url: str = ""
    paper_url: str = ""
    license_notes: str = ""

    def supports_task(self, task: TaskType) -> bool:
        """Check if model supports a given task"""
        return task in self.tasks

    def supports_modality(self, modality: Modality) -> bool:
        """Check if model supports a given modality"""
        return modality in self.modalities

    def supports_species(self, species: str) -> bool:
        """Check if model supports a given species"""
        return species.lower() in [s.lower() for s in self.species]

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization"""
        return {
            "name": self.name,
            "version": self.version,
            "skill_ready": self.skill_ready.value,
            "tasks": [t.value for t in self.tasks],
            "modalities": [m.value for m in self.modalities],
            "species": self.species,
            "gene_id_scheme": self.gene_id_scheme.value,
            "zero_shot_embedding": self.zero_shot_embedding,
            "zero_shot_annotation": self.zero_shot_annotation,
            "requires_finetuning": self.requires_finetuning,
            "embedding_dim": self.embedding_dim,
            "output_keys": {
                "embedding": self.output_keys.embedding_key,
                "annotation": self.output_keys.annotation_key,
                "confidence": self.output_keys.confidence_key,
            },
            "hardware": {
                "gpu_required": self.hardware.gpu_required,
                "min_vram_gb": self.hardware.min_vram_gb,
                "cpu_fallback": self.hardware.cpu_fallback,
            },
            "differentiator": self.differentiator,
            "prefer_when": self.prefer_when,
            "documentation_url": self.documentation_url,
        }


# =============================================================================
# Model Registry - Skill-Ready Models (✅)
# =============================================================================

SCGPT_SPEC = ModelSpec(
    name="scgpt",
    version="whole-human-2024",
    skill_ready=SkillReadyStatus.READY,
    tasks=[TaskType.EMBED, TaskType.INTEGRATE],  # ANNOTATE/PERTURB require fine-tuning (not yet implemented)
    modalities=[Modality.RNA, Modality.ATAC, Modality.SPATIAL],
    species=["human", "mouse"],
    gene_id_scheme=GeneIDScheme.SYMBOL,
    requires_finetuning=True,  # For annotation
    zero_shot_embedding=True,
    zero_shot_annotation=False,
    output_keys=OutputKeys(
        embedding_key="X_scGPT",
        annotation_key="scgpt_pred",
        confidence_key="scgpt_pred_score",
        integration_key="X_scGPT_integrated",
    ),
    embedding_dim=512,
    hardware=HardwareRequirements(
        gpu_required=True,
        min_vram_gb=8,
        recommended_vram_gb=16,
        cpu_fallback=True,
        default_batch_size=64,
    ),
    differentiator="Multi-modal transformer (RNA+ATAC+Spatial), attention-based gene interaction modeling",
    prefer_when="User needs multi-modal analysis (RNA+ATAC or spatial), or explicit attention-based gene interaction maps",
    checkpoint_url="https://github.com/bowang-lab/scGPT#pretrained-scgpt-model-zoo",
    documentation_url="https://scgpt.readthedocs.io/",
    paper_url="https://www.nature.com/articles/s41592-024-02201-0",
    license_notes="Check upstream LICENSE; treat as restricted until verified",
)

GENEFORMER_SPEC = ModelSpec(
    name="geneformer",
    version="v2-106M",
    skill_ready=SkillReadyStatus.READY,
    tasks=[TaskType.EMBED, TaskType.INTEGRATE],  # ANNOTATE/PERTURB require fine-tuning (not yet implemented)
    modalities=[Modality.RNA],
    species=["human"],
    gene_id_scheme=GeneIDScheme.ENSEMBL,  # Requires Ensembl IDs
    requires_finetuning=True,
    zero_shot_embedding=True,
    zero_shot_annotation=False,
    output_keys=OutputKeys(
        embedding_key="X_geneformer",
        annotation_key="geneformer_pred",
        confidence_key="",
    ),
    embedding_dim=512,
    hardware=HardwareRequirements(
        gpu_required=False,  # Recommended but not required
        min_vram_gb=4,
        recommended_vram_gb=16,
        cpu_fallback=True,
        default_batch_size=32,
    ),
    differentiator="Rank-value encoded transformer, Ensembl gene IDs, CPU-capable, network biology pretraining",
    prefer_when="User has Ensembl gene IDs, needs CPU-only inference, or wants gene-network-aware embeddings",
    checkpoint_url="https://huggingface.co/ctheodoris/Geneformer",
    documentation_url="https://geneformer.readthedocs.io/",
    paper_url="https://www.nature.com/articles/s41586-023-06139-9",
    license_notes="Apache 2.0 (code); check model weights terms",
)

UCE_SPEC = ModelSpec(
    name="uce",
    version="4-layer",
    skill_ready=SkillReadyStatus.READY,
    tasks=[TaskType.EMBED, TaskType.INTEGRATE],
    modalities=[Modality.RNA],
    species=["human", "mouse", "zebrafish", "mouse_lemur", "macaque", "frog", "pig"],
    gene_id_scheme=GeneIDScheme.SYMBOL,
    requires_finetuning=False,  # Zero-shot only
    zero_shot_embedding=True,
    zero_shot_annotation=False,  # No annotation task
    output_keys=OutputKeys(
        embedding_key="X_uce",
        annotation_key="",
        confidence_key="",
    ),
    embedding_dim=1280,
    hardware=HardwareRequirements(
        gpu_required=True,
        min_vram_gb=16,
        recommended_vram_gb=16,
        cpu_fallback=False,
        default_batch_size=100,
    ),
    differentiator="Broadest species support (7 species), 1280-dim embeddings, universal cell embedding via protein structure",
    prefer_when="User has non-human/non-mouse species (zebrafish, frog, pig, macaque, lemur), or needs cross-species comparison",
    checkpoint_url="https://github.com/snap-stanford/UCE",
    documentation_url="https://github.com/snap-stanford/UCE",
    paper_url="https://www.nature.com/articles/s41592-024-02201-0",
    license_notes="MIT License",
)

# =============================================================================
# Model Registry - Partial Specs (⚠️)
# =============================================================================

SCFOUNDATION_SPEC = ModelSpec(
    name="scfoundation",
    version="xTrimoGene",
    skill_ready=SkillReadyStatus.PARTIAL,
    tasks=[TaskType.EMBED, TaskType.INTEGRATE],  # ANNOTATE/PERTURB/DRUG_RESPONSE require fine-tuning
    modalities=[Modality.RNA],
    species=["human"],
    gene_id_scheme=GeneIDScheme.CUSTOM,  # Model's 19,264 gene set
    requires_finetuning=True,
    zero_shot_embedding=True,
    zero_shot_annotation=False,
    output_keys=OutputKeys(
        embedding_key="X_scfoundation",
        annotation_key="scfoundation_pred",
        confidence_key="",
    ),
    embedding_dim=512,
    hardware=HardwareRequirements(
        gpu_required=True,
        min_vram_gb=16,
        recommended_vram_gb=32,
        cpu_fallback=False,
        default_batch_size=64,
    ),
    differentiator="Large-scale asymmetric transformer (xTrimoGene), custom 19264 gene vocabulary, pre-trained for perturbation/drug response",
    prefer_when="User needs perturbation prediction, drug response modeling, or works with the xTrimoGene gene vocabulary",
    checkpoint_url="https://github.com/biomap-research/scFoundation",
    documentation_url="https://github.com/biomap-research/scFoundation",
    paper_url="https://www.nature.com/articles/s41592-024-02305-7",
    license_notes="Check upstream LICENSE",
)

SCBERT_SPEC = ModelSpec(
    name="scbert",
    version="v1.0",
    skill_ready=SkillReadyStatus.PARTIAL,
    tasks=[TaskType.EMBED, TaskType.INTEGRATE],  # ANNOTATE requires fine-tuning
    modalities=[Modality.RNA],
    species=["human"],
    gene_id_scheme=GeneIDScheme.SYMBOL,
    requires_finetuning=True,  # For annotation
    zero_shot_embedding=True,
    zero_shot_annotation=False,
    output_keys=OutputKeys(
        embedding_key="X_scBERT",
        annotation_key="scbert_pred",
        confidence_key="",
    ),
    embedding_dim=200,
    hardware=HardwareRequirements(
        gpu_required=True,
        min_vram_gb=8,
        recommended_vram_gb=16,
        cpu_fallback=True,
        default_batch_size=64,
    ),
    differentiator="Compact 200-dim embeddings, BERT-style masked gene pretraining, lightweight model",
    prefer_when="User needs compact 200-dim embeddings, BERT-style pretraining, or a lightweight model for constrained hardware",
    checkpoint_url="https://github.com/TencentAILabHealthcare/scBERT",
    documentation_url="https://github.com/TencentAILabHealthcare/scBERT",
    paper_url="https://www.nature.com/articles/s42256-022-00534-z",
    license_notes="Check upstream LICENSE",
)

GENECOMPASS_SPEC = ModelSpec(
    name="genecompass",
    version="120M-cells",
    skill_ready=SkillReadyStatus.PARTIAL,
    tasks=[TaskType.EMBED, TaskType.INTEGRATE],
    modalities=[Modality.RNA],
    species=["human", "mouse"],
    gene_id_scheme=GeneIDScheme.SYMBOL,
    requires_finetuning=True,
    zero_shot_embedding=True,
    zero_shot_annotation=False,
    output_keys=OutputKeys(
        embedding_key="X_genecompass",
        annotation_key="genecompass_pred",
        confidence_key="",
    ),
    embedding_dim=512,
    hardware=HardwareRequirements(
        gpu_required=True,
        min_vram_gb=16,
        recommended_vram_gb=32,
        cpu_fallback=False,
        default_batch_size=32,
    ),
    differentiator="Prior-knowledge-enhanced pretraining (gene regulatory networks + pathway info), 120M cell training corpus",
    prefer_when="User mentions prior knowledge, gene regulatory networks, pathway-informed embeddings, or mouse+human cross-species",
    checkpoint_url="https://github.com/xCompass-AI/GeneCompass",
    documentation_url="https://github.com/xCompass-AI/GeneCompass",
    paper_url="https://www.biorxiv.org/content/10.1101/2023.09.26.559542v1",
    license_notes="Check upstream LICENSE; prior-knowledge enhanced",
)

CELLPLM_SPEC = ModelSpec(
    name="cellplm",
    version="v1.0",
    skill_ready=SkillReadyStatus.PARTIAL,
    tasks=[TaskType.EMBED, TaskType.INTEGRATE],
    modalities=[Modality.RNA],
    species=["human"],
    gene_id_scheme=GeneIDScheme.SYMBOL,
    requires_finetuning=False,  # Efficient cell-centric embeddings
    zero_shot_embedding=True,
    zero_shot_annotation=False,
    output_keys=OutputKeys(
        embedding_key="X_cellplm",
        annotation_key="",
        confidence_key="",
    ),
    embedding_dim=512,
    hardware=HardwareRequirements(
        gpu_required=True,
        min_vram_gb=8,
        recommended_vram_gb=16,
        cpu_fallback=True,
        default_batch_size=128,  # Fast inference
    ),
    differentiator="Cell-centric (not gene-centric) architecture, highest batch throughput (batch_size=128), fast inference",
    prefer_when="User needs fast inference, high throughput, million-cell scale processing, or cell-level (not gene-level) modeling",
    checkpoint_url="https://github.com/OmicsML/CellPLM",
    documentation_url="https://github.com/OmicsML/CellPLM",
    paper_url="https://www.biorxiv.org/content/10.1101/2023.10.03.560734v1",
    license_notes="Check upstream LICENSE",
)

NICHEFORMER_SPEC = ModelSpec(
    name="nicheformer",
    version="v1.0",
    skill_ready=SkillReadyStatus.PARTIAL,
    tasks=[TaskType.EMBED, TaskType.INTEGRATE, TaskType.SPATIAL],
    modalities=[Modality.SPATIAL, Modality.RNA],
    species=["human", "mouse"],
    gene_id_scheme=GeneIDScheme.SYMBOL,
    requires_finetuning=False,
    zero_shot_embedding=True,
    zero_shot_annotation=False,
    output_keys=OutputKeys(
        embedding_key="X_nicheformer",
        annotation_key="",
        confidence_key="",
    ),
    embedding_dim=512,
    hardware=HardwareRequirements(
        gpu_required=True,
        min_vram_gb=16,
        recommended_vram_gb=32,
        cpu_fallback=False,
        default_batch_size=32,
    ),
    differentiator="Niche-aware spatial transformer, jointly models spatial coordinates and gene expression",
    prefer_when="User has spatial transcriptomics data (Visium, MERFISH, Slide-seq) and wants niche-aware or spatial-context embeddings",
    checkpoint_url="https://github.com/theislab/nicheformer",
    documentation_url="https://github.com/theislab/nicheformer",
    paper_url="https://www.biorxiv.org/content/10.1101/2024.04.15.589472v1",
    license_notes="Check upstream LICENSE; spatial transcriptomics focus",
)

SCMULAN_SPEC = ModelSpec(
    name="scmulan",
    version="v1.0",
    skill_ready=SkillReadyStatus.PARTIAL,
    tasks=[TaskType.EMBED, TaskType.INTEGRATE],
    modalities=[Modality.RNA, Modality.ATAC, Modality.PROTEIN, Modality.MULTIOMICS],
    species=["human"],
    gene_id_scheme=GeneIDScheme.SYMBOL,
    requires_finetuning=False,
    zero_shot_embedding=True,
    zero_shot_annotation=False,
    output_keys=OutputKeys(
        embedding_key="X_scmulan",
        annotation_key="",
        confidence_key="",
    ),
    embedding_dim=512,
    hardware=HardwareRequirements(
        gpu_required=True,
        min_vram_gb=16,
        recommended_vram_gb=32,
        cpu_fallback=False,
        default_batch_size=32,
    ),
    differentiator="Native multi-omics joint modeling (RNA+ATAC+Protein simultaneously), designed for CITE-seq/10x Multiome",
    prefer_when="User has multi-omics data (CITE-seq, 10x Multiome, RNA+ATAC+Protein), or wants joint multi-modal embedding",
    checkpoint_url="https://github.com/SuperBianC/scMulan",
    documentation_url="https://github.com/SuperBianC/scMulan",
    paper_url="https://www.biorxiv.org/content/10.1101/2024.01.25.577152v1",
    license_notes="Check upstream LICENSE; multi-omics focus",
)

# =============================================================================
# Model Registry - Specialized & Emerging Models (2024-2025)
# =============================================================================

TGPT_SPEC = ModelSpec(
    name="tgpt",
    version="v1.0",
    skill_ready=SkillReadyStatus.PARTIAL,
    tasks=[TaskType.EMBED, TaskType.INTEGRATE],
    modalities=[Modality.RNA],
    species=["human"],
    gene_id_scheme=GeneIDScheme.SYMBOL,
    requires_finetuning=False,
    zero_shot_embedding=True,
    zero_shot_annotation=False,
    output_keys=OutputKeys(
        embedding_key="X_tgpt",
        annotation_key="",
        confidence_key="",
    ),
    embedding_dim=512,
    hardware=HardwareRequirements(
        gpu_required=True,
        min_vram_gb=16,
        recommended_vram_gb=32,
        cpu_fallback=False,
        default_batch_size=32,
    ),
    differentiator="Autoregressive next-token prediction of gene expression values (GPT-style, not masked)",
    prefer_when="User wants autoregressive/generative modeling, next-token prediction of gene expression, or GPT-style generation",
    checkpoint_url="https://github.com/deeplearningplus/tGPT",
    documentation_url="https://github.com/deeplearningplus/tGPT",
    paper_url="",
    license_notes="Check upstream LICENSE; next-token prediction",
)

CELLFM_SPEC = ModelSpec(
    name="cellfm",
    version="v1.0",
    skill_ready=SkillReadyStatus.PARTIAL,
    tasks=[TaskType.EMBED, TaskType.INTEGRATE],
    modalities=[Modality.RNA],
    species=["human"],
    gene_id_scheme=GeneIDScheme.SYMBOL,
    requires_finetuning=False,
    zero_shot_embedding=True,
    zero_shot_annotation=False,
    output_keys=OutputKeys(
        embedding_key="X_cellfm",
        annotation_key="",
        confidence_key="",
    ),
    embedding_dim=512,
    hardware=HardwareRequirements(
        gpu_required=True,
        min_vram_gb=16,
        recommended_vram_gb=32,
        cpu_fallback=False,
        default_batch_size=32,
    ),
    differentiator="MLP architecture (not transformer), trained on ~126M cells (largest training corpus)",
    prefer_when="User explicitly wants MLP-based (not transformer) model, or wants the largest pretraining scale (~126M cells)",
    checkpoint_url="https://github.com/cellverse/CellFM",
    documentation_url="https://github.com/cellverse/CellFM",
    paper_url="",
    license_notes="Check upstream LICENSE; MLP architecture, largest scale (~126M)",
)

SCCELLO_SPEC = ModelSpec(
    name="sccello",
    version="v1.0",
    skill_ready=SkillReadyStatus.PARTIAL,
    tasks=[TaskType.EMBED, TaskType.INTEGRATE, TaskType.ANNOTATE],
    modalities=[Modality.RNA],
    species=["human"],
    gene_id_scheme=GeneIDScheme.SYMBOL,
    requires_finetuning=False,
    zero_shot_embedding=True,
    zero_shot_annotation=True,  # Ontology-optimized
    output_keys=OutputKeys(
        embedding_key="X_sccello",
        annotation_key="sccello_pred",
        confidence_key="",
    ),
    embedding_dim=512,
    hardware=HardwareRequirements(
        gpu_required=True,
        min_vram_gb=16,
        recommended_vram_gb=32,
        cpu_fallback=False,
        default_batch_size=32,
    ),
    differentiator="Cell ontology-aligned embeddings, zero-shot cell type annotation with hierarchical coherence",
    prefer_when="User wants zero-shot cell type annotation, ontology-consistent predictions, or hierarchical cell-type labeling",
    checkpoint_url="https://github.com/cellarium-ai/scCello",
    documentation_url="https://github.com/cellarium-ai/scCello",
    paper_url="",
    license_notes="Check upstream LICENSE; ontology alignment, cell-type coherence",
)

SCPRINT_SPEC = ModelSpec(
    name="scprint",
    version="v1.0",
    skill_ready=SkillReadyStatus.PARTIAL,
    tasks=[TaskType.EMBED, TaskType.INTEGRATE],
    modalities=[Modality.RNA],
    species=["human"],
    gene_id_scheme=GeneIDScheme.SYMBOL,
    requires_finetuning=False,
    zero_shot_embedding=True,
    zero_shot_annotation=False,
    output_keys=OutputKeys(
        embedding_key="X_scprint",
        annotation_key="",
        confidence_key="",
    ),
    embedding_dim=512,
    hardware=HardwareRequirements(
        gpu_required=True,
        min_vram_gb=16,
        recommended_vram_gb=32,
        cpu_fallback=False,
        default_batch_size=32,
    ),
    differentiator="Protein-coding gene focus with built-in denoising, robust batch integration",
    prefer_when="User mentions denoising, protein-coding genes, ambient RNA removal, or wants built-in noise reduction",
    checkpoint_url="https://github.com/scprint/scPRINT",
    documentation_url="https://github.com/scprint/scPRINT",
    paper_url="",
    license_notes="Check upstream LICENSE; protein-coding genes focus, robust integration",
)

AIDOCELL_SPEC = ModelSpec(
    name="aidocell",
    version="v1.0",
    skill_ready=SkillReadyStatus.PARTIAL,
    tasks=[TaskType.EMBED, TaskType.INTEGRATE],
    modalities=[Modality.RNA],
    species=["human"],
    gene_id_scheme=GeneIDScheme.SYMBOL,
    requires_finetuning=False,
    zero_shot_embedding=True,
    zero_shot_annotation=False,
    output_keys=OutputKeys(
        embedding_key="X_aidocell",
        annotation_key="",
        confidence_key="",
    ),
    embedding_dim=512,
    hardware=HardwareRequirements(
        gpu_required=True,
        min_vram_gb=16,
        recommended_vram_gb=32,
        cpu_fallback=False,
        default_batch_size=32,
    ),
    differentiator="Dense transformer optimized for unsupervised cell clustering without predefined labels",
    prefer_when="User wants unsupervised clustering, label-free cell grouping, or dense transformer embeddings for discovery",
    checkpoint_url="https://github.com/genbio-ai/AIDO",
    documentation_url="https://github.com/genbio-ai/AIDO",
    paper_url="",
    license_notes="Check upstream LICENSE; dense transformer, zero-shot clustering",
)

PULSAR_SPEC = ModelSpec(
    name="pulsar",
    version="v1.0",
    skill_ready=SkillReadyStatus.PARTIAL,
    tasks=[TaskType.EMBED, TaskType.INTEGRATE],
    modalities=[Modality.RNA],
    species=["human"],
    gene_id_scheme=GeneIDScheme.SYMBOL,
    requires_finetuning=False,
    zero_shot_embedding=True,
    zero_shot_annotation=False,
    output_keys=OutputKeys(
        embedding_key="X_pulsar",
        annotation_key="",
        confidence_key="",
    ),
    embedding_dim=512,
    hardware=HardwareRequirements(
        gpu_required=True,
        min_vram_gb=16,
        recommended_vram_gb=32,
        cpu_fallback=False,
        default_batch_size=32,
    ),
    differentiator="Multi-scale multicellular biology modeling, captures cell-cell communication and tissue-level organization",
    prefer_when="User wants cell-cell communication analysis, tissue-level modeling, multicellular programs, or intercellular signaling",
    checkpoint_url="https://github.com/pulsar-ai/PULSAR",
    documentation_url="https://github.com/pulsar-ai/PULSAR",
    paper_url="",
    license_notes="Check upstream LICENSE; multi-scale, multicellular biology",
)

ATACFORMER_SPEC = ModelSpec(
    name="atacformer",
    version="v1.0",
    skill_ready=SkillReadyStatus.PARTIAL,
    tasks=[TaskType.EMBED, TaskType.INTEGRATE],
    modalities=[Modality.ATAC],
    species=["human"],
    gene_id_scheme=GeneIDScheme.CUSTOM,  # Peak-based, not gene-based
    requires_finetuning=False,
    zero_shot_embedding=True,
    zero_shot_annotation=False,
    output_keys=OutputKeys(
        embedding_key="X_atacformer",
        annotation_key="",
        confidence_key="",
    ),
    embedding_dim=512,
    hardware=HardwareRequirements(
        gpu_required=True,
        min_vram_gb=16,
        recommended_vram_gb=32,
        cpu_fallback=False,
        default_batch_size=32,
    ),
    differentiator="ATAC-seq-native transformer, peak-based (not gene-based) input, chromatin accessibility specialist",
    prefer_when="User has ATAC-seq data, chromatin accessibility profiles, or peak-based (not gene expression) inputs",
    checkpoint_url="https://github.com/Atacformer/Atacformer",
    documentation_url="https://github.com/Atacformer/Atacformer",
    paper_url="",
    license_notes="Check upstream LICENSE; ATAC-seq chromatin accessibility",
)

SCPLANTLLM_SPEC = ModelSpec(
    name="scplantllm",
    version="v1.0",
    skill_ready=SkillReadyStatus.PARTIAL,
    tasks=[TaskType.EMBED, TaskType.INTEGRATE],
    modalities=[Modality.RNA],
    species=["plant"],  # Plant-specific
    gene_id_scheme=GeneIDScheme.SYMBOL,
    requires_finetuning=False,
    zero_shot_embedding=True,
    zero_shot_annotation=False,
    output_keys=OutputKeys(
        embedding_key="X_scplantllm",
        annotation_key="",
        confidence_key="",
    ),
    embedding_dim=512,
    hardware=HardwareRequirements(
        gpu_required=True,
        min_vram_gb=16,
        recommended_vram_gb=32,
        cpu_fallback=False,
        default_batch_size=32,
    ),
    differentiator="Plant-specific single-cell model, handles polyploidy and plant gene nomenclature",
    prefer_when="User has plant single-cell data (Arabidopsis, rice, maize, etc.) or mentions polyploidy",
    checkpoint_url="https://github.com/scPlantLLM/scPlantLLM",
    documentation_url="https://github.com/scPlantLLM/scPlantLLM",
    paper_url="",
    license_notes="Check upstream LICENSE; plant-specific, handles polyploidy",
)

LANGCELL_SPEC = ModelSpec(
    name="langcell",
    version="v1.0",
    skill_ready=SkillReadyStatus.PARTIAL,
    tasks=[TaskType.EMBED, TaskType.INTEGRATE],
    modalities=[Modality.RNA],
    species=["human"],
    gene_id_scheme=GeneIDScheme.SYMBOL,
    requires_finetuning=False,
    zero_shot_embedding=True,
    zero_shot_annotation=False,
    output_keys=OutputKeys(
        embedding_key="X_langcell",
        annotation_key="",
        confidence_key="",
    ),
    embedding_dim=512,
    hardware=HardwareRequirements(
        gpu_required=True,
        min_vram_gb=16,
        recommended_vram_gb=32,
        cpu_fallback=False,
        default_batch_size=32,
    ),
    differentiator="Two-tower (text + cell) architecture, aligns natural language descriptions with cell embeddings",
    prefer_when="User wants text-guided cell retrieval, natural language cell queries, or text-cell alignment",
    checkpoint_url="https://github.com/langcell/LangCell",
    documentation_url="https://github.com/langcell/LangCell",
    paper_url="",
    license_notes="Check upstream LICENSE; two-tower architecture, text+cell alignment",
)

CELL2SENTENCE_SPEC = ModelSpec(
    name="cell2sentence",
    version="v1.0",
    skill_ready=SkillReadyStatus.PARTIAL,
    tasks=[TaskType.EMBED],
    modalities=[Modality.RNA],
    species=["human"],
    gene_id_scheme=GeneIDScheme.SYMBOL,
    requires_finetuning=True,  # LLM fine-tuning approach
    zero_shot_embedding=False,
    zero_shot_annotation=False,
    output_keys=OutputKeys(
        embedding_key="X_cell2sentence",
        annotation_key="",
        confidence_key="",
    ),
    embedding_dim=768,  # Typical LLM dimension
    hardware=HardwareRequirements(
        gpu_required=True,
        min_vram_gb=16,
        recommended_vram_gb=32,
        cpu_fallback=False,
        default_batch_size=16,
    ),
    differentiator="Converts cells to text sentences for LLM fine-tuning, 768-dim LLM embeddings",
    prefer_when="User wants to leverage general-purpose LLMs, convert cells to text, or use LLM fine-tuning workflows",
    checkpoint_url="https://github.com/vandijklab/cell2sentence",
    documentation_url="https://github.com/vandijklab/cell2sentence",
    paper_url="",
    license_notes="Check upstream LICENSE; flattens cells to text for LLM fine-tuning",
)

GENEPT_SPEC = ModelSpec(
    name="genept",
    version="v1.0",
    skill_ready=SkillReadyStatus.PARTIAL,
    tasks=[TaskType.EMBED],
    modalities=[Modality.RNA],
    species=["human"],
    gene_id_scheme=GeneIDScheme.SYMBOL,
    requires_finetuning=False,
    zero_shot_embedding=True,
    zero_shot_annotation=False,
    output_keys=OutputKeys(
        embedding_key="X_genept",
        annotation_key="",
        confidence_key="",
    ),
    embedding_dim=1536,  # GPT-3.5 embedding dimension
    hardware=HardwareRequirements(
        gpu_required=False,  # API-based, no local GPU needed
        min_vram_gb=0,
        recommended_vram_gb=0,
        cpu_fallback=True,
        default_batch_size=32,
    ),
    differentiator="API-based GPT-3.5 gene embeddings (1536-dim), no local GPU required, gene-level (not cell-level)",
    prefer_when="User wants gene-level embeddings (not cell-level), has no local GPU, or wants API-based OpenAI embeddings",
    checkpoint_url="https://github.com/yiqunchen/GenePT",
    documentation_url="https://github.com/yiqunchen/GenePT",
    paper_url="",
    license_notes="Check upstream LICENSE; uses GPT-3.5 embeddings, requires OpenAI API",
)

CHATCELL_SPEC = ModelSpec(
    name="chatcell",
    version="v1.0",
    skill_ready=SkillReadyStatus.PARTIAL,
    tasks=[TaskType.EMBED, TaskType.ANNOTATE],
    modalities=[Modality.RNA],
    species=["human"],
    gene_id_scheme=GeneIDScheme.SYMBOL,
    requires_finetuning=False,
    zero_shot_embedding=True,
    zero_shot_annotation=True,  # Chat-based annotation
    output_keys=OutputKeys(
        embedding_key="X_chatcell",
        annotation_key="chatcell_pred",
        confidence_key="",
    ),
    embedding_dim=512,
    hardware=HardwareRequirements(
        gpu_required=True,
        min_vram_gb=16,
        recommended_vram_gb=32,
        cpu_fallback=False,
        default_batch_size=16,
    ),
    differentiator="Conversational chat interface for single-cell analysis, zero-shot annotation via dialogue",
    prefer_when="User wants interactive chat-based cell analysis, conversational annotation, or dialogue-driven exploration",
    checkpoint_url="https://github.com/chatcell/CHATCELL",
    documentation_url="https://github.com/chatcell/CHATCELL",
    paper_url="",
    license_notes="Check upstream LICENSE; chat interface for single-cell data",
)

TABULA_SPEC = ModelSpec(
    name="tabula",
    version="federated-v1",
    skill_ready=SkillReadyStatus.PARTIAL,
    tasks=[TaskType.EMBED, TaskType.ANNOTATE, TaskType.INTEGRATE, TaskType.PERTURB],
    modalities=[Modality.RNA],
    species=["human"],
    gene_id_scheme=GeneIDScheme.CUSTOM,  # 60,697 gene vocabulary via vocab.json
    requires_finetuning=True,  # Annotation/perturbation require fine-tuning
    zero_shot_embedding=True,  # Embedding via forward pass without fine-tuning
    zero_shot_annotation=False,
    output_keys=OutputKeys(
        embedding_key="X_tabula",
        annotation_key="tabula_pred",
        confidence_key="tabula_pred_score",
        integration_key="X_tabula_integrated",
    ),
    embedding_dim=192,
    hardware=HardwareRequirements(
        gpu_required=True,
        min_vram_gb=8,
        recommended_vram_gb=16,
        cpu_fallback=False,
        default_batch_size=64,
    ),
    differentiator="Privacy-preserving federated learning + tabular transformer, 60697 gene vocabulary, quantile-binned expression, FlashAttention",
    prefer_when="User needs privacy-preserving analysis, federated-trained embeddings, or perturbation prediction with tabular modeling approach",
    checkpoint_url="https://github.com/aristoteleo/tabula",
    documentation_url="https://github.com/aristoteleo/tabula",
    paper_url="",  # Preprint: Ding et al., 2025
    license_notes="Check upstream LICENSE from aristoteleo/tabula",
)


# =============================================================================
# Model Registry Class
# =============================================================================

class ModelRegistry:
    """
    Registry of available single-cell foundation models.

    Provides capability-driven model discovery and selection.
    """

    def __init__(self):
        self._models: dict[str, ModelSpec] = {}
        self._register_default_models()

    def _register_default_models(self):
        """Register skill-ready and partial-spec models"""
        # Skill-ready (✅)
        self.register(SCGPT_SPEC)
        self.register(GENEFORMER_SPEC)
        self.register(UCE_SPEC)
        # Partial (⚠️) - Core Models
        self.register(SCFOUNDATION_SPEC)
        self.register(SCBERT_SPEC)
        self.register(GENECOMPASS_SPEC)
        self.register(CELLPLM_SPEC)
        self.register(NICHEFORMER_SPEC)
        self.register(SCMULAN_SPEC)
        # Partial (⚠️) - Specialized & Emerging (2024-2025)
        self.register(TGPT_SPEC)
        self.register(CELLFM_SPEC)
        self.register(SCCELLO_SPEC)
        self.register(SCPRINT_SPEC)
        self.register(AIDOCELL_SPEC)
        self.register(PULSAR_SPEC)
        self.register(ATACFORMER_SPEC)
        self.register(SCPLANTLLM_SPEC)
        self.register(LANGCELL_SPEC)
        self.register(CELL2SENTENCE_SPEC)
        self.register(GENEPT_SPEC)
        self.register(CHATCELL_SPEC)
        self.register(TABULA_SPEC)

    def register(self, spec: ModelSpec):
        """Register a model specification"""
        self._models[spec.name.lower()] = spec

    def get(self, name: str) -> Optional[ModelSpec]:
        """Get model spec by name"""
        return self._models.get(name.lower())

    def list_models(self, skill_ready_only: bool = False) -> list[ModelSpec]:
        """List all registered models"""
        models = list(self._models.values())
        if skill_ready_only:
            models = [m for m in models if m.skill_ready == SkillReadyStatus.READY]
        return models

    def find_models(
        self,
        task: Optional[TaskType] = None,
        modality: Optional[Modality] = None,
        species: Optional[str] = None,
        gene_scheme: Optional[GeneIDScheme] = None,
        zero_shot: bool = False,
        max_vram_gb: Optional[int] = None,
    ) -> list[ModelSpec]:
        """
        Find models matching criteria.

        Args:
            task: Required task (embed, annotate, etc.)
            modality: Required modality (RNA, ATAC, etc.)
            species: Required species support
            gene_scheme: Required gene ID scheme
            zero_shot: If True, only return zero-shot capable models
            max_vram_gb: Maximum VRAM constraint

        Returns:
            List of matching ModelSpecs, sorted by skill-ready status
        """
        matches = []

        for spec in self._models.values():
            # Filter by task
            if task and not spec.supports_task(task):
                continue

            # Filter by modality
            if modality and not spec.supports_modality(modality):
                continue

            # Filter by species
            if species and not spec.supports_species(species):
                continue

            # Filter by gene scheme
            if gene_scheme and spec.gene_id_scheme != gene_scheme:
                continue

            # Filter by zero-shot capability
            if zero_shot and task == TaskType.ANNOTATE and not spec.zero_shot_annotation:
                continue
            if zero_shot and task == TaskType.EMBED and not spec.zero_shot_embedding:
                continue

            # Filter by VRAM
            if max_vram_gb and spec.hardware.min_vram_gb > max_vram_gb:
                continue

            matches.append(spec)

        # Sort by skill-ready status (ready first)
        def sort_key(s: ModelSpec) -> int:
            if s.skill_ready == SkillReadyStatus.READY:
                return 0
            elif s.skill_ready == SkillReadyStatus.PARTIAL:
                return 1
            return 2

        return sorted(matches, key=sort_key)


# Global registry instance
_registry: Optional[ModelRegistry] = None


def get_registry() -> ModelRegistry:
    """Get the global model registry instance"""
    global _registry
    if _registry is None:
        _registry = ModelRegistry()
    return _registry
