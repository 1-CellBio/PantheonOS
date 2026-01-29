# File: distilled_code.py
"""
Distilled classifier for CellTypist Immune_All_Low model.

Target fidelity: >= 95% agreement with original model.
"""

import math

LOG1P = math.log1p

# Per-class bias terms (to be learned offline)
BIAS = {
    "Plasma cells": -0.25,
    "Mast cells": -0.20,
    "DC1": -0.15,
    "Kupffer cells": -0.10,
    "pDC": -0.15,
    "gamma-delta T cells": -0.10,
    "Endothelial cells": -0.05,
    "Follicular B cells": -0.20,
    "Alveolar macrophages": -0.10,
    "Neutrophil-myeloid progenitor": -0.10,
}

# Per-class sparse linear weights over log1p(expression[gene]) features.
# These should be learned via cross-entropy distillation to the teacher model.
WEIGHTS = {
    "Plasma cells": {
        "JCHAIN": 1.2,
        "MZB1": 1.1,
        "XBP1": 1.0,
        "SDC1": 0.9,
        "TNFRSF17": 0.8,
        "IGHG1": 0.6,
        "IGKC": 0.5,
        "MS4A1": -0.8,
        "CD79A": -0.6,
        "CD74": -0.5,
        "HLA-DRA": -0.4,
        "CCR7": -0.3,
        "TRAC": -0.6,
        "NKG7": -0.6,
        # Fidelity-oriented marker additions
        "PRDM1": 0.9,
        "IRF4": 0.9,
        "IGHA1": 0.7,
        "IGHA2": 0.7,
        "IGHG3": 0.5,
        "IGHG4": 0.5,
        "TNFRSF13B": 0.6,
        "DERL3": 0.6,
        "TCL1A": -0.6,
        "IL3RA": -0.3,
    },
    "Mast cells": {
        "TPSAB1": 1.2,
        "TPSB2": 1.2,
        "CPA3": 1.1,
        "KIT": 0.9,
        "MS4A2": 1.0,
        "GATA2": 0.6,
        "NKG7": -0.6,
        "TRAC": -0.6,
        "MS4A1": -0.4,
        "LYZ": -0.3,
        "S100A8": -0.3,
        "S100A9": -0.3,
        # Fidelity-oriented marker additions
        "HDC": 0.7,
        "SRGN": 0.4,
        "MPO": -0.4,
        "ELANE": -0.4,
        "CD79A": -0.3,
    },
    "DC1": {
        "CLEC9A": 1.3,
        "XCR1": 1.2,
        "BATF3": 0.8,
        "CADM1": 0.7,
        "IRF8": 0.6,
        "IL3RA": -0.8,
        "CLEC4C": -0.8,
        "LILRA4": -0.8,
        "MARCO": -0.4,
        "MRC1": -0.4,
        "S100A8": -0.3,
        "S100A9": -0.3,
        # Fidelity-oriented marker additions
        "WDFY4": 0.9,
        "CLNK": 0.6,
        "IDO1": 0.3,
    },
    "Kupffer cells": {
        "MARCO": 0.8,
        "TIMD4": 1.0,
        "CD163": 0.7,
        "VSIG4": 0.9,
        "C1QC": 0.7,
        "C1QA": 0.6,
        "APOE": 0.6,
        "CLEC9A": -0.6,
        "XCR1": -0.6,
        "IL3RA": -0.5,
        "CLEC4C": -0.5,
        "S100A8": -0.4,
        "S100A9": -0.4,
        "FCGR3B": -0.5,
        # Fidelity-oriented marker additions
        "CLEC4F": 1.2,
        "LYVE1": 0.6,
        "STAB1": 0.5,
        "MSR1": 0.5,
        "C2": 0.4,
        "FABP4": -0.6,
        "PPARG": -0.5,
        "LPL": -0.6,
        "MPO": -0.4,
        "ELANE": -0.4,
    },
    "pDC": {
        "LILRA4": 1.3,
        "IL3RA": 1.2,
        "CLEC4C": 1.2,
        "GZMB": 0.7,
        "TCF4": 0.7,
        "SERPINF1": 0.5,
        "CLEC9A": -0.9,
        "XCR1": -0.8,
        "MARCO": -0.4,
        "MRC1": -0.4,
        "S100A8": -0.3,
        "S100A9": -0.3,
        "NKG7": -0.5,
        "TRAC": -0.5,
        # Fidelity-oriented marker additions
        "SPIB": 0.8,
        "IRF7": 0.6,
        "BATF3": -0.6,
    },
    "gamma-delta T cells": {
        "TRDC": 1.3,
        "TRGC1": 1.1,
        "TRGC2": 1.1,
        "TRAC": 0.7,
        "CD3D": 0.6,
        "CD3E": 0.6,
        "MS4A1": -0.6,
        "CD79A": -0.5,
        "JCHAIN": -0.6,
        "MZB1": -0.5,
        "S100A8": -0.3,
        "S100A9": -0.3,
        "FCGR3B": -0.4,
        # Fidelity-oriented marker additions
        "TRDV2": 0.8,
        "TRGV9": 0.7,
        "MPO": -0.3,
        "ELANE": -0.3,
    },
    "Endothelial cells": {
        "VWF": 1.1,
        "CDH5": 1.0,
        "PECAM1": 1.0,
        "KDR": 0.8,
        "EMCN": 0.7,
        "RAMP2": 0.7,
        "ENG": 0.6,
        "PTPRC": -0.9,
        "LYZ": -0.5,
        "TRAC": -0.5,
        "NKG7": -0.5,
        "MS4A1": -0.4,
        "S100A8": -0.3,
        "S100A9": -0.3,
        # Fidelity-oriented marker additions
        "PLVAP": 0.7,
        "ESAM": 0.6,
        "FLT1": 0.6,
        "MPO": -0.4,
        "ELANE": -0.4,
    },
    "Follicular B cells": {
        "MS4A1": 1.1,
        "CD79A": 1.0,
        "CD19": 0.9,
        "CD74": 0.7,
        "HLA-DRA": 0.6,
        "BANK1": 0.6,
        "CD37": 0.5,
        "JCHAIN": -0.9,
        "MZB1": -0.8,
        "XBP1": -0.7,
        "SDC1": -0.6,
        "TRAC": -0.5,
        "NKG7": -0.5,
        "S100A8": -0.3,
        "S100A9": -0.3,
        # Fidelity-oriented marker additions
        "TCL1A": 0.9,
        "CD22": 0.6,
        "BLK": 0.6,
        "HVCN1": 0.4,
        "BACH2": 0.4,
        "IGHM": 0.5,
        "PRDM1": -0.7,
        "IRF4": -0.7,
        "TNFRSF17": -0.7,
    },
    "Alveolar macrophages": {
        "FABP4": 1.1,
        "MRC1": 0.8,
        "PPARG": 0.7,
        "MARCO": 0.7,
        "APOE": 0.6,
        "C1QC": 0.6,
        "TIMD4": -0.7,
        "CLEC9A": -0.5,
        "XCR1": -0.5,
        "IL3RA": -0.4,
        "CLEC4C": -0.4,
        "S100A8": -0.4,
        "S100A9": -0.4,
        "FCGR3B": -0.5,
        # Fidelity-oriented marker additions
        "LPL": 0.9,
        "FABP5": 0.6,
        "LIPA": 0.4,
        "SLC40A1": 0.5,
        "CLEC4F": -0.9,
        "VSIG4": -0.5,
        "MPO": -0.4,
        "ELANE": -0.4,
    },
    "Neutrophil-myeloid progenitor": {
        "MPO": 1.2,
        "ELANE": 1.1,
        "AZU1": 0.9,
        "PRTN3": 0.9,
        "CTSG": 0.8,
        "S100A8": 0.7,
        "S100A9": 0.7,
        "FCGR3B": 0.9,
        "MS4A1": -0.5,
        "CD79A": -0.5,
        "TRAC": -0.5,
        "NKG7": -0.5,
        "MARCO": -0.4,
        "MRC1": -0.4,
        "VWF": -0.4,
        "CDH5": -0.4,
        # Fidelity-oriented marker additions
        "CSF3R": 0.9,
        "LTF": 0.8,
        "S100A12": 0.7,
        "MMP8": 0.6,
        "MMP9": 0.6,
        "LCN2": 0.6,
        "APOE": -0.4,
        "C1QC": -0.4,
        "PECAM1": -0.4,
    },
}

# Union of all genes used by any class (for per-sample feature caching).
ALL_GENES = frozenset({g for w in WEIGHTS.values() for g in w})


def predict_cell_type(expression: dict) -> str:
    """
    Predict cell type from gene expression.

    Args:
        expression: Dict mapping gene names to expression values.

    Returns:
        Predicted cell type name (must match CellTypist labels exactly).
    """
    # Distilled multinomial linear "teacher head":
    #
    #   score_c(x) = b_c + sum_g w_{c,g} * log1p(x_g)
    #
    # NOTE: For >=95% fidelity, you must (1) match the teacher preprocessing and
    # (2) fit WEIGHTS/BIAS offline to teacher probabilities/logits.

    # --- Preprocessing (Change 2) ---
    # CellTypist commonly uses library-size normalization (e.g., to 1e4) + log1p.
    # If your `expression` is already normalized/log1p, do preprocessing upstream
    # and pass those values here; otherwise this block helps align inputs.
    total = 0.0
    for v in expression.values():
        total += float(v)
    scale = 1e4 / total if total > 0.0 else 1.0

    # --- Performance: on-demand log1p caching (Change 4) + micro-opts (Change 5) ---
    log1p = LOG1P
    expr_get = expression.get
    weights_by_class = WEIGHTS
    biases = BIAS

    phi = {}

    def feature(gene: str) -> float:
        val = phi.get(gene)
        if val is None:
            val = log1p(float(expr_get(gene, 0.0)) * scale)
            phi[gene] = val
        return val

    best_label = None
    best_score = float("-inf")

    for label in biases:
        w = weights_by_class.get(label, {})
        score = float(biases[label])
        for gene, weight in w.items():
            score += float(weight) * feature(gene)
        if score > best_score:
            best_score = score
            best_label = label

    return best_label