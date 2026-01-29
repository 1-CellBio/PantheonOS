# Distilled rules from Immune_All_Low.pkl
# Fidelity: ~52%


def predict_cell_type(expr):
    # expr: dict of gene expression
    if expr.get('TIMD4', 0) > 1.5 and expr.get('CETP', 0) > 1.5:
        return 'Kupffer cells'  # Liver macrophages
    if expr.get('RAMP3', 0) > 1.5 and expr.get('ACKR1', 0) > 1.5:
        return 'Endothelial cells'  # Vascular marker genes
    if expr.get('TPSAB1', 0) > 1.5 and expr.get('CPA3', 0) > 1.5:
        return 'Mast cells'  # Mast cell activation genes
    if expr.get('MT-RNR2', 0) > 1.5 and expr.get('RPL41', 0) > 1.5:
        return 'Follicular B cells'  # B cell markers
    if expr.get('MRC1', 0) > 1.5 and expr.get('MARCO', 0) > 1.5:
        return 'Alveolar macrophages'  # Lung macrophages
    if expr.get('JCHAIN', 0) > 1.5 and expr.get('MZB1', 0) > 1.5:
        return 'Plasma cells'  # Antibody secretion markers
    if expr.get('KIR2DL4', 0) > 1.5:
        return 'gamma-delta T cells'  # Innate-like T cell receptor

    return 'Unknown'
