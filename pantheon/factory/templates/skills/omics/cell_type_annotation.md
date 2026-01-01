---
id: cell_type_annotation
name: Cell Type Annotation
description: |
  Approaches for annotating cell types in single-cell data using
  marker genes and reference-based methods.
tags: [annotation, cell types, markers, scanpy]
---

# Cell Type Annotation Strategies

Methods for assigning cell type labels to clusters in single-cell RNA-seq data.

## Approach Overview

| Method | When to Use | Pros | Cons |
|--------|-------------|------|------|
| Marker genes | Known tissue, well-defined types | Interpretable, flexible | Requires domain knowledge |
| Reference mapping | Good reference available | Objective, comprehensive | Dependent on reference quality |
| Automated tools | Quick exploration | Fast, unbiased | May miss rare types |

## 1. Marker Gene-Based Annotation

### Step 1: Find Cluster Markers

```python
import scanpy as sc

# Compute differentially expressed genes per cluster
sc.tl.rank_genes_groups(
    adata, 
    groupby='leiden',  # or your cluster key
    method='wilcoxon',
    pts=True,  # Include percentage expressed
)

# View top markers for each cluster
sc.pl.rank_genes_groups(adata, n_genes=10, sharey=False)
plt.savefig('cluster_markers.png', dpi=150, bbox_inches='tight')
```

### Step 2: Extract and Review Markers

```python
# Get marker genes as DataFrame
markers = sc.get.rank_genes_groups_df(adata, group=None)

# Filter by log fold change and p-value
markers_filtered = markers[
    (markers['logfoldchanges'] > 1) & 
    (markers['pvals_adj'] < 0.01)
]

# Top 5 markers per cluster
top_markers = markers_filtered.groupby('group').head(5)
print(top_markers)
```

### Step 3: Visualize Known Markers

```python
# Define known cell type markers (example for PBMC)
markers_dict = {
    'T cells': ['CD3D', 'CD3E', 'CD4', 'CD8A'],
    'B cells': ['CD79A', 'MS4A1', 'CD19'],
    'NK cells': ['NKG7', 'GNLY', 'KLRD1'],
    'Monocytes': ['CD14', 'LYZ', 'S100A8'],
    'Dendritic': ['FCER1A', 'CST3', 'CLEC10A'],
    'Platelets': ['PPBP', 'PF4'],
}

# Dot plot
sc.pl.dotplot(adata, markers_dict, groupby='leiden', dendrogram=True)
plt.savefig('markers_dotplot.png', dpi=150, bbox_inches='tight')

# Stacked violin
sc.pl.stacked_violin(adata, markers_dict, groupby='leiden', rotation=90)
plt.savefig('markers_violin.png', dpi=150, bbox_inches='tight')
```

### Step 4: Assign Annotations

```python
# Create mapping from cluster to cell type
cluster_to_celltype = {
    '0': 'CD4+ T cells',
    '1': 'CD14+ Monocytes',
    '2': 'B cells',
    '3': 'NK cells',
    '4': 'CD8+ T cells',
    # ... add all clusters
}

# Apply annotation
adata.obs['cell_type'] = adata.obs['leiden'].map(cluster_to_celltype)

# Visualize
sc.pl.umap(adata, color='cell_type', title='Cell Type Annotation')
plt.savefig('celltype_umap.png', dpi=150, bbox_inches='tight')
```

## 2. Reference-Based Annotation

### Using CellTypist (Automated)

```python
import celltypist
from celltypist import models

# Download model (run once)
models.download_models(model='Immune_All_Low.pkl')

# Load model
model = models.Model.load('Immune_All_Low.pkl')

# Predict cell types
predictions = celltypist.annotate(
    adata,
    model=model,
    majority_voting=True,  # Smooth predictions over clusters
)

# Add predictions to adata
adata.obs['celltypist_label'] = predictions.predicted_labels.majority_voting
```

### Using scANVI (Transfer Learning)

```python
import scvi

# Train on reference
scvi.model.SCANVI.setup_anndata(
    adata_ref,
    labels_key='cell_type',
    unlabeled_category='Unknown',
)
model = scvi.model.SCANVI(adata_ref)
model.train()

# Transfer to query
scvi.model.SCANVI.setup_anndata(adata_query)
model_query = scvi.model.SCANVI.load_query_data(adata_query, model)
adata_query.obs['scANVI_pred'] = model_query.predict()
```

## 3. Create Annotation Table

```python
import pandas as pd

# Create annotation summary
annotation_table = pd.DataFrame({
    'Cluster': sorted(adata.obs['leiden'].unique()),
    'Cell Type': [cluster_to_celltype.get(c, 'Unknown') for c in sorted(adata.obs['leiden'].unique())],
    'n_cells': adata.obs.groupby('leiden').size().values,
    'Top Markers': ['GENE1, GENE2, GENE3' for _ in adata.obs['leiden'].unique()],
    'Confidence': ['High', 'Medium', 'High', ...]  # Your assessment
})

annotation_table.to_csv('cell_type_annotations.csv', index=False)
print(annotation_table.to_markdown())
```

## Quality Checks

After annotation, verify the quality of your cell type assignments.

> [!IMPORTANT]
> You should verify marker specificity before proceeding with downstream analysis.

### Validation Checklist

1. **Diagonal pattern check**: High expression should be on-diagonal (marker in its cluster)
2. **Off-diagonal contamination**: If markers appear in unrelated clusters, investigate:
   - Ambient RNA contamination (see Ambient RNA section in quality_control skill)
   - Doublet contamination
   - True biological co-expression
3. **Expression scale**: If one marker has much higher scale than others, 
   this may indicate ambient RNA contamination

### Verify Marker Specificity

```python
# Heatmap of marker expression
sc.pl.heatmap(
    adata,
    var_names=[m for markers in markers_dict.values() for m in markers],
    groupby='cell_type',
    cmap='viridis',
    dendrogram=True,
)
plt.savefig('markers_heatmap.png', dpi=150, bbox_inches='tight')
```

### Check for Mixed Clusters

```python
# Look for clusters expressing multiple lineage markers
mixed_markers = {
    'T_cell': ['CD3D'],
    'B_cell': ['CD79A'],
    'Myeloid': ['CD14'],
}

for cluster in adata.obs['leiden'].unique():
    cluster_data = adata[adata.obs['leiden'] == cluster]
    print(f"\nCluster {cluster}:")
    for lineage, genes in mixed_markers.items():
        for gene in genes:
            if gene in cluster_data.var_names:
                expr = cluster_data[:, gene].X.mean()
                pct = (cluster_data[:, gene].X > 0).mean() * 100
                print(f"  {lineage} ({gene}): mean={expr:.2f}, %exp={pct:.1f}%")
```

### Action Required

- If contamination detected: Return to QC and apply correction
- If clusters cannot be distinguished: Consider merging
- Document your observations in the analysis report

## Tips

> [!TIP]
> - Always verify annotations with multiple marker genes
> - Consider sub-clustering for large heterogeneous clusters
> - Use literature to validate marker selections
> - For novel tissues, combine marker-based and reference-based approaches

> [!WARNING]
> - Marker genes can be dataset-specific
> - Some gene names differ between human and mouse (case sensitivity)
> - Rare populations may be missed or merged with larger clusters
