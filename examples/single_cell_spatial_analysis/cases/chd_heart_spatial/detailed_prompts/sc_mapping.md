# Single-Cell Mapping

Goal: Mapping the single-cell data to the spatial data to enhance the spatial data.

Datasets:
- Single-cell data: human_fetal_heart.h5ad
- Spatial data: merfish_human_heart_3d.h5ad

Detailed tasks:

1. Map the single-cell data to the spatial data with moscot.
2. Draw a sanky plot to visualize the mapping relationship between the single-cell data and the spatial data.
3. Do some quantitative analysis to evaluate the mapping quality, for example, calculate the accuracy, precision, recall, F1 score, etc.
Note the cell type label in the different datasets are different,
you should review the cell type labels in each dataset,
and generate a table to represent the correspondence of cell type labels across different datasets.
Then evaluate the mapping quality using the information in the table.
4. Visualize the original spatial data's cell type distribution and the mapped single-cell
data's cell type distribution on the spatial data's spatial coordinates with pyvista.
If the data is 3D, draw the 3D scatter plot.
