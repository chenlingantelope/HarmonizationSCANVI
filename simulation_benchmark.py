from scvi.harmonization.utils_chenling import CompareModels
import sys
models = str(sys.argv[1])
plotname = 'sim'

from scvi.dataset import GeneExpressionDataset
import numpy as np
import pandas as pd
import os
save_path = "../symsim_scVI/symsim_result/DE/"

label_array = pd.read_csv(os.path.join(save_path, "DE.cell_meta.csv"),
                          sep=",", index_col=0)["pop"].values

batch_array = pd.read_csv(os.path.join(save_path, "DE.batchid.csv"),
                          sep=",", index_col=0)["x"].values
batch_array -= 1
batch_array = batch_array[:, np.newaxis]
count_matrix = pd.read_csv(os.path.join(save_path, "DE.obsv.2.csv"),
                           sep=",", index_col=0).T

gene_names = np.array(count_matrix.columns, dtype=str)

dataset1 = GeneExpressionDataset(*GeneExpressionDataset.get_attributes_from_matrix(
    count_matrix.values, labels=label_array,batch_indices=batch_array),
    gene_names=gene_names, cell_types=np.unique(label_array))

dataset1.update_cells(batch_array.ravel()==0)

count_matrix = pd.read_csv(os.path.join(save_path, "DE.obsv.4.csv"),
                           sep=",", index_col=0).T

dataset2 = GeneExpressionDataset(*GeneExpressionDataset.get_attributes_from_matrix(
    count_matrix.values, labels=label_array,batch_indices=batch_array),
    gene_names=gene_names, cell_types=np.unique(label_array))

dataset2.update_cells(batch_array.ravel()==1)

gene_dataset = GeneExpressionDataset.concat_datasets(dataset1, dataset2)
# gene_dataset.subsample_genes(500)
labels = [int(gene_dataset.cell_types[i])-1 for i in gene_dataset.labels.ravel()]
gene_dataset.labels = np.asarray(labels).reshape(len(labels),1)
gene_dataset.cell_types = dataset2.cell_types
# from scipy import sparse

# gene_dataset.X = sparse.csr_matrix(gene_dataset.X )
gene_dataset.gene_names = gene_dataset.gene_names.astype('int')
dataset1.gene_names = dataset1.gene_names.astype('int')
dataset2.gene_names = dataset2.gene_names.astype('int')
# dataset1, dataset2, gene_dataset = SubsetGenes(dataset1, dataset2, gene_dataset, plotname)

CompareModels(gene_dataset, dataset1, dataset2, plotname, models)
