use_cuda = True
from scvi.harmonization.utils_chenling import get_matrix_from_dir,assign_label
from scvi.dataset.pbmc import PbmcDataset

import numpy as np
from scvi.dataset.dataset import GeneExpressionDataset
from scvi.harmonization.utils_chenling import CompareModels
import sys
models = str(sys.argv[1])
plotname = 'Easy1'

dataset1 = PbmcDataset(filter_out_de_genes=False)
dataset1.update_cells(dataset1.batch_indices.ravel()==0)
dataset1.subsample_genes(dataset1.nb_genes)


count, geneid, cellid = get_matrix_from_dir('cite')
count = count.T.tocsr()
seurat = np.genfromtxt('../cite/cite.seurat.labels', dtype='str', delimiter=',')
cellid = np.asarray([x.split('-')[0] for x in cellid])
labels_map = [0, 0, 1, 2, 3, 4, 5, 6]
labels = seurat[1:, 4]
cell_type = ['CD4 T cells', 'NK cells', 'CD14+ Monocytes', 'B cells','CD8 T cells', 'FCGR3A+ Monocytes', 'Other']
dataset2 = assign_label(cellid, geneid, labels_map, count, cell_type, seurat)
set(dataset2.cell_types).intersection(set(dataset2.cell_types))

dataset1.subsample_genes(dataset1.nb_genes)
dataset2.subsample_genes(dataset2.nb_genes)
gene_dataset = GeneExpressionDataset.concat_datasets(dataset1, dataset2)

import numpy as np
f = open("../%s/celltypeprop.txt"%plotname, "w+")
f.write("%s\t"*len(gene_dataset.cell_types)%tuple(gene_dataset.cell_types)+"\n")
freq = [np.mean(gene_dataset.labels.ravel()==i) for i in np.unique(gene_dataset.labels.ravel())]
f.write("%f\t"*len(gene_dataset.cell_types)%tuple(freq)+"\n")
freq1 = [np.mean(gene_dataset.labels.ravel()[gene_dataset.batch_indices.ravel()==0]==i) for i in np.unique(gene_dataset.labels.ravel())]
f.write("%f\t"*len(gene_dataset.cell_types)%tuple(freq1)+"\n")
freq2 = [np.mean(gene_dataset.labels.ravel()[gene_dataset.batch_indices.ravel()==1]==i) for i in np.unique(gene_dataset.labels.ravel())]
f.write("%f\t"*len(gene_dataset.cell_types)%tuple(freq2)+"\n")
f.close()

CompareModels(gene_dataset, dataset1, dataset2, plotname, models)

