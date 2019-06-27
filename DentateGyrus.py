plotname = 'DentateGyrus'
import sys
models = str(sys.argv[1])
import pickle as pkl
import numpy as np
import sys
sys.path.append("/data/yosef2/users/chenling/HarmonizationSCANVI")

from scvi.dataset.dataset import GeneExpressionDataset
from scvi.dataset.MouseBrain import DentateGyrus10X, DentateGyrusC1

dataset1= DentateGyrus10X()
dataset1.subsample_genes(dataset1.nb_genes)
dataset2 = DentateGyrusC1()
dataset2.subsample_genes(dataset2.nb_genes)
gene_dataset = GeneExpressionDataset.concat_datasets(dataset1,dataset2)

###########################################################################
# f = open('../%s/gene_dataset.pkl'%plotname, 'wb')
# pkl.dump(gene_dataset, f)
# f.close()
# f = open('../%s/gene_dataset.pkl'%plotname, 'rb')
# gene_dataset = pkl.load(f)
# f.close()
###########################################################################

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

from scvi.harmonization.utils_chenling import CompareModels
CompareModels(gene_dataset, dataset1, dataset2, plotname, models)

###########################################################################
# from scipy.sparse import csr_matrix
# from scvi.harmonization.utils_chenling import CompareModels, VAEstats
# gene_dataset.X = csr_matrix(gene_dataset.X)
# CompareModels(gene_dataset, dataset1, dataset2, plotname, models)
# latent, batch_indices, labels, stats = VAEstats(full)
# dropout, mean, disp = full.generate_parameters()
#
###########################################################################
