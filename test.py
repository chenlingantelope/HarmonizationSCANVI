# import numpy as np
# X1 = np.random.normal(size=(100,100))
# X2 = np.random.normal(size=(100,100))
# Y1 = np.random.choice([1,2,3,4,5], 100)
# coral = CORAL()
# Y1 = coral.fit_predict(X1,Y1, X2)
#
plotname = 'Pancreas'
import numpy as np
from scvi.harmonization.classification.CORAL import CORAL
from scvi.dataset.dataset import GeneExpressionDataset
from scvi.harmonization.utils_chenling import SubsetGenes

import pickle as pkl
f = open('../%s/gene_dataset.pkl'%plotname, 'rb')
all_dataset, dataset1, dataset2 = pkl.load(f)
f.close()
all_dataset = GeneExpressionDataset.concat_datasets(dataset1,dataset2)
dataset1, dataset2, gene_dataset = SubsetGenes(dataset1, dataset2, all_dataset, plotname)



import time
from scvi.harmonization.utils_chenling import run_model
start = time.time()
latent, batch_indices, labels, keys, stats = run_model('scmap', gene_dataset, dataset1, dataset2,filename=plotname)
end = time.time()
print( end - start)



batch = gene_dataset.batch_indices.ravel()
labels = gene_dataset.labels.ravel()
scaling_factor = gene_dataset.X.mean(axis=1)
norm_X = gene_dataset.X / scaling_factor.reshape(len(scaling_factor), 1)
index_0 = np.where(batch == 0)[0]
index_1 = np.where(batch == 1)[0]
X1 = np.log(1 + norm_X[index_0])
X2 = np.log(1 + norm_X[index_1])

coral = CORAL()
pred1 = coral.fit_predict(X1, labels[index_0], X2)


