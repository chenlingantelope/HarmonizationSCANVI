import pickle as pkl
plotname = 'DentateGyrus'
from scvi.dataset.MouseBrain import DentateGyrus10X, DentateGyrusC1
from scvi.dataset.dataset import GeneExpressionDataset

dataset1= DentateGyrus10X()
dataset1.subsample_genes(dataset1.nb_genes)
dataset2 = DentateGyrusC1()
dataset2.subsample_genes(dataset2.nb_genes)
gene_dataset = GeneExpressionDataset.concat_datasets(dataset1,dataset2)

from scvi.dataset.dataset import SubsetGenes
dataset1, dataset2, gene_dataset = SubsetGenes(dataset1, dataset2, gene_dataset, plotname)

magan = MAGAN(dim_b1=10, dim_b2=10, correspondence_loss=correspondence_loss)
from scvi.harmonization.clustering.Magan import Magan
from utils import now
from model import MAGAN
from loader import Loader
import tensorflow as tf
from sklearn.decomposition import PCA
magan = Magan()
out1, out2 = magan.fit_transform(gene_dataset.X, gene_dataset.batch_indices.ravel(), [0, 1])

import numpy as np
batch = gene_dataset.batch_indices.ravel()
index_0 = np.where(batch == 0)[0]
index_1 = np.where(batch == 1)[0]

X = gene_dataset.X

X1 = np.log(1 + X[index_0])
X2 = np.log(1 + X[index_1])

loadb1 = Loader(X1, shuffle=True)
loadb2 = Loader(X2, shuffle=True)
# Build the tf graph

def correspondence_loss(b1, b2):
    """
    The correspondence loss.
    :param b1: a tensor representing the object in the graph of the current minibatch from domain one
    :param b2: a tensor representing the object in the graph of the current minibatch from domain two
    :returns a scalar tensor of the correspondence loss
    """
    domain1cols = [0]
    domain2cols = [0]
    loss = tf.constant(0.)
    for c1, c2 in zip(domain1cols, domain2cols):
        loss += tf.reduce_mean((b1[:, c1] - b2[:, c2])**2)

    return loss


magan = MAGAN(dim_b1=X1.shape[1], dim_b2=X2.shape[1], correspondence_loss=correspondence_loss)
# Train
for i in range(1, self.n_epochs):
    if i % 100 == 0: print("Iter {} ({})".format(i, now()))
    xb1_ = loadb1.next_batch(self.batch_size)
    xb2_ = loadb2.next_batch(self.batch_size)
    magan.train(xb1_, xb2_)


latent = PCA(n_components=10).fit_transform(out1)
# np.save('../' + filename + '/' + 'MNN' + '.npy', latent)