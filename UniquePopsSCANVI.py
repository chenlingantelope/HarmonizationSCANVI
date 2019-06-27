use_cuda = True
from sklearn.neighbors import KNeighborsClassifier
from scipy.stats import entropy
from scvi.harmonization.utils_chenling import run_model
from copy import deepcopy

import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt
import seaborn as sns
from scvi.dataset.dataset import SubsetGenes

from umap import UMAP
from scvi.models.vae import VAE
from scvi.models.scanvi import SCANVI
from scvi.inference import UnsupervisedTrainer, AlternateSemiSupervisedTrainer
import torch
import os

from scvi.harmonization.utils_chenling import get_matrix_from_dir,assign_label
from scvi.dataset.pbmc import PbmcDataset

import numpy as np
from scvi.dataset.dataset import GeneExpressionDataset

plotname = 'UniquePops'
dataset1 = PbmcDataset(filter_out_de_genes=False)
dataset1.update_cells(dataset1.batch_indices.ravel()==0)
dataset1.subsample_genes(dataset1.nb_genes)

save_path='/data/yosef2/scratch/chenling/scanvi_data/'
count, geneid, cellid = get_matrix_from_dir(save_path + 'cite')
count = count.T.tocsr()
seurat = np.genfromtxt(save_path + 'cite/cite.seurat.labels', dtype='str', delimiter=',')
cellid = np.asarray([x.split('-')[0] for x in cellid])
labels_map = [0, 0, 1, 2, 3, 4, 5, 6]
labels = seurat[1:, 4]
cell_type = ['CD4 T cells', 'NK cells', 'CD14+ Monocytes', 'B cells','CD8 T cells', 'FCGR3A+ Monocytes', 'Other']
dataset2 = assign_label(cellid, geneid, labels_map, count, cell_type, seurat)

dataset1.subsample_genes(dataset1.nb_genes)
dataset2.subsample_genes(dataset2.nb_genes)

def JaccardIndex(x1,x2):
    intersection = np.sum(x1 * x2)
    union = np.sum((x1 + x2) > 0)
    return intersection / union

def entropy_from_indices(indices):
    return entropy(np.array(np.unique(indices, return_counts=True)[1].astype(np.int32)))


def entropy_batch_mixing_subsampled(latent, batches, labels, removed_type, n_neighbors=20, n_pools=1, n_samples_per_pool=100):
    X = latent[labels == removed_type,:]
    nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(latent)
    indices = nbrs.kneighbors(X, return_distance=False)[:, 1:]
    batch_indices = np.vectorize(lambda i: batches[i])(indices)
    entropies = np.apply_along_axis(entropy_from_indices, axis=1, arr=batch_indices)
    if n_pools == 1:
        res = np.mean(entropies)
    else:
        res = np.mean([
            np.mean(entropies[np.random.choice(len(entropies), size=n_samples_per_pool)])
            for _ in range(n_pools)
        ])
    return res


def plotUMAP(latent,plotname,method,keys, rmCellTypes,batch_indices):
    colors = sns.color_palette('tab20')
    # sample = select_indices_evenly(2000, labels)
    sample=np.arange(0,len(labels))
    latent_s = latent[sample, :]
    label_s = labels[sample]
    batch_s = batch_indices[sample]
    if latent_s.shape[1] != 2:
        latent_s = UMAP(spread=2).fit_transform(latent_s)
    fig, ax = plt.subplots(figsize=(10, 10))
    key_order = np.argsort(keys)
    for i, k in enumerate(key_order):
        ax.scatter(latent_s[label_s == k, 0], latent_s[label_s == k, 1], color=colors[i % 20], label=keys[k],
                   edgecolors='none')
        # ax.legend(bbox_to_anchor=(1.1, 0.5), borderaxespad=0, fontsize='x-large')
    ax.axis("off")
    fig.tight_layout()
    plt.savefig('../%s/%s.%s.%s.labels.pdf' % (plotname, plotname, method, rmCellTypes))
    plt.figure(figsize=(10, 10))
    plt.scatter(latent_s[:, 0], latent_s[:, 1], c=batch_s, edgecolors='none')
    plt.axis("off")
    plt.tight_layout()
    plt.savefig('../%s/%s.%s.%s.batches.pdf' % (plotname, plotname, method, rmCellTypes))
    plt.close()

from sklearn.neighbors import NearestNeighbors
def KNNpurity(latent1, latent2,latent,batchid,labels, keys, nn=50):
    knn = NearestNeighbors(n_neighbors=nn, algorithm='auto')
    nbrs1 = knn.fit(latent1)
    nbrs1 = nbrs1.kneighbors_graph(latent1).toarray()
    np.fill_diagonal(nbrs1,0)
    nbrs2 = knn.fit(latent2)
    nbrs2 = nbrs2.kneighbors_graph(latent2).toarray()
    np.fill_diagonal(nbrs2,0)
    nbrs_1 = knn.fit(latent[batchid==0,:])
    nbrs_1 = nbrs_1.kneighbors_graph(latent[batchid==0,:]).toarray()
    np.fill_diagonal(nbrs_1,0)
    nbrs_2 = knn.fit(latent[batchid==1,:])
    nbrs_2 = nbrs_2.kneighbors_graph(latent[batchid==1,:]).toarray()
    np.fill_diagonal(nbrs_2,0)
    JI1 = [JaccardIndex(x1, x2) for x1, x2 in zip(nbrs1, nbrs_1)]
    JI1 = np.asarray(JI1)
    JI2 = [JaccardIndex(x1, x2) for x1, x2 in zip(nbrs2, nbrs_2)]
    JI2 = np.asarray(JI2)
    res1 = [np.mean(JI1[labels[batchid==0] == i])  if np.sum(labels[batchid==0] == i)>0 else 0 for i in np.unique(labels)]
    res2 = [np.mean(JI2[labels[batchid==1] == i])  if np.sum(labels[batchid==1] == i)>0 else 0 for i in np.unique(labels)]
    res = (np.asarray(res1)+np.asarray(res2))/2
    cell_type = keys[np.unique(labels)]
    return res,cell_type

def KNNacc(latent, labels, keys):
    neigh = KNeighborsClassifier(n_neighbors=10)
    neigh = neigh.fit(latent, labels)
    labels_pred = neigh.predict(latent)
    acc = [np.mean(labels[labels_pred == i] == i) for i in np.unique(labels)]
    cell_type = keys[np.unique(labels)]
    return acc, cell_type

def BEbyType(keys,latent,labels,batch_indices, rmCellTypes):
    rm_idx = np.arange(len(keys))[keys == rmCellTypes][0]
    other_idx = np.arange(len(keys))[keys != rmCellTypes]
    cell_type = [keys[rm_idx]] + list(keys[other_idx])
    BE1 = entropy_batch_mixing_subsampled(latent, batch_indices, labels, removed_type=rm_idx)
    BE2 = [entropy_batch_mixing_subsampled(latent, batch_indices, labels, removed_type=i) for i in other_idx]
    res = [BE1] + BE2
    return(res, cell_type)


def trainVAE(gene_dataset, rmCellTypes,rep):
    vae = VAE(gene_dataset.nb_genes, n_batch=gene_dataset.n_batches, n_labels=gene_dataset.n_labels,
              n_hidden=128, n_latent=10, n_layers=2, dispersion='gene')
    trainer = UnsupervisedTrainer(vae, gene_dataset, train_size=1.0)
    if os.path.isfile('../UniquePops/vae.%s%s.pkl' % (rmCellTypes,rep)):
        trainer.model.load_state_dict(torch.load('../UniquePops/vae.%s%s.pkl' % (rmCellTypes,rep)))
        trainer.model.eval()
    else:
        trainer.train(n_epochs=150)
        torch.save(trainer.model.state_dict(), '../UniquePops/vae.%s%s.pkl' % (rmCellTypes,rep))
    full = trainer.create_posterior(trainer.model, gene_dataset, indices=np.arange(len(gene_dataset)))
    latent, batch_indices, labels = full.sequential().get_latent()
    batch_indices = batch_indices.ravel()
    return latent, batch_indices,labels,trainer


f = open('../UniquePops/'+plotname+'.kp.res.txt', "w+")
g = open('../UniquePops/'+plotname+'.res.txt', "w+")
# f.write('model_type\tcell_type\tBE_removed\tBE_kept\tasw\tnmi\tari\tca\twca\n')

# scp chenlingantelope@s128.millennium.berkeley.edu:/data/yosef2/users/chenling/harmonization/Seurat_data/UniquePops* .
for celltype1 in dataset2.cell_types[:6]:
    for celltype2 in dataset2.cell_types[:6]:
        if celltype1 != celltype2:
            print(celltype1+' '+celltype2)
            pbmc = deepcopy(dataset1)
            newCellType = [k for i, k in enumerate(dataset1.cell_types) if k not in [celltype1, 'Other']]
            pbmc.filter_cell_types(newCellType)
            pbmc2 = deepcopy(dataset2)
            newCellType = [k for i, k in enumerate(dataset2.cell_types) if k not in [celltype2, 'Other']]
            pbmc2.filter_cell_types(newCellType)
            gene_dataset = GeneExpressionDataset.concat_datasets(pbmc, pbmc2)
            # _,_,_,_,_ = run_model('writedata', gene_dataset, pbmc, pbmc2,filename=plotname+'.'
            #                                                                       +celltype1.replace(' ','')+'.'
            #                                                                       +celltype2.replace(' ',''))
            rmCellTypes = '.'+celltype1.replace(' ', '')+'.'+celltype2.replace(' ', '')
            latent1 = np.genfromtxt('../harmonization/Seurat_data/' + plotname + rmCellTypes.replace(' ','') + '.1.CCA.txt')
            latent2 = np.genfromtxt('../harmonization/Seurat_data/' + plotname + rmCellTypes.replace(' ','') + '.2.CCA.txt')
            latent, batch_indices, labels, keys, stats = run_model(
                'readSeurat', gene_dataset, pbmc, pbmc2, filename=plotname + rmCellTypes.replace(' ', ''))
            acc, cell_type = KNNpurity(latent1, latent2,latent,batch_indices.ravel(),labels,keys)
            f.write('Seurat' + '\t' + rmCellTypes + ("\t%.4f" * 8 + "\t%s" * 8 + "\n") % tuple(list(acc) + list(cell_type)))
            be, temp1 = BEbyType(keys, latent, labels, batch_indices, celltype1)
            g.write('Seurat' + '\t' + rmCellTypes + ("\t%.4f" * 8 + "\t%s" * 8 + "\n") % tuple(be + list(temp1)))
            plotUMAP(latent, plotname, 'Seurat', gene_dataset.cell_types, rmCellTypes, gene_dataset.batch_indices.ravel())
            pbmc, pbmc2, gene_dataset = SubsetGenes(pbmc, pbmc2, gene_dataset, plotname + rmCellTypes.replace(' ', ''))
            latent1, _, _, _ = trainVAE(pbmc, rmCellTypes, rep='1')
            latent2, _, _, _ = trainVAE(pbmc2, rmCellTypes, rep='2')
            latent, batch_indices, labels, trainer = trainVAE(gene_dataset, rmCellTypes, rep='')
            acc, cell_type = KNNpurity(latent1, latent2,latent,batch_indices.ravel(),labels,keys)
            f.write('vae' + '\t' + rmCellTypes + ("\t%.4f" * 8 + "\t%s" * 8 + "\n") % tuple(list(acc) + list(cell_type)))
            be, cell_type2 = BEbyType(keys, latent, labels, batch_indices,celltype1)
            g.write('vae' + '\t' + rmCellTypes + ("\t%.4f" * 8 + "\t%s" * 8 + "\n") % tuple(be + list(cell_type2)))
            plotUMAP(latent, plotname, 'vae', gene_dataset.cell_types, rmCellTypes, gene_dataset.batch_indices.ravel())
            scanvi = SCANVI(gene_dataset.nb_genes, 2, (gene_dataset.n_labels + 1),
                            n_hidden=128, n_latent=10, n_layers=2, dispersion='gene')
            scanvi.load_state_dict(trainer.model.state_dict(), strict=False)
            trainer_scanvi = AlternateSemiSupervisedTrainer(scanvi, gene_dataset, n_epochs_classifier=10, lr_classification=5 * 1e-3)
            trainer_scanvi.labelled_set = trainer_scanvi.create_posterior(indices=gene_dataset.batch_indices.ravel() == 0)
            trainer_scanvi.unlabelled_set = trainer_scanvi.create_posterior(indices=gene_dataset.batch_indices.ravel() == 1)
            trainer_scanvi.train(n_epochs=10)
            scanvi_full = trainer_scanvi.create_posterior(trainer_scanvi.model, gene_dataset, indices=np.arange(len(gene_dataset)))
            latent, _, _ = scanvi_full.sequential().get_latent()
            acc, cell_type = KNNpurity(latent1, latent2,latent,batch_indices.ravel(),labels,keys)
            f.write('scanvi' + '\t' + rmCellTypes + ("\t%.4f" * 8 + "\t%s" * 8 + "\n") % tuple(list(acc) + list(cell_type)))
            be, cell_type2 = BEbyType(keys, latent, labels, batch_indices, celltype1)
            g.write('scanvi' + '\t' + rmCellTypes + ("\t%.4f" * 8 + "\t%s" * 8 + "\n") % tuple(be + list(cell_type2)))
            plotUMAP(latent, plotname, 'scanvi', gene_dataset.cell_types, rmCellTypes, gene_dataset.batch_indices.ravel())
            f.flush()
            g.flush()

f.close()
g.close()
