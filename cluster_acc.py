from scvi.harmonization.utils_chenling import run_model, SubsetGenes
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
import numpy as np
import pickle as pkl
import sys
sys.path.append("/data/yosef2/users/chenling/HarmonizationSCANVI")


def computeARI(latent1, latent2, latent, cluster1, cluster2, batch_indices):
    c11 = cluster1.fit_predict(latent1)
    c12 = cluster1.fit_predict(latent[batch_indices == 0])
    c21 = cluster2.fit_predict(latent2)
    c22 = cluster2.fit_predict(latent[batch_indices == 1])
    return (adjusted_rand_score(c11, c12), adjusted_rand_score(c21, c22))


def RunClusterAcc(dataset1, dataset2, gene_dataset,plotname):
    cluster1 = KMeans(len(dataset1.cell_types))
    cluster2 = KMeans(len(dataset2.cell_types))

    latent1 = np.genfromtxt('../harmonization/Seurat_data/' + plotname + '.1.CCA.txt')
    latent2 = np.genfromtxt('../harmonization/Seurat_data/' + plotname + '.2.CCA.txt')
    latent, batch_indices, labels, keys, stats = run_model('readSeurat', gene_dataset, dataset1, dataset2,
                                                           filename=plotname)
    res_seurat = computeARI(latent1, latent2, latent, cluster1, cluster2, batch_indices)

    latent, batch_indices, labels, keys, stats = run_model('MNN', gene_dataset, dataset1, dataset2,
                                                           filename=plotname)
    res_MNN = computeARI(latent1, latent2, latent, cluster1, cluster2, batch_indices)

    latent, batch_indices, labels, keys, stats = run_model('PCA', gene_dataset, dataset1, dataset2,
                                                           filename=plotname)
    res_PCA = computeARI(latent1, latent2, latent, cluster1, cluster2, batch_indices)

    dataset1, dataset2, gene_dataset = SubsetGenes(dataset1, dataset2, gene_dataset, plotname)
    latent1, _, _, _, _ = run_model('vae', dataset1, 0, 0, filename=plotname, rep='vae1')
    latent2, _, _, _, _ = run_model('vae', dataset2, 0, 0, filename=plotname, rep='vae2')
    latent, batch_indices, labels, keys, stats = run_model('vae', gene_dataset, dataset1, dataset2, filename=plotname)
    res_scvi = computeARI(latent1, latent2, latent, cluster1, cluster2, batch_indices)

    latent, batch_indices, labels, keys, stats = run_model('vae_nb', gene_dataset, dataset1, dataset2,
                                                           filename=plotname)
    res_scvi_nb = computeARI(latent1, latent2, latent, cluster1, cluster2, batch_indices)

    latent, batch_indices, labels, keys, stats = run_model('scanvi1', gene_dataset, dataset1, dataset2,
                                                           filename=plotname)
    res_scanvi1 = computeARI(latent1, latent2, latent, cluster1, cluster2, batch_indices)

    latent, batch_indices, labels, keys, stats = run_model('scanvi2', gene_dataset, dataset1, dataset2,
                                                           filename=plotname)
    res_scanvi2 = computeARI(latent1, latent2, latent, cluster1, cluster2, batch_indices)

    res = [res_scvi, res_scvi_nb, res_scanvi1, res_scanvi2, res_seurat, res_MNN, res_PCA]
    res = np.asarray(res)
    np.savetxt("%s.clusterScore.csv" % (plotname), res, "%.4f", ',')


plotname = 'DentateGyrus'
from scvi.dataset.dataset import GeneExpressionDataset
from scvi.dataset.MouseBrain import DentateGyrus10X, DentateGyrusC1

dataset1= DentateGyrus10X()
dataset1.subsample_genes(dataset1.nb_genes)
dataset2 = DentateGyrusC1()
dataset2.subsample_genes(dataset2.nb_genes)
gene_dataset = GeneExpressionDataset.concat_datasets(dataset1,dataset2)
RunClusterAcc(dataset1,dataset2,gene_dataset,plotname)

plotname = 'Pancreas'
f = open('../%s/gene_dataset.pkl'%plotname, 'rb')
gene_dataset, dataset1, dataset2 = pkl.load(f)
f.close()
RunClusterAcc(dataset1,dataset2,gene_dataset,plotname)

from scvi.dataset.pbmc import PbmcDataset
from scvi.harmonization.utils_chenling import get_matrix_from_dir,assign_label
plotname = 'PBMC8KCITE'
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
set(dataset2.cell_types).intersection(set(dataset2.cell_types))
dataset1.subsample_genes(dataset1.nb_genes)
dataset2.subsample_genes(dataset2.nb_genes)
gene_dataset = GeneExpressionDataset.concat_datasets(dataset1, dataset2)
RunClusterAcc(dataset1,dataset2,gene_dataset,plotname)

plotname = 'MarrowTM'
from scvi.dataset.muris_tabula import TabulaMuris
dataset1 = TabulaMuris('facs', save_path='/data/yosef2/scratch/chenling/scanvi_data/')
dataset2 = TabulaMuris('droplet', save_path='/data/yosef2/scratch/chenling/scanvi_data/')
dataset1.subsample_genes(dataset1.nb_genes)
dataset2.subsample_genes(dataset2.nb_genes)
gene_dataset = GeneExpressionDataset.concat_datasets(dataset1, dataset2)
RunClusterAcc(dataset1,dataset2,gene_dataset,plotname)
