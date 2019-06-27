use_cuda = True
from scvi.dataset.dataset import GeneExpressionDataset
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
from scvi.harmonization.utils_chenling import trainSCANVI,trainVAE
from scvi.models import SCANVI
from scvi.inference.annotation import AlternateSemiSupervisedTrainer,SemiSupervisedTrainer
import numpy as np
from scvi.harmonization.utils_chenling import SubsetGenes
from scvi.harmonization.classification.scmap import SCMAP

def SCANVI_acc(gene_dataset:GeneExpressionDataset, plotname: str,pred1,pred2,coral1,coral2, rep='0'):
    fname = '../%s/scanvi_acc.txt'%(plotname)
    methods = ['scanvi','scanvi1','scanvi2']
    f = open(fname, "w+")
    f.write('method\t' +  "%s\t" * len(gene_dataset.cell_types) % tuple(gene_dataset.cell_types) + "\n")
    for i,method in enumerate(methods):
        vae_posterior = trainVAE(gene_dataset,plotname,rep)
        scanvi = SCANVI(gene_dataset.nb_genes, gene_dataset.n_batches, gene_dataset.n_labels, n_layers=2)
        scanvi.load_state_dict(vae_posterior.model.state_dict(), strict=False)
        if method=='scanvi1':
            trainer_scanvi = AlternateSemiSupervisedTrainer(scanvi, gene_dataset, classification_ratio=10,
                                                   n_epochs_classifier=50, lr_classification=5 * 1e-3)
            trainer_scanvi.labelled_set = trainer_scanvi.create_posterior(indices=(gene_dataset.batch_indices == 0))
            trainer_scanvi.unlabelled_set = trainer_scanvi.create_posterior(indices=(gene_dataset.batch_indices == 1))
        elif method=='scanvi2':
            trainer_scanvi = AlternateSemiSupervisedTrainer(scanvi, gene_dataset, classification_ratio=10,
                                                   n_epochs_classifier=50, lr_classification=5 * 1e-3)
            trainer_scanvi.labelled_set = trainer_scanvi.create_posterior(indices=(gene_dataset.batch_indices == 1))
            trainer_scanvi.unlabelled_set = trainer_scanvi.create_posterior(indices=(gene_dataset.batch_indices == 0))
        else:
            trainer_scanvi = SemiSupervisedTrainer(scanvi, gene_dataset, classification_ratio=50,
                                                   n_epochs_classifier=1, lr_classification=5 * 1e-3)
        trainer_scanvi.train(n_epochs=5)
        labelled_idx = trainer_scanvi.labelled_set.indices
        unlabelled_idx = trainer_scanvi.unlabelled_set.indices
        full = trainer_scanvi.create_posterior(trainer_scanvi.model, gene_dataset, indices=np.arange(len(gene_dataset)))
        labels, labels_pred = full.sequential().compute_predictions()
        shared = set(labels[labelled_idx]).intersection(set(labels[unlabelled_idx]))
        acc = [np.mean(labels_pred[unlabelled_idx][labels[unlabelled_idx] == i] == i) for i in np.unique(labels)]
        for x in np.unique(labels):
            if x not in [*shared] and method!='scanvi':
                acc[x]=-1
        f.write(method + "\t" + "%.4f\t" * len(acc) % tuple(acc) + "\n")

    labels = gene_dataset.labels.ravel()
    batch = gene_dataset.batch_indices.ravel()
    acc = [np.mean(pred1[labels[batch == 1] == i] == i) for i in np.unique(labels)]
    f.write('scmap1' + "\t" + "%.4f\t" * len(acc) % tuple(acc) + "\n")
    acc = [np.mean(pred2[labels[batch == 0] == i] == i) for i in np.unique(labels)]
    f.write('scmap2' + "\t" + "%.4f\t" * len(acc) % tuple(acc) + "\n")
    acc = [np.mean(coral1[labels[batch == 1] == i] == i) for i in np.unique(labels)]
    f.write('coral1' + "\t" + "%.4f\t" * len(acc) % tuple(acc) + "\n")
    acc = [np.mean(coral2[labels[batch == 0] == i] == i) for i in np.unique(labels)]
    f.write('coral2' + "\t" + "%.4f\t" * len(acc) % tuple(acc) + "\n")
    f.close()

def labelpred(gene_dataset, dataset1, dataset2, plotname):
    print("Starting scmap")
    n_features = 500
    scmap = SCMAP()
    scmap.set_parameters(n_features=n_features)
    scmap.fit_scmap_cluster(gene_dataset, plotname, batch=0)
    pred1 = scmap.predict_scmap_cluster(gene_dataset, plotname, batch=1)
    scmap = SCMAP()
    scmap.set_parameters(n_features=n_features)
    scmap.fit_scmap_cluster(gene_dataset, plotname, batch=1)
    pred2 = scmap.predict_scmap_cluster(gene_dataset, plotname, batch=0)
    dataset1, dataset2, gene_dataset = SubsetGenes(dataset1, dataset2, gene_dataset, plotname)
    batch = gene_dataset.batch_indices.ravel()
    labels = gene_dataset.labels.ravel()
    scaling_factor = gene_dataset.X.mean(axis=1)
    norm_X = gene_dataset.X / scaling_factor.reshape(len(scaling_factor), 1)
    index_0 = np.where(batch == 0)[0]
    index_1 = np.where(batch == 1)[0]
    X1 = np.log(1 + norm_X[index_0])
    X2 = np.log(1 + norm_X[index_1])
    from scvi.harmonization.classification.CORAL import CORAL
    coral = CORAL()
    coral1 = coral.fit_predict(X1, labels[index_0], X2)
    coral = CORAL()
    coral2 = coral.fit_predict(X2, labels[index_1], X1)
    return pred1,pred2, coral1, coral2


# from scvi.dataset.muris_tabula import TabulaMuris
# plotname = 'MarrowTM'
# dataset1 = TabulaMuris('facs', save_path='/data/yosef2/scratch/chenling/scanvi_data/')
# dataset2 = TabulaMuris('droplet', save_path='/data/yosef2/scratch/chenling/scanvi_data/')
# dataset1.subsample_genes(dataset1.nb_genes)
# dataset2.subsample_genes(dataset2.nb_genes)
# gene_dataset = GeneExpressionDataset.concat_datasets(dataset1, dataset2)
#
#
# pred1, pred2, coral1, coral2 = labelpred(gene_dataset, dataset1, dataset2, plotname)
# SCANVI_acc(gene_dataset, plotname, pred1,pred2,coral1,coral2)

#
#
plotname = 'PBMC8KCITE'
from scvi.harmonization.utils_chenling import get_matrix_from_dir,assign_label
from scvi.dataset.pbmc import PbmcDataset
from scvi.dataset.dataset import GeneExpressionDataset
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
pred1, pred2, coral1, coral2 = labelpred(gene_dataset, dataset1, dataset2, plotname)
SCANVI_acc(gene_dataset, plotname, pred1,pred2,coral1,coral2)

# plotname='Pancreas'
# import pickle as pkl
# f = open('../%s/gene_dataset.pkl'%plotname, 'rb')
# gene_dataset, dataset1, dataset2 = pkl.load(f)
# f.close()
# dataset1, dataset2, gene_dataset = SubsetGenes(dataset1, dataset2, gene_dataset, plotname)
# pred1, pred2, coral1, coral2 = labelpred(gene_dataset, dataset1, dataset2, plotname)
# SCANVI_acc(gene_dataset, plotname, pred1,pred2,coral1,coral2)
#
#
# plotname = 'DentateGyrus'
# from scvi.dataset.dataset import GeneExpressionDataset
# from scvi.dataset.MouseBrain import DentateGyrus10X, DentateGyrusC1
# dataset1= DentateGyrus10X()
# dataset1.subsample_genes(dataset1.nb_genes)
# dataset2 = DentateGyrusC1()
# dataset2.subsample_genes(dataset2.nb_genes)
# gene_dataset = GeneExpressionDataset.concat_datasets(dataset1,dataset2)
# dataset1, dataset2, gene_dataset = SubsetGenes(dataset1, dataset2, gene_dataset, plotname)
# pred1, pred2, coral1, coral2 = labelpred(gene_dataset, dataset1, dataset2, plotname)
# SCANVI_acc(gene_dataset, plotname, pred1,pred2,coral1,coral2)
#
