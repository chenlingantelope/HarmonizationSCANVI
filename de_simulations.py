from scvi.dataset import GeneExpressionDataset
from scvi.models import VAE
from scvi.inference import UnsupervisedTrainer, AlternateSemiSupervisedTrainer
from scvi.inference.posterior import get_IS_bayes_factors
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr
from sklearn.neighbors import KNeighborsClassifier
from scipy.stats import kendalltau
from scvi.models.scanvi import SCANVI
import pickle

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os
os.chdir('/data/yosef2/users/chenling/HarmonizationSCANVI')

import sys
rep = int(sys.argv[1])

def transfer_nn_labels(latent_array, labels_array, batch_indices_array):
    # Transfer labels from batch 0 to batch 1 using scVI
    latent_labelled = latent_array[batch_indices_array.ravel() == 0, :]
    labels_labelled = labels_array[batch_indices_array.ravel() == 0]
    neigh = KNeighborsClassifier(n_neighbors=10)
    neigh = neigh.fit(latent_labelled, labels_labelled)
    return neigh.predict(latent_array)


def get_bayes_factor_scvi(cset_a, cset_b, sampling_n, cells_sampled, use_is, force_batch=None):
    subset_a = np.random.choice(cset_a, cells_sampled)
    subset_b = np.random.choice(cset_b, cells_sampled)
    posterior_a = trainer.create_posterior(trainer.model, gene_dataset,
                                           indices=subset_a)
    posterior_b = trainer.create_posterior(trainer.model, gene_dataset,
                                           indices=subset_b)
    px_scale_a, log_ratios_a, labels_a = posterior_a.differential_expression_stats(M_sampling=sampling_n,
                                                                                   force_batch=force_batch)
    px_scale_b, log_ratios_b, labels_b = posterior_b.differential_expression_stats(M_sampling=sampling_n,
                                                                                   force_batch=force_batch)
    px_scale = np.concatenate((px_scale_a, px_scale_b), axis=1)
    log_ratios = np.concatenate((log_ratios_a, log_ratios_b), axis=1)
    labels_de = np.concatenate((0 * labels_a, 0 * labels_b + 1), axis=0)
    return get_IS_bayes_factors(px_scale, log_ratios, labels_de, 0,
                                other_cell_idx=1,
                                importance_sampling=use_is, permutation=False)


def eval_bayes_factor(log_fold_change, bayes_f):
    """
    :param log_fold_change: groundtruth
    :param bayes_f: non-log Bayes Factor
    :return:
    """
    bayes_f = np.log(bayes_f + 1e-8) - np.log(1 - bayes_f + 1e-8)
    auc_1 = roc_auc_score(np.abs(log_fold_change) >= 0.6, np.abs(bayes_f))
    auc_2 = roc_auc_score(np.abs(log_fold_change) >= 0.8, np.abs(bayes_f))
    spear = spearmanr(bayes_f, log_fold_change)[0]
    kend = kendalltau(bayes_f, log_fold_change)[0]
    return auc_1, auc_2, spear, kend


save_path = "../symsim_scVI/symsim_result/DE/"

# count_matrix = pd.read_csv(os.path.join(save_path, "DE.obsv.csv"),
#                            sep=",", index_col=0).T

label_array = pd.read_csv(os.path.join(save_path, "DE.cell_meta.csv"),
                          sep=",", index_col=0)["pop"].values

batch_array = pd.read_csv(os.path.join(save_path, "DE.batchid.csv"),
                          sep=",", index_col=0)["x"].values
# Renumerate the batches to be between 0 and N-batches
batch_array -= 1
batch_array = batch_array[:, np.newaxis]

# gene_names = np.array(count_matrix.columns, dtype=str)
#
# gene_dataset = GeneExpressionDataset(*GeneExpressionDataset.get_attributes_from_matrix(
#     count_matrix.values, labels=label_array,
#     batch_indices=batch_array),
#                                      gene_names=gene_names, cell_types=np.unique(label_array))

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

theoretical_FC = pd.read_csv(os.path.join(save_path, "theoreticalFC.csv"),
                             sep=",", index_col=0, header=0)
nevf = pd.read_csv(os.path.join(save_path, "n_evf.csv"),
                             sep=",", index_col=0, header=0)

# for key in theoretical_FC.columns:
#     log_FC = theoretical_FC[key]
#     plt.hist(log_FC)
#     detected_genes = np.sum(np.abs(log_FC) >= 0.8)
#     plt.title(key + ": " + str(detected_genes) + " genes / " + str(log_FC.shape[0]))
#     plt.axvline(x=-0.8)
#     plt.axvline(x=0.8)
#     plt.savefig(os.path.join(save_path, key + ".png"))
#     plt.clf()

# vae = VAE(gene_dataset.nb_genes, n_batch=gene_dataset.n_batches, reconstruction_loss="zinb", n_latent=10)
vae = VAE(gene_dataset.nb_genes, n_batch=gene_dataset.n_batches, reconstruction_loss="nb", n_latent=10)
trainer = UnsupervisedTrainer(vae,
                              gene_dataset,
                              train_size=0.75,
                              use_cuda=True,
                              frequency=5)

# file_name = '%s/vae.pkl' % save_path
# if os.path.isfile(file_name):
#     print("loaded model from: " + file_name)
#     trainer.model.load_state_dict(torch.load(file_name))
#     trainer.model.eval()
# else:
#     print('file not found')
    # train & save
n_epochs = 100
trainer.train(n_epochs=n_epochs, lr=0.001)
torch.save(trainer.model.state_dict(), save_path+'SIM.%i.nb.pkl'%rep)
#
# # write training info
# ll_train_set = trainer.history["ll_train_set"][1:]
# ll_test_set = trainer.history["ll_test_set"][1:]
# x = np.linspace(1, n_epochs, (len(ll_train_set)))
# plt.plot(x, ll_train_set)
# plt.plot(x, ll_test_set)
# plt.title("training ll")
# plt.savefig(os.path.join(save_path, "loss_training.png"))
# plt.clf()

# get latent space
full = trainer.create_posterior(trainer.model, gene_dataset, indices=np.arange(len(gene_dataset)))
latent, batch_indices, labels = full.sequential().get_latent()
# n_samples_tsne = 4000
# full.show_t_sne(n_samples=n_samples_tsne, color_by='batches and labels',
#                 save_name=os.path.join(save_path, "scVI_tSNE_batches_labels.png"))

print("Transferring labels from scVI")

true_labels = labels.ravel()
scVI_labels = transfer_nn_labels(latent, labels, batch_indices)

# train scANVI
print("Training scANVI")
# scanvi = SCANVI(gene_dataset.nb_genes, gene_dataset.n_batches, gene_dataset.n_labels, n_latent=10)
scanvi = SCANVI(gene_dataset.nb_genes, gene_dataset.n_batches, gene_dataset.n_labels, n_latent=10,
                reconstruction_loss='nb')
scanvi.load_state_dict(trainer.model.state_dict(), strict=False)
trainer_scanvi = AlternateSemiSupervisedTrainer(scanvi, gene_dataset,
                                                n_epochs_classifier=5, lr_classification=5 * 1e-3)
labelled = np.where(gene_dataset.batch_indices == 0)[0]
# np.random.shuffle(labelled)
unlabelled = np.where(gene_dataset.batch_indices == 1)[0]
# np.random.shuffle(unlabelled)
trainer_scanvi.labelled_set = trainer_scanvi.create_posterior(indices=labelled)
trainer_scanvi.unlabelled_set = trainer_scanvi.create_posterior(indices=unlabelled)

# file_name = '%s/scanvi.pkl' % save_path
# if os.path.isfile(file_name):
#     print("loaded model from: " + file_name)
#     trainer_scanvi.model.load_state_dict(torch.load(file_name))
#     trainer_scanvi.model.eval()
# else:
# train & save
trainer_scanvi.train(n_epochs=5)
# torch.save(trainer_scanvi.model.state_dict(), file_name)

scanvi_labels = trainer_scanvi.full_dataset.sequential().compute_predictions()[1]

predicted_labels = pd.DataFrame([gene_dataset.labels.ravel(),scVI_labels, scanvi_labels],index=['labels','scVI','scANVI'])
predicted_labels.T.to_csv(save_path+'pred_labels.nb.%i.csv'%rep)

# get latent space
full_scanvi = trainer.create_posterior(trainer_scanvi.model, gene_dataset, indices=np.arange(len(gene_dataset)))
latent, batch_indices, labels = full_scanvi.sequential().get_latent()
# n_samples_tsne = 4000
# full_scanvi.show_t_sne(n_samples=n_samples_tsne, color_by='batches and labels',
#                        save_name=os.path.join(save_path, "scANVI_tSNE_batches_labels.png"))
#
print("OVERLAP scANVI = scVI ", np.mean(scanvi_labels == scVI_labels))
print("accuracy scVI ", np.mean(gene_dataset.labels.ravel() == scVI_labels))
print("accuracy scANVI ", np.mean(gene_dataset.labels.ravel() == scanvi_labels))

# prepare for differential expression
cell_types = gene_dataset.cell_types
couple_celltypes_list = [(0, 1), (1, 2), (1, 3), (3, 4)]
results_DE_scVI_A = {}
results_DE_scVI_B = {}
results_DE_scVI_AB = {}
results_DE_true_A = {}
results_DE_true_B = {}
results_DE_true_AB = {}
results_DE_scANVI = {}
for result in (results_DE_scVI_A, results_DE_scVI_B, results_DE_scVI_AB, results_DE_scANVI,
               results_DE_true_A,results_DE_true_B,results_DE_true_AB):
    for key in theoretical_FC.columns:
        result[key] = {"R1": [], "R2": [], "S": [], "K": []}

for key in theoretical_FC.columns:
    print(key)
    couple_celltypes = (int(key[0]) - 1, int(key[1]) - 1)
    print("\nDifferential Expression A/B for cell types\nA: %s\nB: %s\n" %
          tuple((cell_types[couple_celltypes[i]] for i in [0, 1])))

    # parameters
    n_cells = 30
    n_samples = 100
    use_IS = False

    # Reference data
    log_FC = theoretical_FC[key]

    # for rep in range(20):

    # DE with scVI
    # labels with scVI should come from the kNN, but if you want the real labels, uncomment this line

    # cell A & batch 0 VS cell B & batch 0
    # print(rep)
    set_a = np.where(
        np.logical_and(scVI_labels == couple_celltypes[0], gene_dataset.batch_indices.ravel() == 0))[0]
    set_b = np.where(
        np.logical_and(scVI_labels == couple_celltypes[1], gene_dataset.batch_indices.ravel() == 0))[0]
    print(len(set_a),len(set_b))
    bayes_A = get_bayes_factor_scvi(set_a, set_b, n_samples, n_cells, use_is=use_IS)
    # roc_auc_1, roc_auc_2, spearman, kendall = eval_bayes_factor(log_FC, bayes_A)
    # results_DE_scVI_A[key]["R1"].append(roc_auc_1)
    # results_DE_scVI_A[key]["R2"].append(roc_auc_2)
    # results_DE_scVI_A[key]["S"].append(spearman)
    # results_DE_scVI_A[key]["K"].append(kendall)

    # cell A & batch 1 VS cell B & batch 1
    set_a = np.where(
        np.logical_and(scVI_labels == couple_celltypes[0], gene_dataset.batch_indices.ravel() == 1))[0]
    set_b = np.where(
        np.logical_and(scVI_labels == couple_celltypes[1], gene_dataset.batch_indices.ravel() == 1))[0]
    bayes_B = get_bayes_factor_scvi(set_a, set_b, n_samples, n_cells, use_is=use_IS)
    # roc_auc_1, roc_auc_2, spearman, kendall = eval_bayes_factor(log_FC, bayes_B)
    # results_DE_scVI_B[key]["R1"].append(roc_auc_1)
    # results_DE_scVI_B[key]["R2"].append(roc_auc_2)
    # results_DE_scVI_B[key]["S"].append(spearman)
    # results_DE_scVI_B[key]["K"].append(kendall)

    # all cell A FORCE batch 0 VS all cell B FORCE batch 0
    set_a = np.where(scVI_labels == couple_celltypes[0])[0]
    set_b = np.where(scVI_labels == couple_celltypes[1])[0]
    bayes_AB1 = get_bayes_factor_scvi(set_a, set_b, n_samples, n_cells, use_is=use_IS, force_batch=0)
    # all cell A FORCE batch 1 VS all cell B FORCE batch 1
    set_a = np.where(scVI_labels == couple_celltypes[0])[0]
    set_b = np.where(scVI_labels == couple_celltypes[1])[0]
    bayes_AB2 = get_bayes_factor_scvi(set_a, set_b, n_samples, n_cells, use_is=use_IS, force_batch=1)
    # Merge BFs
    bayes_AB = 0.5 * bayes_AB1 + 0.5 * bayes_AB2
    # roc_auc_1, roc_auc_2, spearman, kendall = eval_bayes_factor(log_FC, bayes_AB)
    # results_DE_scVI_AB[key]["R1"].append(roc_auc_1)
    # results_DE_scVI_AB[key]["R2"].append(roc_auc_2)
    # results_DE_scVI_AB[key]["S"].append(spearman)
    # results_DE_scVI_AB[key]["K"].append(kendall)

    #
    # ##########################################################
    # # nevf = np.asarray(nevf['x'])
    # print(eval_bayes_factor(log_FC[nevf==0],bayes_AB[nevf==0]))
    # print(eval_bayes_factor(log_FC[nevf==1],bayes_AB[nevf==1]))
    # print(eval_bayes_factor(log_FC[nevf==2],bayes_AB[nevf==2]))
    # print(eval_bayes_factor(log_FC[nevf==3],bayes_AB[nevf==3]))
    # print(eval_bayes_factor(log_FC[nevf==4],bayes_AB[nevf==4]))
    # ##########################################################
    # DE with true labels
    # set_a = np.where(
    #     np.logical_and(true_labels == couple_celltypes[0], gene_dataset.batch_indices.ravel() == 0))[0]
    # set_b = np.where(
    #     np.logical_and(true_labels == couple_celltypes[1], gene_dataset.batch_indices.ravel() == 0))[0]
    # print(len(set_a),len(set_b))
    # bayes_A = get_bayes_factor_scvi(set_a, set_b, n_samples, n_cells, use_is=use_IS)
    # roc_auc_1, roc_auc_2, spearman, kendall = eval_bayes_factor(log_FC, bayes_A)
    # results_DE_true_A[key]["R1"].append(roc_auc_1)
    # results_DE_true_A[key]["R2"].append(roc_auc_2)
    # results_DE_true_A[key]["S"].append(spearman)
    # results_DE_true_A[key]["K"].append(kendall)

    # cell A & batch 1 VS cell B & batch 1
    # set_a = np.where(
    #     np.logical_and(true_labels == couple_celltypes[0], gene_dataset.batch_indices.ravel() == 1))[0]
    # set_b = np.where(
    #     np.logical_and(true_labels == couple_celltypes[1], gene_dataset.batch_indices.ravel() == 1))[0]
    # bayes_B = get_bayes_factor_scvi(set_a, set_b, n_samples, n_cells, use_is=use_IS)
    # roc_auc_1, roc_auc_2, spearman, kendall = eval_bayes_factor(log_FC, bayes_B)
    # results_DE_true_B[key]["R1"].append(roc_auc_1)
    # results_DE_true_B[key]["R2"].append(roc_auc_2)
    # results_DE_true_B[key]["S"].append(spearman)
    # results_DE_true_B[key]["K"].append(kendall)

    # all cell A FORCE batch 0 VS all cell B FORCE batch 0
    # set_a = np.where(true_labels == couple_celltypes[0])[0]
    # set_b = np.where(true_labels == couple_celltypes[1])[0]
    # bayes_AB1 = get_bayes_factor_scvi(set_a, set_b, n_samples, n_cells, use_is=use_IS, force_batch=0)
    # # all cell A FORCE batch 1 VS all cell B FORCE batch 1
    # set_a = np.where(true_labels == couple_celltypes[0])[0]
    # set_b = np.where(true_labels == couple_celltypes[1])[0]
    # bayes_AB2 = get_bayes_factor_scvi(set_a, set_b, n_samples, n_cells, use_is=use_IS, force_batch=1)
    # Merge BFs
    # bayes_AB = 0.5 * bayes_AB1 + 0.5 * bayes_AB2
    # roc_auc_1, roc_auc_2, spearman, kendall = eval_bayes_factor(log_FC, bayes_AB)
    # results_DE_true_AB[key]["R1"].append(roc_auc_1)
    # results_DE_true_AB[key]["R2"].append(roc_auc_2)
    # results_DE_true_AB[key]["S"].append(spearman)
    # results_DE_true_AB[key]["K"].append(kendall)

    # DE with scANVI

    # using approximate posterior, can stick to prior instead
    n_cells = 30
    n_samples = 100
    use_agg_post = True

    # n_cells = 0
    # n_samples = 3000
    # use_agg_post = False


    def scanvi_generate_scale(trainer_info, labels_info, agg_post, cell_type, batch, ncells, nsamples):
        if agg_post:
            # DE from aggregate posterior
            cell_idx_a = np.random.choice(np.where(labels_info == cell_type[0])[0], ncells)
            local_post_a = trainer_info.create_posterior(trainer_info.model,
                                                         trainer_info.gene_dataset, indices=cell_idx_a)
            px_scale_a = local_post_a.get_regenerate_scale(batch, cell_type[0], nsamples)
            cell_idx_b = np.random.choice(np.where(labels_info == cell_type[1])[0], ncells)
            local_post_b = trainer_info.create_posterior(trainer_info.model,
                                                         trainer_info.gene_dataset, indices=cell_idx_b)
            px_scale_b = local_post_b.get_regenerate_scale(batch, cell_type[1], nsamples)
        else:
            # DE from prior
            px_scale_a = np.array(trainer_info.model.generate_latent_samples(cell_type[0],
                                                                             batch, nsamples).cpu())
            px_scale_a = px_scale_a[np.newaxis, :]
            px_scale_b = np.array(trainer_info.model.generate_latent_samples(cell_type[1],
                                                                             batch, nsamples).cpu())
            px_scale_b = px_scale_b[np.newaxis, :]
        return px_scale_a, px_scale_b


    px_scale_a0, px_scale_b0 = scanvi_generate_scale(trainer_scanvi, scanvi_labels,
                                                     use_agg_post, couple_celltypes, 0, n_cells, n_samples)
    labels_de = np.concatenate((np.zeros((px_scale_a0.shape[1],)), np.ones((px_scale_b0.shape[1],))))
    px_scale_0 = np.concatenate((px_scale_a0, px_scale_b0), axis=1)
    bayes_scanviAB1 = get_IS_bayes_factors(px_scale_0, None, labels_de, 0,
                                           other_cell_idx=1,
                                           importance_sampling=False, permutation=False)

    px_scale_a1, px_scale_b1 = scanvi_generate_scale(trainer_scanvi, scanvi_labels,
                                                     use_agg_post, couple_celltypes, 1, n_cells, n_samples)
    labels_de = np.concatenate((np.zeros((px_scale_a0.shape[1],)), np.ones((px_scale_b0.shape[1],))))
    px_scale_1 = np.concatenate((px_scale_a1, px_scale_b1), axis=1)
    bayes_scanviAB2 = get_IS_bayes_factors(px_scale_1, None, labels_de, 0,
                                           other_cell_idx=1,
                                           importance_sampling=False, permutation=False)
    # Merge BFs
    bayes_scanviAB = 0.5 * bayes_scanviAB1 + 0.5 * bayes_scanviAB2
    # roc_auc_1, roc_auc_2, spearman, kendall = eval_bayes_factor(log_FC, bayes_scanviAB)
    # results_DE_scANVI[key]["R1"].append(roc_auc_1)
    # results_DE_scANVI[key]["R2"].append(roc_auc_2)
    # results_DE_scANVI[key]["S"].append(spearman)
    # results_DE_scANVI[key]["K"].append(kendall)
    res = pd.DataFrame([bayes_A,bayes_B,bayes_AB,bayes_scanviAB], index=['bayes_A','bayes_B','bayes_AB','bayes_scanviAB'])
    res.T.to_csv(save_path + "result.nb.%s.%i.csv"%(key, rep))

    # pickle.dump(results_DE_scVI_A, open(os.path.join(save_path, "results0.%i.dic"%rep), "wb"))
    # pickle.dump(results_DE_scVI_B, open(os.path.join(save_path, "results1.%i.dic"%rep), "wb"))
    # pickle.dump(results_DE_scVI_AB, open(os.path.join(save_path, "results2.%i.dic"%rep), "wb"))
    # pickle.dump(results_DE_scANVI, open(os.path.join(save_path, "results3.%i.dic"%rep), "wb"))
    # pickle.dump(results_DE_true_A, open(os.path.join(save_path, "results0true.%i.dic"%rep), "wb"))
    # pickle.dump(results_DE_true_B, open(os.path.join(save_path, "results1true.%i.dic"%rep), "wb"))
    # pickle.dump(results_DE_true_AB, open(os.path.join(save_path, "results2true.%i.dic"%rep), "wb"))
