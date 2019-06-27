import rpy2
import pandas
import numpy as np
import rpy2.robjects.numpy2ri
import warnings

from rpy2.robjects import r
from scvi.inference.posterior import get_IS_bayes_factors
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr
from sklearn.neighbors import KNeighborsClassifier
from scipy.stats import kendalltau
from collections import OrderedDict
from rpy2.rinterface import RRuntimeWarning

rpy2.robjects.numpy2ri.activate()
py2ri_orig = rpy2.robjects.conversion.py2ri
warnings.filterwarnings("ignore", category=RRuntimeWarning)

r["library"]("scmap")
r["library"]("SingleCellExperiment")
r["library"]("matrixStats")
r["library"]("Matrix")
r["library"]("RcppCNPy")
r["library"]("reticulate")
r["library"]("edgeR")


def conversion_pydataframe(obj):
    """
    Convert pandas DataFrame or python object to an R dataframe/object.
    """
    if isinstance(obj, pandas.DataFrame):
        od = OrderedDict()
        for name, values in obj.iteritems():
            if values.dtype.kind == 'O':
                od[name] = rpy2.robjects.vectors.StrVector(values)
            else:
                od[name] = rpy2.robjects.conversion.py2ri(values)
        return rpy2.robjects.vectors.DataFrame(od)
    else:
        return py2ri_orig(obj)


def run_edgeR(gene_expression, bio_assignment, gene_names, batch_info=None, batch=True):
    if batch_info is None:
        batch = False
    r_counts = conversion_pydataframe(gene_expression)
    r_bio_group = conversion_pydataframe(bio_assignment)
    r_dge = r.DGEList(counts=r.t(r_counts), genes=gene_names)
    r.assign("dge", r_dge)
    r.assign("bio_group", r.factor(r_bio_group))
    r("dge$samples$bio_group <- bio_group")

    if batch:
        r_batch_group = conversion_pydataframe(batch_info)
        r.assign("batch_group", r.factor(r_batch_group))
        r("dge$samples$batch_group <- batch_group")

    r("""dge <- suppressWarnings(edgeR::calcNormFactors(dge))""")

    if not batch:
        r("""design <- model.matrix(~bio_group, data = dge$samples)""")
        r("""colnames(design) <- c("Intercept", "bio")""")

    if batch:
        r("""design <- model.matrix(~bio_group+batch_group, data = dge$samples)""")
        r("""colnames(design) <- c("Intercept", "bio", "batch")""")

    r("""dge <- estimateDisp(dge, design)""")

    r("""fit <- glmFit(dge, design)""")
    if not batch:
        r("""lrt <- glmLRT(fit)""")
    if batch:
        r("""lrt <- glmLRT(fit, coef="bio")""")
    return r("lrt$table$PValue")


def transfer_nn_labels(latent_array, labels_array, batch_indices_array):
    # Transfer labels from batch 0 to batch 1 using scVI
    latent_labelled = latent_array[batch_indices_array.ravel() == 0, :]
    labels_labelled = labels_array[batch_indices_array.ravel() == 0]
    neigh = KNeighborsClassifier(n_neighbors=10)
    neigh = neigh.fit(latent_labelled, labels_labelled)
    return neigh.predict(latent_array)


def get_bayes_factor_scvi(cset_a, cset_b, sampling_n, cells_sampled, use_is, trainer, gene_dataset, force_batch=None):
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
