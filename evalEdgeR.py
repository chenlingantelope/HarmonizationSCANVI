from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr
from scipy.stats import kendalltau
import os
import numpy as np
import pandas as pd

save_path = "../symsim_scVI/symsim_result/DE/"
os.chdir('/data/yosef2/users/chenling/HarmonizationSCANVI')

def eval_de(log_fold_change, logFC):
    """
    :param log_fold_change: groundtruth
    :param bayes_f: non-log Bayes Factor
    :return:
    """
    auc_1 = roc_auc_score(np.abs(log_fold_change) >= 0.6, np.abs(logFC))
    auc_2 = roc_auc_score(np.abs(log_fold_change) >= 0.8, np.abs(logFC))
    spear = spearmanr(logFC, log_fold_change)[0]
    kend = kendalltau(logFC, log_fold_change)[0]
    return auc_1, auc_2, np.abs(spear), np.abs(kend)


theoretical_FC = pd.read_csv(os.path.join(save_path, "theoreticalFC.csv"),
                             sep=",", index_col=0, header=0)


for type in ['.scVI','.scANVI','']:
    for comparison in ['1_2','2_3','2_4','4_5']:
        edgeR_res = pd.read_csv(os.path.join(save_path, 'EdgeR/AB%s.%s.edgeR.csv')%(type,comparison))
        eval_de(theoretical_FC[comparison.replace('_','')], edgeR_res['logFC'])

type = '.scVI'
comparison ='1_2'
rep=1
edgeR_res = pd.read_csv(os.path.join(save_path, 'EdgeR/AB%s.%i.%s.edgeR.csv') % (type,rep, comparison))
eval_de(theoretical_FC[comparison.replace('_','')], edgeR_res['logFC'])