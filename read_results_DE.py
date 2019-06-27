import pickle
import os
import numpy as np
save_path = "../symsim_scVI/symsim_result/DE/"


results_DE_scVI_A_all = pickle.load(open(os.path.join(save_path, "results0.dic"), "rb"))
results_DE_scVI_B_all = pickle.load(open(os.path.join(save_path, "results1.dic"), "rb"))
results_DE_scVI_AB_all = pickle.load(open(os.path.join(save_path, "results2.dic"), "rb"))
results_DE_scANVI_all = pickle.load(open(os.path.join(save_path, "results3.dic"), "rb"))

for rep in range(50):
    results_DE_scVI_A = pickle.load(open(os.path.join(save_path, "results0.%i.dic"%rep), "rb"))
    results_DE_scVI_B = pickle.load(open(os.path.join(save_path, "results1.%i.dic"%rep), "rb"))
    results_DE_scVI_AB = pickle.load(open(os.path.join(save_path, "results2.%i.dic"%rep), "rb"))
    results_DE_scANVI = pickle.load(open(os.path.join(save_path, "results3.%i.dic"%rep), "rb"))
    for x in results_DE_scVI_A.keys():
        for y in results_DE_scVI_A[x].keys():
            results_DE_scVI_A_all[x][y].append(results_DE_scVI_A[x][y][0])
            results_DE_scVI_B_all[x][y].append(results_DE_scVI_B[x][y][0])
            results_DE_scVI_AB_all[x][y].append(results_DE_scVI_AB[x][y][0])
            results_DE_scANVI_all[x][y].append(results_DE_scANVI[x][y][0])

for k in results_DE_scVI_A.keys():
    for res_type in results_DE_scVI_A[k].keys():
        print(k, res_type)
        print("results_DE_scVI_A", np.mean(results_DE_scVI_A_all[k][res_type]),
              "+-", np.std(results_DE_scVI_A_all[k][res_type]))
        print("results_DE_scVI_B", np.mean(results_DE_scVI_B_all[k][res_type]),
              "+-", np.std(results_DE_scVI_B_all[k][res_type]))
        print("results_DE_scVI_AB", np.mean(results_DE_scVI_AB_all[k][res_type]),
              "+-", np.std(results_DE_scVI_AB_all[k][res_type]))
        print("results_DE_scANVI", np.mean(results_DE_scANVI_all[k][res_type]),
              "+-", np.std(results_DE_scANVI_all[k][res_type]))



scVI_res = {'scVI_A':{},'scVI_B':{},'scVI_AB':{},'scANVI':{}}

for i, type in enumerate(['scVI_A','scVI_B','scVI_AB','scANVI']):
    X = pickle.load(open(os.path.join(save_path, "results%i.dic"%i), "rb"))
    scVI_res[type] = X
    for rep in range(30):
        X = pickle.load(open(os.path.join(save_path, "results%i.%i.dic" % (i,rep)), "rb"))
        for x in X.keys():
            for y in X[x].keys():
                scVI_res[type][x][y].append(X[x][y][0])



for type in scVI_res.keys():
    for comparison in scVI_res[type].keys():
        for res_type in scVI_res[type][comparison].keys():
            print("results_scVI %s %s %s"%(type, comparison, res_type), np.mean(scVI_res[type][comparison][res_type]),
                  "+-", np.std(scVI_res[type][comparison][res_type]))

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

edgeR_res={'.scVI':{},'.scANVI':{},'':{}}

for type in ['.scVI','.scANVI','']:
    edgeR_res[type] = {'1_2':{},'2_3':{},'2_4':{},'4_5':{}}
    for comparison in ['1_2','2_3','2_4','4_5']:
        edgeR_res[type][comparison] = {'A': {}, 'B': {}, 'AB': {}}
        for batch in ['A', 'B', 'AB']:
            edgeR_res[type][comparison][batch] = {'R1':[],'R2':[],'S':[],'K':[]}
            for rep in range(50):
                X = pd.read_csv(os.path.join(save_path, 'EdgeR/%s%s.%i.%s.edgeR.csv') % (batch, type, (rep+1), comparison))
                stats = eval_de(theoretical_FC[comparison.replace('_', '')], X['logFC'])
                for i,x in enumerate(['R1','R2','S','K']):
                    edgeR_res[type][comparison][batch][x].append(stats[i])

for type in edgeR_res.keys():
    for comparison in edgeR_res[type].keys():
        for batch in edgeR_res[type][comparison].keys():
            for res_type in edgeR_res[type][comparison][batch].keys():
                print("results_EdgeR%s %s %s %s"%(type, comparison, batch, res_type), np.mean(edgeR_res[type][comparison][batch][res_type]),
                      "+-", np.std(edgeR_res[type][comparison][batch][res_type]))
