import pickle
import os
import numpy as np
save_path = "DE_info"


results_DE_scVI_A = pickle.load(open(os.path.join(save_path, "results0.dic"), "rb"))
results_DE_scVI_B = pickle.load(open(os.path.join(save_path, "results1.dic"), "rb"))
results_DE_scVI_AB = pickle.load(open(os.path.join(save_path, "results2.dic"), "rb"))
results_DE_scANVI = pickle.load(open(os.path.join(save_path, "results3.dic"), "rb"))

for k in results_DE_scVI_A.keys():
    for res_type in results_DE_scVI_A[k].keys():
        print(k, res_type)
        print("results_DE_scVI_A", np.mean(results_DE_scVI_A[k][res_type]),
              "+-", np.std(results_DE_scVI_A[k][res_type]))
        print("results_DE_scVI_B", np.mean(results_DE_scVI_B[k][res_type]),
              "+-", np.std(results_DE_scVI_B[k][res_type]))
        print("results_DE_scVI_AB", np.mean(results_DE_scVI_AB[k][res_type]),
              "+-", np.std(results_DE_scVI_AB[k][res_type]))
        print("results_DE_scANVI", np.mean(results_DE_scANVI[k][res_type]),
              "+-", np.std(results_DE_scANVI[k][res_type]))
