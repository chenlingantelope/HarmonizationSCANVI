import pandas as pd
import numpy as np
import pandas as pd
import os

save_path = "/home/romain/data_DE"

batch = pd.read_csv(os.path.join(save_path, "batchid.csv"),
                    sep=",", index_col=0)
label = pd.read_csv(os.path.join(save_path, "DE.cell_meta.csv"),
                    sep=",", index_col=0)
count_matrix = pd.read_csv(os.path.join(save_path, "DE.obsv.csv"),
                           sep=",", index_col=0).T
count_matrix.index = label.index

count_matrix = count_matrix[batch.x != 3].T
label = label[batch.x != 3]
batch = batch[batch.x != 3]

batch.to_csv(os.path.join(save_path, "filtered_batchid.csv"))
count_matrix.to_csv(os.path.join(save_path, "filtered_DE.obsv.csv"))
label.to_csv(os.path.join(save_path, "filtered_DE.cell_meta.csv"))
