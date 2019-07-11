import pandas as pd
import numpy as np
import sys
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


dataname = str(sys.argv[1])
# scvi = pd.read_csv('../' + dataname + '/scvi_nb.res.txt',delimiter=' ')
# scvi['model_type'] = [x + 'NB' for x in scvi['model_type'] ]
# others = pd.read_csv('../' + dataname + '/scvi.res.txt',delimiter=' ')
scvi = pd.read_csv('../' + dataname + '/scvi.res.txt',delimiter=' ')
others = pd.read_csv('../' + dataname + '/others.res.txt',delimiter=' ')


stats = pd.concat([scvi,others])
model_types = stats['model_type']
stat_names = np.asarray(list(scvi.columns)[1:])
model_types = np.asarray(model_types)

res=[]
for x in np.unique(model_types):
    stat = np.mean(np.asarray(stats[model_types==x])[:, 1:], axis=0)
    res.append(stat)

model_types = np.unique(model_types)
res = np.asarray(res)

sorted_res=[]
methods = ['vae', 'scanvi1', 'scanvi2', 'vae_nb', 'scanvi1_nb', 'scanvi2_nb']
model_names = ['scVI', 'SCANVI1', 'SCANVI2', 'scVI_NB', 'SCANVI1_NB', 'SCANVI2_NB']
colors = ('r', 'g', 'g--', 'r:', 'g:', 'g-.', 'b', 'y', 'y--', 'b:', 'y:', 'y-.')

# methods = ['vae', 'scanvi1', 'scanvi2','readSeurat', 'MNN', 'Combat', 'PCA']
# model_names = ['scVI', 'SCANVI1', 'SCANVI2',  'CCA', 'MNN', 'Combat', 'PCA']
# colors = ('r', 'g', 'g--', 'b', 'y', 'm', 'c')


for x in methods:
    sorted_res.append(res[model_types==x,:])

sorted_res = np.asarray(sorted_res)
sorted_res = sorted_res.squeeze()

filtered_res = [np.asarray(sorted_res[:, i]) for i,x in enumerate(stat_names) if 'res_jaccard' in x]
filtered_res = np.asarray(filtered_res)

import matplotlib
KNeighbors = np.concatenate([np.arange(10, 100, 10), np.arange(100, 500, 50)])
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt

plt.figure(figsize=(5, 5))

for i,x in enumerate(model_names):
    plt.plot(KNeighbors, filtered_res[:,i],colors[i],label=x)

legend = plt.legend(loc='lower right', shadow=False)
plt.savefig("../%s/%s_compare4_KNN.pdf" % (dataname,dataname))
# plt.savefig("../%s/%s.KNN.pdf" % (dataname,dataname))
#
# plt.figure(figsize=(5, 5))
# colors = ('r','g','b','y','m','c')
# for i in [0,2,3]:
#     x = model_names[i]
#     plt.plot(KNeighbors, filtered_res[:,i],colors[i],label=x)
#
# legend = plt.legend(loc='lower right', shadow=False)
# plt.savefig("../%s/%s_compare3_KNN.pdf" % (dataname,dataname))
#
