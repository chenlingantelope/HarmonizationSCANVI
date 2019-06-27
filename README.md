# HarmonizationSCANVI
+ Reproducing results in the "Harmonization and Annotation of Single-cell Transcriptomics data with Deep Generative Models" paper
+ Demonstration of how to use scVI and scANVI for the harmonization and annotation problem

# Contact
chenlingantelope [at] berkeley [dot] edu

# Datasets
| **Analysis** | **Associated Script** |**Datasets** | **Technology** |**Number of Cells**| **Ref.**| 
|---|---|---|---|---|---|
|**Figure 2:** Benchmark| PBMC8KCITE.py|PBMC-8K; PBMC-CITE | 10x | 8,381; 7,667 | [10x Datasets](https://support.10xgenomics.com/single-cell-gene-expression/datasets)[Stoeckius, Marlon, et al. 2017](https://www.nature.com/nmeth/journal/v14/n9/abs/nmeth.4380.html&hl=en&sa=T&oi=gsb&ct=res&cd=0&d=543871865635877584&ei=K6QuXOCKAsf1ygSZ6rDgCw&scisig=AAGBfm3w0rGPz4b6GzQIHEme1DakHFBkrg)| 
|**Supplementary Figure 2:** UMAP Visualization| PBMC8KCITE.py|PBMC-8K; PBMC-CITE | 10x | 8,381; 7,667 | [10x Datasets](https://support.10xgenomics.com/single-cell-gene-expression/datasets); [Stoeckius, Marlon, et al. 2017](https://www.nature.com/nmeth/journal/v14/n9/abs/nmeth.4380.html&hl=en&sa=T&oi=gsb&ct=res&cd=0&d=543871865635877584&ei=K6QuXOCKAsf1ygSZ6rDgCw&scisig=AAGBfm3w0rGPz4b6GzQIHEme1DakHFBkrg)| 
|**Figure 2:** Benchmark| MarrowTM.py Tech1.pretty.ipynb| MarrowTM-10x; MarrowTM-ss2| 10x; SmartSeq2|4,112;5,351|[Quake, Stephen R., et al. 2018](https://www.biorxiv.org/content/early/2018/03/29/237446.abstract&hl=en&sa=T&oi=gsb&ct=res&cd=0&d=16805566974012009566&ei=RqkuXIuCHoX3ygSgmq_YCw&scisig=AAGBfm10AxWIaxsswNFWHxYXrFtCFq9TYw)|
|**Supplementary Figure 1:** Robustness Analysis for Hyperparameter Choice| Robustness_study.ipynb|  MarrowTM-10x; MarrowTM-ss2| 10x; SmartSeq2|4,112;5,351|[Quake, Stephen R., et al. 2018](https://www.biorxiv.org/content/early/2018/03/29/237446.abstract&hl=en&sa=T&oi=gsb&ct=res&cd=0&d=16805566974012009566&ei=RqkuXIuCHoX3ygSgmq_YCw&scisig=AAGBfm10AxWIaxsswNFWHxYXrFtCFq9TYw)|
|**Supplementary Figure 3:** UMAP Visualization| MarrowTM.py|  MarrowTM-10x; MarrowTM-ss2| 10x; SmartSeq2|4,112;5,351| |[Quake, Stephen R., et al. 2018](https://www.biorxiv.org/content/early/2018/03/29/237446.abstract&hl=en&sa=T&oi=gsb&ct=res&cd=0&d=16805566974012009566&ei=RqkuXIuCHoX3ygSgmq_YCw&scisig=AAGBfm10AxWIaxsswNFWHxYXrFtCFq9TYw)|
|**Figure 2:** Benchmark| Pancreas.py|Pancreas-InDrop; Pancreas-CEL-Seq2| inDrop; CEL-Seq2 | 8,569; 2,449 | [Baron, Maayan, et al. 2016](https://www.sciencedirect.com/science/article/pii/S2405471216302666&hl=en&sa=T&oi=gsb-ggp&ct=res&cd=0&d=5387698199932191078&ei=mqkuXLToE9D0yATyx5bACA&scisig=AAGBfm0u3B4kGoWVsQGzS2QBcbCuqXyWYQ); [Muraro, Mauro J., et al. 2016](https://www.sciencedirect.com/science/article/pii/S2405471216302927&hl=en&sa=T&oi=gsb-ggp&ct=res&cd=0&d=10796381433466141687&ei=3akuXPf4HIX3ygSgmq_YCw&scisig=AAGBfm3frJpLByoBIRELJuZOcxz9c0ghyQ)| 
|**Supplementary Figure 4:** UMAP Visualization|Pancreas.py|Pancreas-InDrop; Pancreas-CEL-Seq2| inDrop; CEL-Seq2 | 8,569; 2,449 | [Baron, Maayan, et al. 2016](https://www.sciencedirect.com/science/article/pii/S2405471216302666&hl=en&sa=T&oi=gsb-ggp&ct=res&cd=0&d=5387698199932191078&ei=mqkuXLToE9D0yATyx5bACA&scisig=AAGBfm0u3B4kGoWVsQGzS2QBcbCuqXyWYQ); [Muraro, Mauro J., et al. 2016](https://www.sciencedirect.com/science/article/pii/S2405471216302927&hl=en&sa=T&oi=gsb-ggp&ct=res&cd=0&d=10796381433466141687&ei=3akuXPf4HIX3ygSgmq_YCw&scisig=AAGBfm3frJpLByoBIRELJuZOcxz9c0ghyQ)| 
|**Figure 2:** Benchmark| DentateGyrus.py| DentateGyrus-10x; DentateGyrus-C1| 10x; Fluidigm C1 | 5,454; 2,303 | [Hochgerner, Hannah, et al. 2018](https://www.nature.com/articles/s41593-017-00562&hl=en&sa=T&oi=gsb&ct=res&cd=0&d=1942730453832570486&ei=IKouXNjDAdD0yATyx5bACA&scisig=AAGBfm2EmFzadvKXZZBYgGrmEf6FX-RgqQ)| 
|**Supplementary Figure 5:** UMAP Visualization| DentateGyrus.py| DentateGyrus-10x; DentateGyrus-C1| 10x; Fluidigm C1 | 5,454; 2,303 | [Hochgerner, Hannah, et al. 2018](https://www.nature.com/articles/s41593-017-00562&hl=en&sa=T&oi=gsb&ct=res&cd=0&d=1942730453832570486&ei=IKouXNjDAdD0yATyx5bACA&scisig=AAGBfm2EmFzadvKXZZBYgGrmEf6FX-RgqQ)| 
|**Figure 3:** Robustness Analysis by subsampling cells **Supplementary Figure 10**| NoOverlapSCANVI.py PopRemoveSCANVI.py SCANVI_posterior-NoOverlap.ipynb SCANVI_posterior_poprm.ipynb|PBMC-8K; PBMC-CITE | 10x | 8,381; 7,667 |[10x Datasets](https://support.10xgenomics.com/single-cell-gene-expression/datasets); [Stoeckius, Marlon, et al. 2017](https://www.nature.com/nmeth/journal/v14/n9/abs/nmeth.4380.html&hl=en&sa=T&oi=gsb&ct=res&cd=0&d=543871865635877584&ei=K6QuXOCKAsf1ygSZ6rDgCw&scisig=AAGBfm3w0rGPz4b6GzQIHEme1DakHFBkrg)| 
|**Figure 4:** Continuous Trajectory Supplementary **Supplementary Figure 6:** UMAP| continuous.ipynb|HEMATO-Tusi; HEMATO-Paul|inDrop;  MARS-seq| 4,016 ; 2,730 | [Tusi, Betsabeh Khoramian, et al. 2018](https://www.nature.com/articles/nature25741&hl=en&sa=T&oi=gsb&ct=res&cd=0&d=14150194670473472645&ei=i6ouXJzEFo7UygTf-ZaYDw&scisig=AAGBfm1gW5IiaTVVH5hx3o9j4lc8BA8-gQ); [Paul, Franziska, et al. 2015](https://www.sciencedirect.com/science/article/pii/S0092867415014932&hl=en&sa=T&oi=gsb-ggp&ct=res&cd=0&d=10576476503186237297&ei=xaouXLWKKMa-ygSzm7mYAg&scisig=AAGBfm35-xWPzpNRxS6BZY34YZ1M2aRGdw)| 10x| 68,579 ;94,655 | 
|**Figure 5:** External Validation by Experimentally Derived Labels, **Supplementary Figure 11**| harmonization-CitePure-SCANVI.ipynb| PBMC-68K; PBMC-Sorted; PBMC-CITE|10x|  68,579; 94,655; 7,667| [Zheng, Grace XY, et al. 2017](https://www.nature.com/articles/ncomms14049&hl=en&sa=T&oi=gsb-ggp&ct=res&cd=0&d=17926869542004746646&ei=maQuXPLtGI7UygTf-ZaYDw&scisig=AAGBfm2osPWuIr9SHJfW08Ib5ZQkOa7BvQ); [Stoeckius, Marlon, et al. 2017](https://www.nature.com/nmeth/journal/v14/n9/abs/nmeth.4380.html&hl=en&sa=T&oi=gsb&ct=res&cd=0&d=543871865635877584&ei=K6QuXOCKAsf1ygSZ6rDgCw&scisig=AAGBfm3w0rGPz4b6GzQIHEme1DakHFBkrg)|
|**Figure 6:** Semi-Supervised Annotation of T Cell Subtypes, **Supplementary Figure 12**| SCANVI-mild-annot-Clustering.ipynb| PBMC-Sorted T cell Subtypes|10x|42919| [Zheng, Grace XY, et al. 2017](https://www.nature.com/articles/ncomms14049&hl=en&sa=T&oi=gsb-ggp&ct=res&cd=0&d=17926869542004746646&ei=maQuXPLtGI7UygTf-ZaYDw&scisig=AAGBfm2osPWuIr9SHJfW08Ib5ZQkOa7BvQ); [Stoeckius, Marlon, et al. 2017](https://www.nature.com/nmeth/journal/v14/n9/abs/nmeth.4380.html&hl=en&sa=T&oi=gsb&ct=res&cd=0&d=543871865635877584&ei=K6QuXOCKAsf1ygSZ6rDgCw&scisig=AAGBfm3w0rGPz4b6GzQIHEme1DakHFBkrg)|
|**Hierarchical Semi-Supervised Annotation** |Hierarchical.ipynb| CORTEX| 10x | 160,796 | [Zeisel, Amit, et al. "Molecular architecture of the mouse nervous system." bioRxiv (2018): 294918.](https://www.biorxiv.org/content/early/2018/04/06/294918.abstract&hl=en&sa=T&oi=gsb&ct=res&cd=0&d=13555086917209182662&ei=YaouXJ3ZKIaaywTa-IK4BA&scisig=AAGBfm0mNKyQodU4OZm291abJvhqXPhHuw)|
|**Supplementary Figure 7**: Scalability Analysis |scanorama.ipynb| SCANORAMA| Mixed| 105,476 | [Hie, Brian L., Bryan Bryson, and Bonnie Berger. "Panoramic stitching of heterogeneous single-cell transcriptomic data." bioRxiv (2018): 371179.](https://www.biorxiv.org/content/early/2018/07/17/371179.abstract&hl=en&sa=T&oi=gsb&ct=res&cd=0&d=4117396372295068293&ei=7KouXOWRJMfwyATEm66wAg&scisig=AAGBfm2g_e9z-pekxTL43Q88G6-_0OHYzw)
|**Supplementary Figure 13**: Differential Expression |DE-final.ipynb| PBMC-8K; PBMC-68K| 10x| 8,381; 68,579 | [10x Datasets](https://support.10xgenomics.com/single-cell-gene-expression/datasets); [Zheng, Grace XY, et al. 2017](https://www.nature.com/articles/ncomms14049&hl=en&sa=T&oi=gsb-ggp&ct=res&cd=0&d=17926869542004746646&ei=maQuXPLtGI7UygTf-ZaYDw&scisig=AAGBfm2osPWuIr9SHJfW08Ib5ZQkOa7BvQ) 

+ **Supplemtary Figure 2,3,4,5,8,9** are generated using scripts in **Additional_Scripts/** using output from the analysis python scripts including **scanvi_acc.R**, **KNNcurves.py** and **BE_curves.py**. 
+ Boxplots for **Figure 3** are generated using **poprm_boxplot.R** in **Additional_Scripts/**
+ The Additional_Scripts also contains code for running Seurat directly from commandline **runSeurat.R** and **SeuratPCA.R**. 
+ All .gmt files in **Additional_Scripts/** are gene signatures. 

# Installation
+ Clone the github repository, install the dependencies and call functions from the modules scVI
+ Install time (< 10 min)


# Requirements
+ Pytorch V0.4.1
+ Python 3
+ scikit-learn V0.19.1

# Instructions
+ To reproduce results from the paper, look up the relevant datasets, python notebooks (located in **notebooks/**), or python scripts (located in the root directory). 
+ Download the relevant datasets except for the ones already wrapped for the scVI package (PBMC-8K, PBMC-CITE, PBMC-68K, PBMC-Sorted, MarrowTM-10x, MarrowTM-ss2 can be loaded directly with the dataloader functions)
+ Annotation files generated by us when the original study did not provide annotation (cite.seurat.labels) can be found in the [scvi-data](https://github.com/YosefLab/scVI-data/blob/master/cite.seurat.labels) repository 
+ Run the analysis and results should match those of the paper. 
+ This repository contains functions written uniquely to produce some of the analysis in this paper. For more up-to-date package refer to main [scVI](https://github.com/YosefLab/scVI) repository 

