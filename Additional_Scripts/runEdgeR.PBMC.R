# Title     : EdgeR
# Objective : run EdgeR on the simulated data
# Created by: chenlingantelope
# Created on: 5/8/19

library(edgeR)
library(Matrix)
setwd('/data/yosef2/users/chenling/symsim_scVI/symsim_result/DE')
count = readMM('PBMC.count.mtx')
count = t(count)
batchid = read.csv("PBMC.batchid.csv")[,1]
batchid = batchid+1
genenames = read.csv('PBMC.genenames.txt',as.is=T)[,1]
# CD4 3364
# CD8 948
# Dendritic 209
# B 1060

PairwiseDE <- function(count, celllabels, pop1, pop2, batchid, name, ncells=30){
    subset1 = c(1:length(celllabels))[celllabels==pop1]
	subset2 = c(1:length(celllabels))[celllabels==pop2]
    subset = c(sample(subset1, ncells, replace=T),sample(subset2, ncells, replace=T))
    count = count[,subset]
	batchid = batchid[subset]
    group = as.factor(celllabels[subset])
    y <- DGEList(counts=count, group=group)
    # cdr <- scale(colMeans(count > 0))
    if( length(unique(batchid))>1 ){
        design = model.matrix(~ group + as.factor(batchid)  )
        colnames(design) <- c("Intercept", "bio", "batch")

    }else{
        design = model.matrix(~group )
        colnames(design) <- c("Intercept", "bio")
    }
    y$genes <- data.frame(Symbol=genenames)
    y <- calcNormFactors(y)
    y <- estimateDisp(y, design)
    fit <- glmFit(y, design)
    lrt <- glmLRT(fit, coef="bio")
    write.csv(lrt$table, file=paste('EdgeR/PBMC.',name,'.edgeR.csv',sep=''))
}

pred_labels = read.csv(paste('PBMC.pred_labels.',rep,'.mis','0.00','.csv',sep=''))
pred_labels = pred_labels+1
temp = table(pred_labels[,2])
CD4 = c(1:10)[temp==3364]
CD8= c(1:10)[temp==948]
DC = c(1:10)[temp==209]
B = c(1:10)[temp==1060]
labels = pred_labels[,2]
labels[batchid==2]=pred_labels[batchid==2,3]

for(rep in c(1:10)){
for (mis in as.character(c('0.00','0.05','0.10','0.15','0.20','0.25','0.30'))){
    print(paste(rep,mis,sep='.'))
    mislabels = labels
    mises = rbinom(prob=as.numeric(mis),size=1,n=length(labels))
    mislabels[mises & (labels==B)]=DC
    mislabels[mises & (labels==DC)]=B

    PairwiseDE(count, mislabels, CD4,CD8,batchid,paste('AB',rep,'CD4CD8',mis,sep='.'))
    PairwiseDE(count, mislabels, B,DC,batchid,paste('AB',rep,'BDC',mis,sep='.'))

    PairwiseDE(count[,batchid==1], mislabels[batchid==1], CD4,CD8,batchid[batchid==1],paste('A',rep,'CD4CD8',mis,sep='.'))
    PairwiseDE(count[,batchid==1], mislabels[batchid==1], B,DC,batchid[batchid==1],paste('A',rep,'BDC',mis,sep='.'))

    PairwiseDE(count[,batchid==2], mislabels[batchid==2], CD4,CD8,batchid[batchid==2],paste('B',rep,'CD4CD8',mis,sep='.'))
    PairwiseDE(count[,batchid==2], mislabels[batchid==2], B,DC,batchid[batchid==2],paste('B',rep,'BDC',mis,sep='.'))

}}

# for(rep in c(1:9)){
# for (mis in c(0.00,0.01,0.05,0.10,0.15,0.20,0.25,0.30)){
#     pred_labels = read.csv(paste('pred_labels.',rep,'.mis',mis,'.csv',sep=''))
#     pred_labels = pred_labels+1
#     print('rep')
#     PairwiseDE(count, pred_labels[,2], 4,5,batchid,paste('AB.scVI',rep,sep='.'))
#     PairwiseDE(count, pred_labels[,2], 2,3,batchid,paste('AB.scVI',rep,sep='.'))
#     PairwiseDE(count, pred_labels[,2], 2,4,batchid,paste('AB.scVI',rep,sep='.'))
#     PairwiseDE(count, pred_labels[,2], 1,2,batchid,paste('AB.scVI',rep,sep='.'))
#
#     PairwiseDE(count[,batchid==1], pred_labels[,2][batchid==1], 4,5,batchid[batchid==1],paste('A.scVI',rep,sep='.'))
#     PairwiseDE(count[,batchid==1], pred_labels[,2][batchid==1], 2,3,batchid[batchid==1],paste('A.scVI',rep,sep='.'))
#     PairwiseDE(count[,batchid==1], pred_labels[,2][batchid==1], 2,4,batchid[batchid==1],paste('A.scVI',rep,sep='.'))
#     PairwiseDE(count[,batchid==1], pred_labels[,2][batchid==1], 1,2,batchid[batchid==1],paste('A.scVI',rep,sep='.'))
#
#
#     PairwiseDE(count[,batchid==2], pred_labels[,2][batchid==2], 4,5,batchid[batchid==2],paste('B.scVI',rep,sep='.'))
#     PairwiseDE(count[,batchid==2], pred_labels[,2][batchid==2], 2,3,batchid[batchid==2],paste('B.scVI',rep,sep='.'))
#     PairwiseDE(count[,batchid==2], pred_labels[,2][batchid==2], 2,4,batchid[batchid==2],paste('B.scVI',rep,sep='.'))
#     PairwiseDE(count[,batchid==2], pred_labels[,2][batchid==2], 1,2,batchid[batchid==2],paste('B.scVI',rep,sep='.'))
#
#
#
#     PairwiseDE(count, pred_labels[,3], 4,5,batchid,paste('AB.scANVI',rep,sep='.'))
#     PairwiseDE(count, pred_labels[,3], 2,3,batchid,paste('AB.scANVI',rep,sep='.'))
#     PairwiseDE(count, pred_labels[,3], 2,4,batchid,paste('AB.scANVI',rep,sep='.'))
#     PairwiseDE(count, pred_labels[,3], 1,2,batchid,paste('AB.scANVI',rep,sep='.'))
#
#     PairwiseDE(count[,batchid==1], pred_labels[,3][batchid==1], 4,5,batchid[batchid==1],paste('A.scANVI',rep,sep='.'))
#     PairwiseDE(count[,batchid==1], pred_labels[,3][batchid==1], 2,3,batchid[batchid==1],paste('A.scANVI',rep,sep='.'))
#     PairwiseDE(count[,batchid==1], pred_labels[,3][batchid==1], 2,4,batchid[batchid==1],paste('A.scANVI',rep,sep='.'))
#     PairwiseDE(count[,batchid==1], pred_labels[,3][batchid==1], 1,2,batchid[batchid==1],paste('A.scANVI',rep,sep='.'))
#
#
#     PairwiseDE(count[,batchid==2], pred_labels[,3][batchid==2], 4,5,batchid[batchid==2],paste('B.scANVI',rep,sep='.'))
#     PairwiseDE(count[,batchid==2], pred_labels[,3][batchid==2], 2,3,batchid[batchid==2],paste('B.scANVI',rep,sep='.'))
#     PairwiseDE(count[,batchid==2], pred_labels[,3][batchid==2], 2,4,batchid[batchid==2],paste('B.scANVI',rep,sep='.'))
#     PairwiseDE(count[,batchid==2], pred_labels[,3][batchid==2], 1,2,batchid[batchid==2],paste('B.scANVI',rep,sep='.'))
#
#
#     PairwiseDE(count, celllabels, 4,5,batchid,paste('AB',rep,sep='.'))
#     PairwiseDE(count, celllabels, 2,3,batchid,paste('AB',rep,sep='.'))
#     PairwiseDE(count, celllabels, 2,4,batchid,paste('AB',rep,sep='.'))
#     PairwiseDE(count, celllabels, 1,2,batchid,paste('AB',rep,sep='.'))
#
#     PairwiseDE(count[,batchid==1], celllabels[batchid==1], 4,5,batchid[batchid==1],paste('A',rep,sep='.'))
#     PairwiseDE(count[,batchid==1], celllabels[batchid==1], 2,3,batchid[batchid==1],paste('A',rep,sep='.'))
#     PairwiseDE(count[,batchid==1], celllabels[batchid==1], 2,4,batchid[batchid==1],paste('A',rep,sep='.'))
#     PairwiseDE(count[,batchid==1], celllabels[batchid==1], 1,2,batchid[batchid==1],paste('A',rep,sep='.'))x
#
#
#     PairwiseDE(count[,batchid==2], celllabels[batchid==2], 4,5,batchid[batchid==2],paste('B',rep,sep='.'))
#     PairwiseDE(count[,batchid==2], celllabels[batchid==2], 2,3,batchid[batchid==2],paste('B',rep,sep='.'))
#     PairwiseDE(count[,batchid==2], celllabels[batchid==2], 2,4,batchid[batchid==2],paste('B',rep,sep='.'))
#     PairwiseDE(count[,batchid==2], celllabels[batchid==2], 1,2,batchid[batchid==2],paste('B',rep,sep='.'))
#
# }
# }
