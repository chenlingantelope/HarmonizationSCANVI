# Title     : EdgeR
# Objective : run EdgeR on the simulated data
# Created by: chenlingantelope
# Created on: 5/8/19

library(edgeR)
setwd('/data/yosef2/users/chenling/symsim_scVI/symsim_result/DE')

count = read.csv("DE.obsv.csv", header=F)
count = t(count)
# count = count[,c(2:length(count[1,]))]
genenames = as.character(c(1:length(count[,1])))
batchid = read.csv("DE.batchid.csv")[,2]
cell_meta = read.csv( "DE.cell_meta.csv" )
celllabels = cell_meta[,3]
PairwiseDE <- function(count, celllabels, pop1, pop2, batchid, name, ncells=30){
    subset1 = c(1:length(celllabels))[celllabels==pop1]
	subset2 = c(1:length(celllabels))[celllabels==pop2]
    subset = c(sample(subset1, ncells, replace=T),sample(subset2, ncells, replace=T))
    count = count[,subset]
	celllabels = celllabels[subset]
	batchid = batchid[subset]
    group = as.factor(celllabels)
    y <- DGEList(counts=count, group=group)
    cdr <- scale(colMeans(count > 0))
    if( length(unique(batchid))>1 ){
        design = model.matrix(~cdr + as.factor(batchid) + group )
    }else{
        design = model.matrix(~cdr + group )
    }
    y$genes <- data.frame(Symbol=genenames)
    y <- calcNormFactors(y)
    y <- estimateDisp(y, design)
    fit <- glmQLFit(y, design, robust=TRUE)
    qlf <- glmQLFTest(fit)
    write.csv(qlf$table, file=paste('EdgeR/',name, '.',pop1,pop2,'.edgeR.csv',sep=''))
}

for(rep in c(1:9)){
for (mis in as.character(c('0.00','0.01','0.05','0.10','0.15','0.20','0.25','0.30'))){
    pred_labels = read.csv(paste('pred_labels.',rep,'.mis',mis,'.csv',sep=''))
    pred_labels = pred_labels+1
    print(paste(rep,mis,sep='.'))
    PairwiseDE(count, pred_labels[,2], 4,5,batchid,paste('AB',rep,mis,sep='.'))
    PairwiseDE(count, pred_labels[,2], 2,3,batchid,paste('AB',rep,mis,sep='.'))
    PairwiseDE(count, pred_labels[,2], 2,4,batchid,paste('AB',rep,mis,sep='.'))
    PairwiseDE(count, pred_labels[,2], 1,2,batchid,paste('AB',rep,mis,sep='.'))

    PairwiseDE(count[,batchid==1], pred_labels[,2][batchid==1], 4,5,batchid[batchid==1],paste('A',rep,mis,sep='.'))
    PairwiseDE(count[,batchid==1], pred_labels[,2][batchid==1], 2,3,batchid[batchid==1],paste('A',rep,mis,sep='.'))
    PairwiseDE(count[,batchid==1], pred_labels[,2][batchid==1], 2,4,batchid[batchid==1],paste('A',rep,mis,sep='.'))
    PairwiseDE(count[,batchid==1], pred_labels[,2][batchid==1], 1,2,batchid[batchid==1],paste('A',rep,mis,sep='.'))


    PairwiseDE(count[,batchid==2], pred_labels[,2][batchid==2], 4,5,batchid[batchid==2],paste('B',rep,mis,sep='.'))
    PairwiseDE(count[,batchid==2], pred_labels[,2][batchid==2], 2,3,batchid[batchid==2],paste('B',rep,mis,sep='.'))
    PairwiseDE(count[,batchid==2], pred_labels[,2][batchid==2], 2,4,batchid[batchid==2],paste('B',rep,mis,sep='.'))
    PairwiseDE(count[,batchid==2], pred_labels[,2][batchid==2], 1,2,batchid[batchid==2],paste('B',rep,mis,sep='.'))


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
