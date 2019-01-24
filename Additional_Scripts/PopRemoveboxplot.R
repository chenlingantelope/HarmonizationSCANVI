library(reshape2)
library(ggplot2)

data1 = read.csv('PopRemove.res.txt',header=F,sep='\t',as.is=T)
data2 = read.csv('PopRemove.kp.res.txt',header=F,sep='\t',as.is=T)

result = lapply(unique(as.character(data1[,2])),function(celltype){
	# celltype = 'CD4 T cells'
	temp = data1[data1[,2]==celltype,]
	pops = as.character(temp[1,c(11:18)])
	temp = temp[,c(3:10)]
	colnames(temp) = pops
	temp = melt(temp)
	temp = cbind(rep(c('Seurat','vae','scanvi'),8),temp)
	colnames(temp)=c('method','type','value')
	temp$celltype = c(rep('removed',3),rep('other',21))
	temp$removed = rep(celltype,24)
	return(temp)
})
result = do.call(rbind,result)
result$type = as.character(result$type)
result = result[result$type!='Dendritic Cells' & result$type!='Megakaryocytes',]
result$method = as.character(result$method)
result$method[result$method=='vae']='scVI'
result$method[result$method=='scanvi']='scanVI'
result$method = factor(result$method,levels=c('Seurat','scVI','scanVI'))
result = result[result$type!='Other',]
p <- ggplot(data=result[result$celltype=='other',], aes(x=type, y=value,fill=method)) +
geom_boxplot() + 
theme_minimal() + 
theme(axis.text.x = element_text(angle = 45, hjust = 1)) + 
scale_fill_manual(values=c("blue", "red", "forestgreen"),guide=FALSE) +
ylab('Batch Entropy Mixing') + 
xlab('Cell Type')
ggsave('all_BE.pdf',height=3.2,width=5)

p <- ggplot(data=result[result$celltype=='removed',], aes(x=method, y=value,fill=method)) +
scale_fill_manual(values=c("blue", "red", "forestgreen"),guide=FALSE) +
ylab('Batch Entropy Mixing') + 
geom_boxplot() + 
theme_minimal()
ggsave('removed_acc.pdf',height=3.2,width=2.5)


result = lapply(unique(as.character(data2[,2])),function(celltype){
	# celltype = 'CD4 T cells'
	temp = data2[data2[,2]==celltype,]
	pops = as.character(temp[1,c(11:18)])
	temp = temp[,c(3:10)]
	colnames(temp) = pops
	temp = melt(temp)
	temp = cbind(rep(c('Seurat','vae','scanvi'),8),temp)
	colnames(temp)=c('method','type','value')
	temp$celltype = c(rep('removed',3),rep('other',21))
	temp$removed = rep(celltype,24)
	return(temp)
})
result = do.call(rbind,result)
result$method = as.character(result$method)
result = result[result$type!='Dendritic Cells' & result$type!='Megakaryocytes',]
result$method[result$method=='vae']='scVI'
result$method[result$method=='scanvi']='scanVI'
result$method = factor(result$method,levels=c('Seurat','scVI','scanVI'))
result = result[result$type!='Other',]
p <- ggplot(data=result, aes(x=type, y=value,fill=method)) +
geom_boxplot() + 
theme_minimal() + 
theme(axis.text.x = element_text(angle = 45, hjust = 1)) + 
scale_fill_manual(values=c("blue", "red", "forestgreen"),guide=FALSE) +
ylab('kNN Purity') + 
xlab('Cell Type')
ggsave('all_acc.pdf',height=3.2,width=5)


# no overlap plots
