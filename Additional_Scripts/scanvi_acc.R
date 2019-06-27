library(ggplot2)
library(reshape2)


# prop = rep(0.2,5)
# prop = rbind(c(1:5),prop)
# colnames(prop)=c(1:5)


prop = read.table('celltypeprop.txt',sep='\t',row.names=NULL,as.is=T)
ncelltypes = length(prop[1,])-1
colnames(prop) = prop[1,]
prop = prop[2:4,1:ncelltypes]
colnames(prop)
colnames(prop)[colnames(prop)=='NaN']='nan'
colnames(prop) = sapply(colnames(prop), function(x){gsub("+", ".", x,fixed=T)})
colnames(prop) = sapply(colnames(prop), function(x){gsub(" ", ".", x,fixed=T)})
colnames(prop) = sapply(colnames(prop), function(x){gsub("-", ".", x,fixed=T)})

others = read.table('others.percluster.res.txt',sep='\t',row.names=NULL,as.is=T)
colnames(others) = c(colnames(others)[2:length(colnames(others))],'NA')
others = others[,1:length(others[1,])-1]

scvi = read.table('scvi.percluster.res.txt',sep='\t',row.names=NULL,as.is=T)
colnames(scvi) = c(colnames(scvi)[2:length(colnames(scvi))],'NA')
scvi = scvi[,1:length(scvi[1,])-1]

scanvi = read.csv('scanvi_acc.txt',sep='\t',row.names=NULL,as.is=T)
# ncols = length(scanvi[1,])
# scanvi = scanvi[,c(1,c(3:(ncols-1)))]

Dotplot <- function(others,scvi,scanvi,ann,methods,plotname){
	scvi = scvi[scvi[,2]==ann & scvi[,1]=='vae',]
	others = others[others[,2]==ann,]
    celltypes = colnames(prop)
    if(ann=='p1'){
		scmap = scanvi[scanvi[,1]=='scmap1',]
		coral = scanvi[scanvi[,1]=='coral1',]
        scanvi = scanvi[scanvi[,1]=='scanvi1',]
    }else if(ann=='p2'){
		scmap = scanvi[scanvi[,1]=='scmap2',]
		coral = scanvi[scanvi[,1]=='coral2',]
        scanvi = scanvi[scanvi[,1]=='scanvi2',]
    }else if(ann=='p'){
        scanvi = scanvi[scanvi[,1]=='scanvi',]
    }
	if(ann=='p1'){prop_values = as.numeric(prop[2,])
		}else if(ann=='p2'){prop_values = as.numeric(prop[3,])
			}else{prop_values = as.numeric(prop[1,])}
	res = lapply(celltypes,function(celltype){
		if(ann!='p'){
			temp =  c(others[,colnames(others)==celltype],
			scvi[,colnames(scvi)==celltype],
            scanvi[,colnames(scanvi)==celltype],
			scmap[,colnames(scmap)==celltype],
			coral[,colnames(coral)==celltype]
		)}else{
			temp =  c(others[,colnames(others)==celltype],
			scvi[,colnames(scvi)==celltype],
            scanvi[,colnames(scanvi)==celltype]
		)}
		return(temp)
	})
	res = do.call(cbind,res)
	if(ann=='p'){
		rownames(res) = c(others[,1],scvi[,1],'scanvi')
	}else{
		rownames(res) = c(others[,1],scvi[,1],'scanvi','scmap','coral')
	}
	celltypes = celltypes[order(prop_values)]
	res = res[,order(prop_values)]
	prop_values = prop_values[order(prop_values)]
    prop_values = prop_values[!is.na(colSums(res))]
	celltypes = celltypes[!is.na(colSums(res))]
	res = res[,!is.na(colSums(res))]
	prop_values = prop_values[celltypes!='nan']
	res = res[,celltypes!='nan']
	celltypes = celltypes[celltypes!='nan']
    df = data.frame(x=c(1:length(celltypes)), scVI=res[rownames(res)=='vae',],celltypes= celltypes)
	if('SCMAP' %in% methods){df$SCMAP = res[rownames(res)=='scmap',]}
	if('CCA' %in% methods){df$CCA = res[rownames(res)=='readSeurat',]}
	if('SCANVI' %in% methods){df$SCANVI = res[rownames(res)=='scanvi',]}
	if('CORAL' %in% methods){df$CORAL = res[rownames(res)=='coral',]}
    if('CORAL' %in% methods){
		colors = c('red','darkgreen','orange','blue','green')
	}else if ('SCMAP' %in% methods){
            colors = c('red','darkgreen','orange','blue')
    }else if('CCA' %in% methods){
            colors = c('red','darkgreen','blue')
    }
	df$prop = prop_values
	df = df[df['SCANVI']>=0,]
	df = melt(df,id=c('celltypes','prop','x'))
	if(ann=='p'){
		df$variable = factor(df$variable, levels = c('scVI','SCANVI','CCA'))
	}else{
		df$variable = factor(df$variable, levels = c('scVI','SCANVI','SCMAP','CCA','CORAL'))
	}
    df$celltypes = gsub("+", "", df$celltypes, fixed=TRUE)
    df$celltypes = gsub("..", " ", df$celltypes, fixed=TRUE)
    df$celltypes = gsub(".", " ", df$celltypes, fixed=TRUE)
    # print(df$celltypes)
	p = ggplot(df,aes(x,value)) +
	geom_point(aes(size=prop,colour=variable), position = position_dodge(.3)) + ylim(0,1.1)  +
	scale_size_area(max_size = max(df$prop)*40,breaks=c(0,0.05,0.1,0.15,0.2))+
	scale_x_continuous(breaks=df$x[df$variable=='scVI'],labels=df$celltypes[df$variable=='scVI']) +
	theme(text = element_text(size=25),axis.text.x = element_text(angle = 45, hjust = 1),
        plot.margin=unit(c(6, 30, 6, 100), "points"),
    axis.title.x=element_blank(),axis.title.y=element_blank(),
    legend.position="none"
    ) +
    scale_color_manual(values=colors)+ ylab("Cell Type Classification Accuracy")+xlab('Cell Type Labels')
	ggsave(plotname,p,width=10,height=6)


    p = ggplot(df,aes(x,value,fill=variable)) +
	scale_fill_manual(values=colors)+ geom_boxplot() +
    ylab("Cell Type Classification Accuracy")+xlab('Method') +
    scale_y_continuous(limits = c(0, 1))+
    # theme(
    theme(legend.position="none",
    axis.title.x=element_blank(),axis.title.y=element_blank(),text = element_text(size=50),
    axis.text.x=element_blank(), axis.ticks.x=element_blank())
	ggsave(paste('box_',plotname,sep=''),p)
}


Dotplot(others,scvi,scanvi,'p1',c('scVI','SCANVI','SCMAP','CCA','CORAL'),'percluster_p1.pdf')
Dotplot(others,scvi,scanvi,'p2',c('scVI','SCANVI','SCMAP','CCA','CORAL'),'percluster_p2.pdf')
Dotplot(others,scvi,scanvi,'p',c('scVI','SCANVI','CCA'),'percluster_p.pdf')

