library(TCGAbiolinks)

#### Data Download ####
cancers <- c("SARC","MESO","ACC","READ","LGG","THCA","CHOL",
             "KIRC","BRCA","OV","GBM","SKCM","DLBC","KICH",
             "UVM","THYM","TGCT","LUSC","PRAD","UCEC",
             "STAD","ESCA","HNSC","LIHC","COAD","LUAD","CESC",
             "PAAD","UCS","KIRP","PCPG","BLCA")

proj <- paste0("TCGA-",cancers)
query <- GDCquery(project = proj,
                  data.category = "Transcriptome Profiling",
                  data.type = "miRNA Expression Quantification",
                  sample.type = c("Primary Tumor"))
setwd("D:/MLSurv/Data")
GDCdownload(query)

#### Data organization #####

query_result_list <- list()
for(project in proj){
  try({
    project_query <- GDCquery(project =project,
                              data.category = "Transcriptome Profiling",
                              data.type = "miRNA Expression Quantification",
                              sample.type = c("Primary Tumor"));
    
    query_result <- GDCprepare(project_query);
    query_result_list[[project]] <- query_result;
  })
}

gene_expr_list = list()
gene_expr <- NULL
for(project in names(query_result_list)){
  query_result <- query_result_list[[project]]
  idx <- NULL
  miRNA_list <- query_result$miRNA_ID   # miRNA_ID
  for (i in 1:ncol(query_result)){
    if (i%%3 == 0){
      idx <- append(idx, i)
    }
  }
  gene_expr_list[[project]] <- query_result[,idx]
}

save(gene_expr_list, file="./mirna_expr_list.RData")

rowranges <- data.frame(id=query_result$miRNA_ID)
write.table(rowranges, "D:/MLSurv/Data/miRNA_rowranges.tsv", row.names = F, col.names = F, quote = F, sep = "\t")
