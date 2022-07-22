# Import library
library(data.table)

# Set working space
setwd("C:/TCGA-LIHC/")

# Read file
clinical <- fread("TCGA.LIHC.sampleMap_LIHC_clinicalMatrix")  # "" 사이에서 tab 누르면 dir 안의 파일 확인
expr <- fread("gene_expression", data.table = F)

# expr에 있는 사람들이 모두 clinical에 있는지 확인
all(colnames(expr) %in% clinical$sampleID)  # Output : TRUE

# expr에서 gene_list 분리
gene_list <- expr[,1]
expr <- expr[,-1]

# Execue : Ctrl + Enter
