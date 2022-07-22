# Execue : Ctrl + Enter

#### Data Loading ####

# Import library
library(data.table)

# Set working space
setwd("C:/TCGA-LIHC/")

# Read file
clinical <- fread("TCGA.LIHC.sampleMap_LIHC_clinicalMatrix")  # "" 사이에서 tab 누르면 dir 안의 파일 확인
expr <- fread("gene_expression", data.table = F)

# Expr에 있는 사람들이 모두 clinical에 있는지 확인
all(colnames(expr) %in% clinical$sampleID)  # Output : TRUE

# Expr에서 gene_list 분리
gene_list <- expr[,1]
expr <- expr[,-1]

# Clinical data의 사람 순서를 expression data의 순서에 맞추기
idx <- NULL
for (sample in colnames(expr)){
  tmp = which(clinical$sampleID == sample)
  idx <- c(idx, tmp)
}
length(idx)   # Expression data의 개수와 같은 지 확인
clinical_core  <- clinical[idx,]

colnames(expr)[1:5]   # 잘 맞췄는 지 확인
clinical_core$sampleID[1:5]

# Tumor와 Normal 분리
tumor_idx <- which(clinical_core$sample_type == "Primary Tumor")
expr_tumor <- expr[,tumor_idx]

normal_idx <- which(clinical_core$sample_type == "Solid Tissue Normal")
expr_normal <- expr[,normal_idx]

