#### Dataset Preparation

# 원하는 결과
# X : (sample, gene)
# time: (sample, )
# event: (sample, )

# KIRC-US, RECA-EU  (Kideny cancer)
# PAAD-US (Pancreas)
# LAML-US (Leukemia)

# TCGA data는 R에서 TCGAbiolinks를 사용해서 data collection
# RECA-EU만 ICGC에서 코드로 처리 (donor, specimen 등의 자료 받아서)
# https://dcc.icgc.org/ -> Data Release -> current
# Tumor Normal 따로

library(TCGAbiolinks)
library(data.table)
library(MASS)
setwd("D:/Survival Analysis/Cox-nnet")

# Download KIRC Tumor data
tumor_query <- GDCquery(
  project = "TCGA-KIRC", 
  data.category = "Transcriptome Profiling", 
  data.type = "Gene Expression Quantification", 
  workflow.type = "STAR - Counts",
  sample.type="Primary Tumor")

GDCdownload(tumor_query)

tumor_expdat <- GDCprepare(
  query = tumor_query,
  save = FALSE, 
  save.filename = "KIRC_tum_exp.rda"
)

# dim(tumor_expdat@assays@data$fpkm_uq_unstrand)  # Output : 60660 540
KIRC_expr <- t(tumor_expdat@assays@data$fpkm_uq_unstrand)
ptn_cod_idx<-which(tumor_expdat@rowRanges$gene_type == "protein_coding")
write.table(KIRC_expr[,ptn_cod_idx], file="KIRC_expr.tsv", sep = "\t", row.names = F, col.names = F) # Split by tab

# Clinical data
clinical <- fread("D:/TCGA/clinical_data_all.tsv")

# Expr에 있는 사람들이 모두 clinical에 있는지 확인
all(tumor_expdat@colData$patient %in% clinical$`Patient ID`)

# Clinical data의 사람 순서를 expression data의 순서에 맞추기
idx <- sapply(tumor_expdat@colData$patient, function(arg) which(clinical$`Patient ID` == arg)[1]) # When one patient has several clinical data, choose first data
length(idx)   # Expression data의 개수와 같은 지 확인
clinical_core  <- clinical[idx,]

# Check NA
# all(!is.na(clinical_core$`Overall Survival (Months)`))
# all(!is.na(clinical_core$`Overall Survival Status`))
time <- clinical_core$`Overall Survival (Months)`
event <- clinical_core$`Overall Survival Status`
write.table(time, file="KIRC_time.tsv", sep = "\t", row.names = F, col.names = F)
write.table(event, file="KIRC_event.tsv", sep = "\t", row.names = F, col.names = F)




# Download PAAD Tumor data
tumor_query <- GDCquery(
  project = "TCGA-PAAD", 
  data.category = "Transcriptome Profiling", 
  data.type = "Gene Expression Quantification", 
  workflow.type = "STAR - Counts",
  sample.type="Primary Tumor")

GDCdownload(tumor_query)

tumor_expdat <- GDCprepare(
  query = tumor_query,
  save = FALSE, 
  # save.filename = "PAAD_tum_exp.rda"
)

# dim(tumor_expdat@assays@data$fpkm_uq_unstrand)  # Output : 60660 178
PAAD_expr <- t(tumor_expdat@assays@data$fpkm_uq_unstrand)
ptn_cod_idx<-which(tumor_expdat@rowRanges$gene_type == "protein_coding")
write.table(PAAD_expr[,ptn_cod_idx], file="PAAD_expr.tsv", sep = "\t", row.names = F, col.names = F) # Split by tab

# Expr에 있는 사람들이 모두 clinical에 있는지 확인
all(tumor_expdat@colData$patient %in% clinical$`Patient ID`)

# Clinical data의 사람 순서를 expression data의 순서에 맞추기
idx <- sapply(tumor_expdat@colData$patient, function(arg) which(clinical$`Patient ID` == arg)[1])
length(idx)   # Expression data의 개수와 같은 지 확인
clinical_core  <- clinical[idx,]

# Check NA
# all(!is.na(clinical_core$`Overall Survival (Months)`))
# all(!is.na(clinical_core$`Overall Survival Status`))
time <- clinical_core$`Overall Survival (Months)`
event <- clinical_core$`Overall Survival Status`
write.table(time, file="PAAD_time.tsv", sep = "\t", row.names = F, col.names = F)
write.table(event, file="PAAD_event.tsv", sep = "\t", row.names = F, col.names = F)




# Download LAML Tumor data
tumor_query <- GDCquery(
  project = "TCGA-LAML", 
  data.category = "Transcriptome Profiling", 
  data.type = "Gene Expression Quantification", 
  workflow.type = "STAR - Counts",
  sample.type="Primary Blood Derived Cancer - Peripheral Blood")  # Check sample type in website

GDCdownload(tumor_query)

tumor_expdat <- GDCprepare(
  query = tumor_query,
  save = TRUE, 
  save.filename = "LAML_tum_exp.rda"
)

# Expr에 있는 사람들이 모두 clinical에 있는지 확인
all(tumor_expdat@colData$patient %in% clinical$`Patient ID`)

# Clinical data의 사람 순서를 expression data의 순서에 맞추기
idx <- sapply(tumor_expdat@colData$patient, function(arg) which(clinical$`Patient ID` == arg)[1])
length(idx)   # Expression data의 개수와 같은 지 확인
clinical_core  <- clinical[idx,]

# Check NA
# all(!is.na(clinical_core$`Overall Survival (Months)`))  # Output: FALSE
# all(!is.na(clinical_core$`Overall Survival Status`))

# dim(tumor_expdat@assays@data$fpkm_uq_unstrand)  # Output : 60660 150
LAML_expr <- t(tumor_expdat@assays@data$fpkm_uq_unstrand)
ptn_cod_idx<-which(tumor_expdat@rowRanges$gene_type == "protein_coding")

time_exist <- which(!is.na(clinical_core$`Overall Survival (Months)`))  # 139명
write.table(LAML_expr[time_exist,ptn_cod_idx], file="LAML_expr.tsv", sep = "\t", row.names = F, col.names = F) # Split by tab

clinical_core <- clinical_core[time_exist,]

time <- clinical_core$`Overall Survival (Months)`
event <- clinical_core$`Overall Survival Status`
write.table(time, file="LAML_time.tsv", sep = "\t", row.names = F, col.names = F)
write.table(event, file="LAML_event.tsv", sep = "\t", row.names = F, col.names = F)