library(TCGAbiolinks)
setwd("D:/TCGA/")

# Kidney renal clear cell carcinoma
# Download Normal Tissue data
normal_query <- GDCquery(
  project = "TCGA-KIRC", 
  data.category = "Transcriptome Profiling", 
  data.type = "Gene Expression Quantification", 
  workflow.type = "STAR - Counts",
  sample.type="Solid Tissue Normal")

GDCdownload(normal_query)

norm_expdat <- GDCprepare(
  query = normal_query,
  save = TRUE, 
  save.filename = "norm_exp.rda"
)


dim(norm_expdat@assays@data$fpkm_uq_unstrand)  # Output : 60660 72

# Extract protein coding data
protein_coding_idx <- which(norm_expdat@rowRanges$gene_type == "protein_coding")
normal_expr_ptn_coding <- norm_expdat@assays@data$fpkm_uq_unstrand[protein_coding_idx,]


# norm_expdat@colData$sample
# norm_expdat@colData$patient

# Download Tumor data
tumor_query <- GDCquery(
  project = "TCGA-KIRC", 
  data.category = "Transcriptome Profiling", 
  data.type = "Gene Expression Quantification", 
  workflow.type = "STAR - Counts",
  sample.type="Primary Tumor")

GDCdownload(tumor_query)

tum_expdat <- GDCprepare(
  query = tumor_query,
  save = TRUE, 
  save.filename = "tum_exp.rda"
)

dim(tum_expdat@assays@data$fpkm_uq_unstrand)  # Output : 60660 540

# Extracting paired data
tum_expdat@colData$patient
idx <- NULL
for (patient in expdat@colData$patient){
  tmp = which(tum_expdat@colData$patient == patient)
  idx <- c(idx, tmp)
}

# 한줄로 처리하기
# idx <- sapply(expdat@colData$patient, function(arg) which(tum_expdat@colData$patient == arg))

length(idx)
tumor_paired <- tum_expdat[,idx]
dim(tumor_paired@assays@data$fpkm_uq_unstrand)  # Output : 60660 72

# Extract protein coding data
tumor_protein_coding_idx <- which(tumor_paired@rowRanges$gene_type == "protein_coding")

tumor_expr_ptn_coding <- tumor_paired@assays@data$fpkm_uq_unstrand[tumor_protein_coding_idx,]

# apply function 사용법 예시
# rm_idx <- which(apply(tumor_expr_ptn_coding, 1, sd) == 0)



