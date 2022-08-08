if(!require(devtools)) install.packages("devtools")
devtools::install_github("sinhrks/ggfortify")
library(ggfortify)

# Binding normal and tumor expression data
expr <- cbind(normal_expr_ptn_coding, tumor_expr_ptn_coding)
type <- rep(c('Normal','Tumor'),each=ncol(normal_expr_ptn_coding))

# PCA
expr.t <- t(expr)
expr.pca <- prcomp(expr.t)
summary(expr.pca)

expr_type <- cbind(expr.t, type)
autoplot(expr.pca, data = expr_type, colour = 'type', scale = 0)
# plot(expr.pca$x[,1],expr.pca$x[,2])

pca_matrix <- cbind(expr.pca$x[,1],expr.pca$x[,2])

# Calculate Euclidean Distance
euclidean <- function(a,b) sqrt(sum((a-b)^2))

distance <- NULL
for (i in 1:ncol(normal_expr_ptn_coding)){
  tmp = euclidean(pca_matrix[i,], pca_matrix[i+ncol(normal_expr_ptn_coding),])
  distance <- c(distance, tmp)
}

# Distance and Metastasis(전이)
metastasis <- clinical_core$`American Joint Committee on Cancer Metastasis Stage Code`
metastasis_distance <- cbind(distance, metastasis)

metastasis_distance <- as.data.frame(metastasis_distance) # Metastasis와 distance의 data type이 달라도 됨.
colnames(metastasis_distance) <- c("Distance", "Metastasis")
metastasis_distance$Distance <- as.numeric(metastasis_distance$Distance)
metastasis_distance$Metastasis <- as.factor(metastasis_distance$Metastasis)
# View(metastasis_distance)

# T-test
t.test(metastasis_distance$Distance[metastasis_distance$Metastasis == "M0"],
       metastasis_distance$Distance[metastasis_distance$Metastasis == "M1"])  # p-value = 0.357


# Distance and Tumor Stage
stage <- clinical_core$`American Joint Committee on Cancer Tumor Stage Code`
stage_distance <- cbind(distance, stage)
