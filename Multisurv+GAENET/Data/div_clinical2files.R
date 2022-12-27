clinical <- read.table("/user_home/euiyoung/hek/multisurv/clinical_byEuiyoung.tsv", header = T, row.names = 1, sep = '\t')

for (r in 1:nrow(clinical)){
  cl_data <- clinical[r, ]
  tb <- data.frame(matrix(ncol = 1, nrow = 0))
  for (d in cl_data){
    tb <- rbind(tb, d)
  }
  f_name <- rownames(clinical)[r]
  f_name <- paste0(f_name, ".tsv")
  f_path <- file.path("/user_home/euiyoung/hek/multisurv/mnt/dataA/TCGA/processed/Clinical", f_name)
  write.table(tb, file = f_path, sep = '/t', row.names = F, col.names = F, quote = F)
}
