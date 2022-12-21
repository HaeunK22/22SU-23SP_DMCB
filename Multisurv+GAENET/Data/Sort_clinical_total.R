#### clinial_total을 mRNA_files 순서대로 정렬 ####
idx <- NULL
clinical_core <- NULL
mRNA_file_list <- NULL
clinical_core <- data.frame(matrix(ncol = 10, nrow = 0)) # Clinical data
mRNA_file_list <- list.files("/user_home/euiyoung/hek/multisurv/mnt/dataA/TCGA/raw/RNA-seq_FPKM-UQ/")
for (file_uuid in mRNA_file_list){
  idx <- which(df$uuid == file_uuid)[1]
  if (is.na(idx)){ # if uuid is not in uuid_submitter_match.tsv
    mRNA_file_list <- mRNA_file_list[!(mRNA_file_list == file_uuid)]
    print("file_uuid deleted from mRNA_file_list.")
  } else {
    p_id <- df[idx,3]
    if (p_id %in% row.names(clincal_data)){
      clinical_core <- rbind(clinical_core, clincal_data[p_id,])
    } else{ # if there is no clinical information
      mRNA_file_list <- mRNA_file_list[!(mRNA_file_list == file_uuid)]
      print("file_uuid deleted from mRNA_file_list.")
    }
  }
}
colnames(clinical_core) <- c("project_id","gender","race","prior_treatment","prior_malignancy","synchronous_malignancy","treatments_pharmaceutical_treatment_or_therapy","treatments_radiation_treatment_or_therapy","tumor_stage","age_at_diagnosis")

save(mRNA_file_list, file="/user_home/euiyoung/hek/multisurv/mnt/dataA/TCGA/raw/mRNA_file_list.RData")
write.table(clinical_core, file = "/user_home/euiyoung/hek/multisurv/mnt/dataA/TCGA/processed/clinical_core.tsv", sep = "\t", row.names = F, col.names = T, quote = F)
