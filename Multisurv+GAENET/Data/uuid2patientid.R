#### File uuid to TCGA patient ID ####

setwd("/user_home/euiyoung/hek/")

## Uuid to case_id ##
file_dir <- "/user_home/euiyoung/hek/gdc-docs/docs/Data/Release_Notes/GCv36_Manifests/"

# Merge TCGA files >> uuid2caseid(id, filename, md5, project_id, case_id ...)
TCGA_filelist <- list.files(file_dir)[grep(pattern = "TCGA", list.files(file_dir))]
uuid2caseid <- NULL
uuid2caseid <- data.frame(matrix(ncol = 14, nrow = 0))
for(file_name in TCGA_filelist){
  n_file_path <- file.path("gdc-docs/docs/Data/Release_Notes/GCv36_Manifests", file_name)
  n_file <- read.table(n_file_path, sep = '\t', fill = TRUE)
  uuid2caseid <- rbind(uuid2caseid, n_file)
}
x <- c("id", "filename", "md5", "size", "project_id", "case_ids", "aliquot_ids", "data_category", "data_type", "experimental_strategy", "workflow_type", "new_id", "new_md5", "new_size")
colnames(uuid2caseid) <- x

# file_uuid : list of file uuid
file_uuid <- read.table("multisurv/file_uuid.txt")$V1

# case_id : list of case_ids
case_id <- NULL
for (uuid in file_uuid){
  temp <- uuid2caseid[which(uuid2caseid$id == uuid), "case_ids"] # temp = case_id of uuid
  if(length(temp)==0) {
    file_uuid <- file_uuid[!(file_uuid == uuid)]
    print("file_uuid deleted.")
  } else {
    case_id <- append(case_id, temp)
    print(length(case_id))
  }
}
# 한 줄로 처리
# case_id_ <- uuid2caseid$case_ids[unlist(sapply(file_uuid, function(arg) which(uuid2caseid$id == arg)))]

# df = (uuid, case_id)
df <- data.frame(file_uuid, case_id)
dim(df)
y <- c("uuid", "caseid")
colnames(df) <- y


## Case_id to submitter_id ##
# local에서 처리 -> save(case_id, file="./multisurv/mRNA_cas행eid.RData")
# local로 파일 옮기고 코드 실행

# https://www.bioconductor.org/packages/devel/bioc/vignettes/TCGAutils/inst/doc/TCGAutils.html#installation
# submitter_id : list of submitter_id
submitter_id <- NULL
for (c_id in case_id){
  submitter_id <- append(submitter_id, UUIDtoBarcode(c_id, from_type='case_id')$submitter_id)
}
# save(submitter_id, file="D:/Multisurv/data/mRNA_submitterid.RData")
# 다시 server로 옮기기

# df = (id, case_id, patient_id)
df$patient_id = submitter_id
