#### File uuid to TCGA patient ID ####

setwd("/user_home/euiyoung/hek/")

file_dir <- "/user_home/euiyoung/hek/gdc-docs/docs/Data/Release_Notes/GCv36_Manifests/"

TCGA_filelist <- list.files(file_dir)[grep(pattern = "TCGA", list.files(file_dir))]

for(file_name in TCGA_filelist){
  n_file_path <- file.path("gdc-docs/docs/Data/Release_Notes/GCv36_Manifests", file_name)
  n_file <- read.table(n_file_path)
  file <- rbind(file, n_file)
}

file_uuid <- read.table("multisurv/file_uuid.txt")$V1
file_uuid
