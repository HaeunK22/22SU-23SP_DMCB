#### labels_total.tsv를 clinical data의 순서대로 정렬 ####
clinical <- read.table("/user_home/euiyoung/hek/multisurv/mnt/dataA/TCGA/processed/Clinical.tsv", header = T, sep = '\t')$X # 맨 윗줄을 column name으로 한다

labels_tt <- read.table("/user_home/euiyoung/hek/multisurv/data/labels_total.tsv", header = T, sep = '\t')
labels <- data.frame(matrix(ncol = 5, nrow = 0))
idx <- NULL
for (id in clinical){
  idx <- which(labels_tt$submitter_id == id)
  if (is.na(idx)){
    print("NOT IN labels_total.tsv!\n")
  }
  else {
    labels <- rbind(labels, labels_tt[idx, ])
  }
}

write.table(labels, file = '/user_home/euiyoung/hek/multisurv/data/labels.tsv', sep = '\t', row.names = F, col.names = T, quote = F) # column name도 저장
