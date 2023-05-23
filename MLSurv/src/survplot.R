library(survival)
library(survminer)


# Load from server to local
label = read.table("D:/haeun/test_label.tsv", sep = "\t", header = T) 
output = read.table("D:/haeun/output.tsv", sep = "\t")

thres = mean(output[,3]) # 3-Years. (Mean? Median?)
group = ifelse(output[,3]>=thres,"Low","High")

survdiff(Surv(label$time, label$event)~ group)

fit <- survfit(Surv(label$time, label$event) ~ group)
ggsurvplot(fit, data = label, pval=TRUE, ggtheme = theme_bw())
