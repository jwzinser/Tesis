library(C50)
library(tidyr)
library(ggplot2)
library(rpart)
library(tree)
library(party)
library(randomForest)
library(pROC)
library(nnet)
library(ROCR)
library(plotROC)
library(dplyr)
library(xtable)

setwd("~/Workspace/Tesis/entregas")

png("/home/juanzinser/Workspace/Tesis/entregas/plots/census_negativte_all.png")
cases = c()
colors <- heat.colors(length(cases), alpha = 1)

roc_df_up <- roc_df %>% rowwise() %>% mutate(privacy = substr(case,1,1)) %>% mutate(include_real = substr(case,2,2)=="t") %>% mutate(uniform = substr(case, 3,3)=="t") %>% mutate(prob_of_real = substr(case, 4, 10))

# uniformes incluyendo real
rp <- ggplot(roc_df_up %>% filter(uniform) %>% filter(include_real), aes(fpr, tpr, color = case)) + geom_line(size=.5) + labs(title= "ROC curve", x = "False Positive Rate (1-Specificity)", y = "True Positive Rate (Sensitivity)")
ggsave('/home/juanzinser/Workspace/Tesis/entregas/plots/negative_census_uniform_w_real.png')

# uniformes sin incluir real
rp <- ggplot(roc_df_up %>% filter(uniform) %>% filter(!include_real), aes(fpr, tpr, color = case)) + geom_line(size=.5) + labs(title= "ROC curve", x = "False Positive Rate (1-Specificity)", y = "True Positive Rate (Sensitivity)")
ggsave('/home/juanzinser/Workspace/Tesis/entregas/plots/negative_census_uniform_wo_real.png')

# not uniform sin incluir real
rp <- ggplot(roc_df_up %>% filter(!uniform) %>% filter(!include_real), aes(fpr, tpr, color = case)) + geom_line(size=.5) + labs(title= "ROC curve", x = "False Positive Rate (1-Specificity)", y = "True Positive Rate (Sensitivity)")
ggsave('/home/juanzinser/Workspace/Tesis/entregas/plots/negative_census_nonuniform_wo_real.png')

# no uniform not including the real
for(p in c(1:9)){
  rp <- ggplot(roc_df_up %>% filter(!uniform) %>% filter(include_real) %>% filter(privacy==p), aes(fpr, tpr, color = case)) + geom_line(size=.5) + labs(title= "ROC curve", x = "False Positive Rate (1-Specificity)", y = "True Positive Rate (Sensitivity)")
  ggsave(paste('/home/juanzinser/Workspace/Tesis/entregas/plots/negative_census_nouniform_w_real_',p,'.png',sep=""))
}

options(viewer = NULL)
viewer <- getOption("viewer")

dev.off()

save_roc_df <- "../data/rdata/roc_df.RData"
save(roc_df, file = save_roc_df)
save.image()

save_auc_df <- "../data/rdata/auc_df.RData"
save(auc_df, file = save_auc_df)
save.image()
auc_sep <- auc_df %>% rowwise() %>% mutate(privacy = substr(case,1,1)) %>% mutate(include_real = substr(case,2,2)=="t") %>% mutate(uniform = substr(case, 3,3)=="t") %>% mutate(prob_of_real = substr(case, 4, 10)) %>% select(-case)
auc_ordered <- auc_sep %>% arrange(desc(auc))
uniform_auc <- auc_ordered %>% filter(uniform)


xtable(uniform_auc)

#load(file=save_path)
#legend(x=0.6, y=.3,legend=good_cases,
#       col=good_colors, lty=c(1,1,1,1))

# el mejor punto de corte, rocr, para encontrae el mejor TPR, y FNR
# label ordering, para decirle cuales son el positivo y el negativo
