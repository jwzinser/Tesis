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

setwd("~/Workspace/Tesis/investigacionII")

png("/home/juanzinser/Workspace/Tesis/entregas/plots/census_negativte_all.png")
cases = c()
for(pr in seq(1:10)){
  for(true_prob in c("None")){
        cases = c(cases, c(paste(pr, "t", "f", true_prob,sep=""),
                            paste(pr, "f", "f", sep=""),
                            paste(pr, "f", "t", sep=""),
                            paste(pr, "t", "t",sep=""),                            
                           paste(pr, "m", "t", sep=""),
                           paste(pr, "m", "f",sep="")))
  }
}

colors <- heat.colors(length(cases), alpha = 1)
iter <- 1
good_cases <- c()
good_colors <- c()
for(i in cases){
  if(!(i %in% good_cases)){
  # read the table
  data_path <- paste("~/Workspace/Tesis/data/census/maybe/negative_census_",i,".csv", sep="")
  if(file.exists(data_path)){
  data <- read.csv(data_path)
  print(i)
  data <- data[sample(nrow(data)),]
  data[is.na(data)] <- 0
  
  columns <- c()
  for(col in seq(1:dim(data)[2])){
    columns[col] <- (length(unique(data[,col]))>1)
  }
  data <- data[,columns]
  
  
  X <- data[,1:(dim(data)[2]-1)]
  y <- data[,dim(data)[2]]
  y <- as.factor(as.integer(y==" >50K"))
  #y <- as.integer(y==">50K")
  ll <- floor(dim(data)[1]*.7)
  ul <- dim(data)[1]
  trainX <- X[1:ll,]
  trainy <- y[1:ll]
  testX <- X[(ll+1):ul,]
  testy <- y[(ll+1):ul]
  
  # hacer
  model <- C5.0(trainX, trainy, trials=10, control = C5.0Control(minCases = 10 )) # min cases de 8 a 15
  
  #formar el objeto prediction
  prediction_obj <- prediction(predict(model, trainX, type="prob")[,2], 
                               trainy)
  
  #formar el objeto performance
  performance_obj <- performance(prediction_obj, measure="tpr", x.measure="fpr")
  #agregamos el area bajo la curva
  auc <- performance(prediction_obj, measure="auc")
  cur_auc <- auc@y.values[[1]]
  
  if(iter==1){
    roc_df <- data.frame(performance_obj@x.values, performance_obj@y.values, 
                         rep(i,length(performance_obj@x.values)))
    auc_df <- data.frame(i,cur_auc)
    names(auc_df) <- c("case", "auc")
    names(roc_df) <- c("fpr", "tpr", "case")
    #plot(performance_obj, col=colors[iter],new=TRUE, colorize = FALSE ,main=paste("C5.0 Census DB Generalization ",sep=""))  
  }
  else{
    roc_df2 <- data.frame(performance_obj@x.values, performance_obj@y.values, 
                         rep(i,length(performance_obj@x.values)))
    names(roc_df2) <- c("fpr", "tpr", "case")
    auc_df2 <- data.frame(i,cur_auc)
    names(auc_df2) <- c("case", "auc")
    roc_df <- rbind(roc_df, roc_df2)
    auc_df <- rbind(auc_df, auc_df2)
    #plot(performance_obj, col=colors[iter], add = TRUE, colorize = FALSE)
  }
  good_cases <- c(good_cases, c(i))
  good_colors <- c(good_colors, c(colors[iter]))
  #agregamos la linea de 45º
  #abline(a=0, b= 1, col="black")
  iter <- iter+1
    }
  }
}

roc_df_up <- roc_df %>% rowwise() %>% mutate(privacy = substr(case,1,1)) %>% mutate(include_real = substr(case,2,2)=="t") %>% mutate(uniform = substr(case, 3,3)=="t") %>% mutate(prob_of_real = substr(case, 4, 10))

write.csv(roc_df, file = "~/Workspace/Tesis/data/census/maybe/roc/df.csv")

# para las que son 
for(p in c(1:9)){
  rp <- ggplot(roc_df_up %>% filter(!uniform) %>% filter(privacy==p), aes(fpr, tpr, color = case)) + geom_line(size=.5) + labs(title= "ROC curve", 
                                                                                                       x = "False Positive Rate (1-Specificity)", 
                                                                                                       y = "True Positive Rate (Sensitivity)")
  print(rp)
  ggsave(paste('/home/juanzinser/Workspace/Tesis/entregas/plots/negative_census_',p,'.png',sep=""))
}

options(viewer = NULL)
viewer <- getOption("viewer")

dev.off()

save_roc_df <- "../data/rdata/roc_df.RData"
save(roc_df, file = save_roc_df)
save.image()

save_roc_df <- "../data/rdata/roc_df.RData"
load( file = save_roc_df)

save_auc_df <- "../data/rdata/auc_df.RData"
save(auc_df, file = save_auc_df)
save.image()
auc_sep <- auc_df %>% rowwise() %>% mutate(privacy = substr(case,1,1)) %>% mutate(include_real = substr(case,2,2)=="t") %>% mutate(uniform = substr(case, 3,3)=="t") %>% mutate(prob_of_real = substr(case, 4, 10)) %>% dplyr::select(-case)
auc_tt <- auc_sep %>% filter(include_real) %>% filter(uniform) %>% dplyr::select(auc, privacy) %>% rename(real=auc)
auc_ft <- auc_sep %>% filter(!include_real) %>% filter(uniform) %>% dplyr::select(auc, privacy) %>% rename(nonreal=auc)
auc_uniforms <- auc_ft %>% inner_join(auc_tt)
xtable(auc_uniforms %>% dplyr::select(privacy, nonreal, real))

auc_ordered <- auc_sep %>% arrange(desc(auc))
uniform_auc <- auc_ordered %>% filter(uniform)

auc_tf <- auc_sep %>% filter(include_real) %>% filter(!uniform) %>% filter(prob_of_real=="None") %>%  dplyr::select(auc, privacy) %>% rename(real=auc)
auc_ff <- auc_sep %>% filter(!include_real) %>% filter(!uniform) %>% dplyr::select(auc, privacy) %>% rename(nonreal=auc)
auc_nonuniforms <- auc_tf %>% inner_join(auc_ff)
xtable(auc_nonuniforms %>% dplyr::select(privacy, nonreal, real))

#load(file=save_path)
#legend(x=0.6, y=.3,legend=good_cases,
#       col=good_colors, lty=c(1,1,1,1))

# el mejor punto de corte, rocr, para encontrae el mejor TPR, y FNR
# label ordering, para decirle cuales son el positivo y el negativo



# tile plot of the 
auc_for_tile <- (auc_sep %>% filter(prob_of_real=="None" | prob_of_real=="") %>% 
                   mutate(case=paste(ifelse(include_real,"T","F"),ifelse(uniform,"T","F"),"")) %>% 
                   dplyr::select(privacy,case,auc))
gg_ch <- (ggplot(auc_for_tile, aes(y = factor(case),  x = factor(privacy))) +  
            geom_tile(aes(fill = auc)) + ggtitle("AUC by case and privacy")
          #+ scale_fill_continuous(low = "green", high = "red")
)
gg_ch
ggsave("plots/tile_auc_case.png")
