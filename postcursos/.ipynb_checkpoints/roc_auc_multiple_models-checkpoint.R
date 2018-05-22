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
library(naivebayes)
library(e1071)
library(fastAdaboost)


setwd("~/Workspace/Tesis/postcursos")

cases = c()
for(pr in seq(1:10)){
  for(true_prob in c("None")){
    cases = c(cases, c(paste(pr, "t", "f","f", true_prob,sep=""),
                       paste(pr, "f", "f","f", sep=""),
                       paste(pr, "f", "t","f", sep=""),
                       paste(pr, "t", "t","f",sep=""),                            
                       paste(pr, "m", "t","f", sep=""),
                       paste(pr, "m", "f","f",sep=""),
                       paste(pr, "t", "f","t", true_prob, sep=""),
                       paste(pr, "f", "f","t", sep=""),
                       paste(pr, "f", "t","t", sep=""),
                       paste(pr, "t", "t","t",sep=""),                            
                       paste(pr, "m", "t","t", sep=""),
                       paste(pr, "m", "f","t",sep="")))
  }
}

iter <- 1
good_cases <- c()
for(i in cases){
  if(!(i %in% good_cases)){
    # read the table
    data_path <- paste("~/Workspace/Tesis/data/census/maybe/twodist/negative_census_",i,".csv", sep="")

    if(file.exists(data_path)){
      data <- read.csv(data_path)
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
      
      # hacer la inicializacion de varios modelos
      model_c5 <- C5.0(trainX, trainy, trials=10, control = C5.0Control(minCases = 10 )) # min cases de 8 a 15
      model_naive <- naive_bayes(trainX, trainy)
      model_svm <- svm(x=trainX, y=trainy, cost = 10, gamma = 1)
      #databoost <- data.frame(X=trainX, Y=trainy)
      #databoost$Y <- factor(databoost$Y)
      #model_adaboost <- adaboost(X~Y, databoost, 10)
      models_list <- list("c5"=model_c5, "naive_bayes"=model_naive)#, "svm"=model_svm)#, "adaboost"=model_adaboost)
      models_list <- list("svm"=model_svm)#, "adaboost"=model_adaboost)
      
      #formar el objeto prediction
      names_models <- names(models_list)
      for(ix_model in seq(length(models_list))){
        try(
          {
          model_name <- names_models[ix_model]
          model <- models_list[[ix_model]]
          prediction_obj <- prediction(predict(model, trainX), trainy)
          
          # mext line is used for c5 and naive bayes 
          # prediction_obj <- prediction(predict(model, trainX, type="prob")[,2], trainy)
          #formar el objeto performance
          performance_obj <- performance(prediction_obj, measure="tpr", x.measure="fpr")
          #agregamos el area bajo la curva
          auc <- performance(prediction_obj, measure="auc")
          cur_auc <- auc@y.values[[1]]
          
          if(iter==1){
            roc_df <- data.frame(performance_obj@x.values, performance_obj@y.values, 
                                 rep(i,length(performance_obj@x.values)),
                                 rep(model_name,length(performance_obj@x.values)))
            auc_df <- data.frame(i,cur_auc, model_name)
            names(auc_df) <- c("case", "auc", "model")
            names(roc_df) <- c("fpr", "tpr", "case", "model")
            #plot(performance_obj, col=colors[iter],new=TRUE, colorize = FALSE ,main=paste("C5.0 Census DB Generalization ",sep=""))  
          }
          else{
            roc_df2 <- data.frame(performance_obj@x.values, performance_obj@y.values, 
                                  rep(i,length(performance_obj@x.values)),
                                  rep(model_name,length(performance_obj@x.values)))
            names(roc_df2) <- c("fpr", "tpr", "case", "model" )
            auc_df2 <- data.frame(i,cur_auc, model_name)
            names(auc_df2) <- c("case", "auc", "model")
            roc_df <- rbind(roc_df, roc_df2)
            auc_df <- rbind(auc_df, auc_df2)
            #plot(performance_obj, col=colors[iter], add = TRUE, colorize = FALSE)
          }
          good_cases <- c(good_cases, c(i))
          print(model_name)},
          silent=FALSE
        )
        iter <- iter+1
      }
      print(i)
    }
  }
}


write.csv(roc_df, file = "~/Workspace/Tesis/data/census/maybe/roc/df_c5nb.csv")
write.csv(auc_df, file = "~/Workspace/Tesis/data/census/maybe/roc/df_c5nb_auc.csv")

