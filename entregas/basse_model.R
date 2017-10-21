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


setwd("~/Workspace/Tesis/entregas")

colors = c("red", "blue", "green", "yellow")
png("/home/juanzinser/Workspace/Tesis/entregas/plots/census_level_all.png")

for(i in c(0,1,2,3)){
  # read the table

  data_path <- paste("~/Workspace/Tesis/data/census/census_level_",i,".csv", sep="")
  data <- read.csv(data_path)
  
  data <- data[sample(nrow(data)),]
  
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
  print(auc@y.values[[1]])
  
  if(i==0){
    plot(performance_obj, col=colors[(i+1)],new=TRUE, colorize = FALSE ,main=paste("C5.0 Census DB Generalization ",sep=""))  
  }
  else{
    plot(performance_obj, col=colors[(i+1)], add = TRUE, colorize = FALSE)
  }
  
  #agregamos la linea de 45º
  #abline(a=0, b= 1, col="black")

}
legend(x=0.6, y=.3,legend=c("Level 0", "Level 1", "Level 2", "Level 3"),
       col=colors, lty=c(1,1,1,1))
dev.off()

# el mejor punto de corte, rocr, para encontrae el mejor TPR, y FNR
# label ordering, para decirle cuales son el positivo y el negativo
