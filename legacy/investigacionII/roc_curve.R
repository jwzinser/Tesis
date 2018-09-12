library(C50)
library(tidyr)
library(ROCR)
library(knitr)

setwd("~/Workspace/Tesis/entregas")


data_path <- paste("~/Workspace/Tesis/data/census/negative_census_2tt.csv", sep="")
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
# hacer
model <- C5.0(trainX, trainy, trials=10, control = C5.0Control(minCases = 10 )) # min cases de 8 a 15

#formar el objeto prediction
prediction_obj <- prediction(predict(model, trainX, type="prob")[,2], 
                             trainy)


#formar el objeto performance
performance_obj <- performance(prediction_obj, measure="tpr", x.measure="fpr")
#agregamos el area bajo la curva
auc <- performance(prediction_obj, measure="auc")
plot(performance_obj, main=paste("C5.0 Census DB Generalization level ",i,sep=""))
#agregamos la linea de 45º
abline(a=0, b= 1, col="red")


#formar el objeto performance
perf_tn_fn <- performance(prediction_obj, measure="tnr", x.measure = "fnr")

roc_table <- data.frame(cutoff=performance_obj@alpha.values[[1]],
                        tp=performance_obj@y.values[[1]],
                        fp=performance_obj@x.values[[1]],
                        tn=perf_tn_fn@y.values[[1]],
                        fn=perf_tn_fn@x.values[[1]])

kable(roc_table, format.args=list(big.mark=","))


dev.copy(rp, '/home/juanzinser/Workspace/Tesis/entregas/plots/negative_census_uniform.png')
cat(model$output)
