
setwd("/home/juanzinser/Workspace/Tesis/entregas")
library(plotrix)
library(tidyr)
library(dplyr)
library(ggplot2)
library(stringr)
library(stringdist)
library(lubridate)
library(aws.s3)
library(forecast)
# recorridos historicos
reco_df2 <- read.csv("../data/census/negative_census_reconstruction_df.csv")
reco_df <- read.csv("../data/census/negative_census_reconstruction_rmse_df.csv")
reco_sep <- (reco_df %>% rowwise() %>% mutate(privacy = substr(case,1,1)) %>% 
               mutate(include_real = substr(case,2,2)=="t") %>% 
               mutate(uniform = substr(case, 3,3)=="t") %>% 
               mutate(prob_of_real = substr(case, 4, 10)))

reco <- reco_sep %>% gather(cn, sum, CIS, NIS) # histogram basic 
columns <-unique(reco_df$column)

# use education for the histogram as representative column
educ_data <- (reco %>% filter(column=="education"))

educ_data_2 <- educ_data %>% mutate(cn=ifelse(cn=="CIS","real",paste("n_",ifelse(include_real,"t","f"),
                                                                   ifelse(uniform,"t","f"),sep="")))

# plots for privacy level with 5 bars per class
# an histogram must filter a specific case
for(p in seq(1:9)){
      gg1 <- (ggplot(educ_data_2 %>% filter(privacy==p), 
                     aes(x = class, y= sum, fill = cn), xlab="Column Class") +
                geom_bar(stat="identity", width=.5, position = "dodge") + 
                theme(axis.text.x = element_text(angle = 90, hjust = 1))+ggtitle(paste("Reconstruct Sanitization RMSE w/privacy",p)))
      print(gg1)
      ggsave(paste('/home/juanzinser/Workspace/Tesis/entregas/plots/reconstruction_rmse_',p,'.png',sep=""))
}

