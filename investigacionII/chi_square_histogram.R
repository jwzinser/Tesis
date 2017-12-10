
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
reco_df <- read.csv("../data/census/negative_census_reconstruction_df.csv")
reco_df2 <- read.csv("../data/census/negative_census_reconstruction_rmse_df.csv")
reco_sep <- (reco_df %>% rowwise() %>% mutate(privacy = substr(case,1,1)) %>% 
               mutate(include_real = substr(case,2,2)=="t") %>% 
               mutate(uniform = substr(case, 3,3)=="t") %>% 
               mutate(prob_of_real = substr(case, 4, 10)))

reco <- reco_sep %>% gather(cn, sum, CIS, NIS) # histogram basic 
columns <-unique(reco_df$column)

# use education for the histogram as representative column
educ_data <- (reco %>% filter(column=="education"))

educ_data_2 <- (educ_data %>% mutate(cn=ifelse(cn=="CIS","real",paste("n_",ifelse(include_real,"t","f"),
                                                                     ifelse(uniform,"t","f"),sep=""))) %>% 
                                          filter(!(prob_of_real %in% c("0.2", "0.4", "0.6", "0.8"))))

# plots for privacy level with 5 bars per class
# an histogram must filter a specific case
for(p in seq(1:9)){
  gg1 <- (ggplot(educ_data_2 %>% filter(privacy==p), 
                 aes(x = class, y= sum, fill = cn), xlab="Column Class") +
            geom_bar(stat="identity", width=.5, position = "dodge") + 
            theme(axis.text.x = element_text(angle = 90, hjust = 1))+ggtitle(paste("Reconstruct RMSE Sanitization w/privacy",p)))
  print(gg1)
  ggsave(paste('/home/juanzinser/Workspace/Tesis/entregas/plots/reconstruction_rmse_',p,'.png',sep=""))
}





# chi square example and its comparisson with ANOVA
library(MASS)

test_df <- data_frame(variable=character(), privacy=integer(), rmse = double(), case=character(), df=integer(), xsquared=double(), pvalue=double())
for(p in seq(1:9)){
  educ_pivot <- (educ_data_2 %>% filter(privacy==p) %>% 
                   dplyr::select(class, cn, sum) %>% 
                   unique %>% spread(cn, sum))
  # how to get the intersection of both the real and the negative to put it on a table
  for(cs in c("n_ff", "n_tf", "n_ft", "n_tt")){
    tes <- chisq.test(t(educ_pivot %>% dplyr::select(c(cs, "real"))), correct=FALSE)
    # tes <- fisher.test(educ_pivot %>% dplyr::select(c(cs, "real")))
    rmse_cs <- educ_pivot %>% dplyr::select(!!cs)
    rmse_real <- educ_pivot %>% dplyr::select(real)
    rmse <- sqrt(sum((rmse_real-rmse_cs)^2))
    curr_df <- data_frame(variable="education", privacy=p, rmse=rmse, case=cs,df=tes$parameter, xsquared=tes$statistic, pvalue=tes$p.value)
    test_df <- bind_rows(test_df, curr_df)
  }
}
test_df <- test_df %>% dplyr::select(privacy, case, rmse)
xtable(test_df %>% arrange(rmse) %>% mutate(include_real=ifelse(str_sub(case,3,3)=="t",TRUE,FALSE)) %>% 
         mutate(uniform=ifelse(str_sub(case,4,4)=="t",TRUE,FALSE)) %>% dplyr::select(privacy, include_real, uniform, rmse))

rmse_df <- test_df %>% spread(privacy, rmse)
test_dfr <- test_df %>% mutate(case=ifelse(case=="n_tt","TT",ifelse(case=="n_tf","TF",ifelse(case=="n_ft","FT","FF"))))
gg_ch <- (ggplot(test_dfr, aes(y = factor(case),  x = factor(privacy))) +  
            geom_tile(aes(fill = rmse)) + ggtitle("Histogram RMSE by case and privacy")
            #+ scale_fill_continuous(low = "green", high = "red")
          )
gg_ch
ggsave("plots/tile_hist_rmse.png")

gg_cc <- (ggplot(test_df, aes(y = factor(case),  x = factor(privacy))) + 
            geom_point(aes( size =rmse)) + 
          #scale_color_gradient(low = "yellow", high = "red") +     
          scale_size(range = c(1, 12)) 
          # + theme_bw()
          )
gg_cc


set.seed (78888)
rectheat = sample(c(rnorm (10, 5,1), NA, NA), 150, replace = T)
circlefill =  rectheat*10 + rnorm (length (rectheat), 0, 3)
circlesize = rectheat*1.5 + rnorm (length (rectheat), 0, 3)
myd <- data.frame (rowv = rep (1:10, 15), columnv = rep(1:15, each = 10),
                   rectheat, circlesize, circlefill)


pl1 <- (ggplot(myd, aes(y = factor(rowv),  x = factor(columnv))) +  
          geom_tile(aes(fill = rectheat)) +  
          scale_fill_continuous(low = "blue", high = "green"))


pl1 <- (pl1 + geom_point(aes(colour = circlefill,  size =circlesize)) + 
          scale_color_gradient(low = "yellow", high = "red") +     
          scale_size(range = c(1, 20)) +   theme_bw())
pl1

