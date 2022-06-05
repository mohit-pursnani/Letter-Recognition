library(randomForest)
library(MASS)
library(dplyr)
library(caret)
library(tidyverse)
library(Matrix)
library(xgboost)
library(gbm)
library(class) # KNN
library(e1071) # SVM
library(doParallel)
library(rpart)
library(adabag)


###############################################################################
#                                 EDA                                         #
###############################################################################

datasetFile = "letter-recognition.data"
dataset = read.csv(datasetFile, header=FALSE, sep=",")
# Change the dependent variable column name to Letter
colnames(dataset)[1] <- "Letter"

# convert letters to number
dataset$Letter = factor(dataset$Letter,
                        levels = c("A", "B","C","D","E","F","G","H","I","J","K","L","M","N","O","P",
                                   "Q","R","S","T","U","V","W","X","Y","Z"),
                        labels = c(1, 2, 3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26))

dataFeatureVariable = dataset[,-which(names(dataset)=='Letter')]

# check for outliers
outliers_row=c() # Loop over the feature columns
for(i in colnames(dataFeatureVariable)) {
  data_mean=mean(dataset[,i]) # Mean of the data in feature column i
  data_sd=sd(dataset[,i]) # Standard deviation of the data in feature column i
  low_cutoff=data_mean-3*data_sd # Lower cutoff value
  upper_cutoff=data_mean+3*data_sd # Upper cutoff value
  outliers_idx=which(dataset[,i]<low_cutoff | dataset[,i]>upper_cutoff)
  outliers_row=c(outliers_row,outliers_idx) 
}
outliers_row=unique(outliers_row) # Remove duplicated row indices
print(paste("Number of Outilers =",length(outliers_row)))

dataset_without_outliers <- dataset[-c(outliers_row),]

set.seed(43)
randomized__withoutOutliers=dataset_without_outliers[sample(1:nrow(dataset_without_outliers),nrow(dataset_without_outliers)),]
tridx_withoutOutliers=sample(1:nrow(dataset_without_outliers),0.7*nrow(dataset_without_outliers),replace=F)
trainingDataset_withoutOutliers = randomized__withoutOutliers[tridx_withoutOutliers,] 
testingDataset_withoutOutliers = randomized__withoutOutliers[-tridx_withoutOutliers,]

###############################################################################
#                   LDA, QDA, SVM, Tree CV                                    #
###############################################################################
trdf <- trainingDataset_withoutOutliers
tstdf <- testingDataset_withoutOutliers

crossValidationModels=function(df,type,numberFolds,dataType){
  cl<-makePSOCKcluster(5)
  registerDoParallel(cl)
  start.time<-proc.time()
  N<-nrow(df)
  NF=numberFolds
  folds<-split(1:N,cut(1:N, quantile(1:N, probs = seq(0, 1, by =1/NF))))
  ridx<-sample(1:nrow(df),nrow(df),replace=FALSE)
  # For lda
  if(type=="lda"){
    cv_df<-do.call('rbind',lapply(folds,FUN=function(idx,data=trdf[ridx,]) {
      m <- lda(Letter~., data = df[-idx,])
      p <- predict(m, df[idx,], type='response')
      pred_tbl<-table(trdf[idx,c(which(names(trdf)=="Letter"))],p$class, dnn = c('Actual Group','Predicted Group'))
      pred_cfm<-caret::confusionMatrix(pred_tbl)
      list(cfm=pred_cfm) # store the fold, model,cfm
    }))
  }
  # For qda
  else if(type=="qda"){
    cv_df<-do.call('rbind',lapply(folds,FUN=function(idx,data=trdf[ridx,]) {
      m <- qda(Letter~., data = df[-idx,])
      p <- predict(m, df[idx,], type='response')
      pred_tbl<-table(trdf[idx,c(which(names(trdf)=="Letter"))],p$class, dnn = c('Actual Group','Predicted Group'))
      pred_cfm<-caret::confusionMatrix(pred_tbl)
      list(cfm=pred_cfm) # store the fold, model,cfm
    }))
  }
  # For SVM
  else if(type=="svm"){
    cv_df<-do.call('rbind',lapply(folds,FUN=function(idx,data=trdf[ridx,]) {
      m <- svm(Letter~., data = df[-idx,])
      p <- predict(m, df[idx,],type="class")
      pred_tbl<-table(trdf[idx,c(which(names(trdf)=="Letter"))],p, dnn = c('Actual Group','Predicted Group'))
      pred_cfm<-caret::confusionMatrix(pred_tbl)
      list(cfm=pred_cfm) # store the fold, model,cfm
    }))
  }
  # For Tree
  else if(type=="tree"){
    cv_df<-do.call('rbind',lapply(folds,FUN=function(idx,data=df[ridx,]) {
      m <- rpart(Letter~., data = df[-idx,])
      temp_df <- df[, -which(names(trainingDataset_withoutOutliers)=="Letter")]
      p <- predict(m, temp_df[idx,],type="class")
      pred_tbl<-table(df[idx,c(which(names(df)=="Letter"))],p, dnn = c('Actual Group','Predicted Group'))
      pred_cfm<-caret::confusionMatrix(pred_tbl)
      list(cfm=pred_cfm) # store the fold, model,cfm
    }))
  }
  # For KNN
  else if(type=="knn"){
    cv_df<-do.call('rbind',lapply(folds,FUN=function(idx,data=df[ridx,]) {
      trdf_knn=df[-idx, -which(names(df)=="Letter")]
      tstdf_knn=df[idx, -which(names(df)=="Letter")]
      trclass_knn=factor(df[-idx, which(names(df)=="Letter")])
      tstclass_knn=factor(df[idx, which(names(df)=="Letter")])
      knn_pred=knn(trdf_knn,tstdf_knn,trclass_knn, k = 1)
      knn_cfm_tst=confusionMatrix(table(tstclass_knn,knn_pred))
      list(fold=idx,knn_pred=knn_pred,cfm=knn_cfm_tst)
    }))
  }
  else {
    print("type should be ’lda’ ’qda’ 'knn' ’tree’ 'svm'")
    return()
  }
  cv_df<-as.data.frame(cv_df)
  trcv.perf<-as.data.frame(do.call('rbind',lapply(cv_df$cfm,FUN=function(cfm)c(cfm$overall))))
  
  acc_varEstp <- trcv.perf$Accuracy
  mean_varEstp = signif(mean(acc_varEstp),4)
  var_varEstp = signif(var(acc_varEstp),4)
  varEstp = data.frame(mean_varEstp,var_varEstp)
  names(varEstp) = c("Mean of Accuracies","Variance of Accuracies")
  varianceEstimation <- t(varEstp)
  print("varianceEstimation")
  print(varianceEstimation)
  print("Mean")
  if(dataType == "train")
  {
    (cv.tr.perf<-apply(trcv.perf[trcv.perf$AccuracyPValue<0.01,-c(6:7)],2,mean))
    print(cv.tr.perf)
    print("Standard Deviation")
    (cv.tr.perf.sd<-apply(trcv.perf[trcv.perf$AccuracyPValue<0.01,-c(6:7)],2,sd))
    print(cv.tr.perf.sd)
    print("Variance")
    (cv.tr.perf.var<-apply(trcv.perf[trcv.perf$AccuracyPValue<0.01,-c(6:7)],2,var))
    print(cv.tr.perf.var)
  }
  else
  {
    (cv.tr.perf<-apply(trcv.perf[trcv.perf$AccuracyPValue>0.01,-c(6:7)],2,mean))
    print(cv.tr.perf)
    print("Standard Deviation")
    (cv.tr.perf.sd<-apply(trcv.perf[trcv.perf$AccuracyPValue>0.01,-c(6:7)],2,sd))
    print(cv.tr.perf.sd)
    print("Variance")
    (cv.tr.perf.var<-apply(trcv.perf[trcv.perf$AccuracyPValue>0.01,-c(6:7)],2,var))
    print(cv.tr.perf.var)
  }
  confusion_matrix <- cv_df$cfm
  cfm_len <- length(cv_df$cfm)
  for(c in confusion_matrix) {
    confusionMatrixByClass <- c$byClass
    confusion_matrix_sd <- c$byClass
    break
  }
  y <- 0
  z <- c()
  for(column in colnames(confusionMatrixByClass)) {
    for(row in rownames(confusionMatrixByClass)) {
      for (x in confusion_matrix) {
        my_mat <- as.matrix(x$byClass) 
        y <- y + my_mat[row,column]
        z <- c(z,my_mat[row,column])
      }
      confusionMatrixByClass[row,column] <- y/cfm_len
      confusion_matrix_sd[row, column] <- sd(z)
      y <- 0
      z <- c()
    }
  }
  print(confusionMatrixByClass)
  stopCluster(cl)
}

crossValidationModels(trdf, "lda", 20, "train")
crossValidationModels(tstdf, "lda", 20, "test")


crossValidationModels(trdf, "qda", 20, "train")
crossValidationModels(tstdf, "qda", 20, "test")

crossValidationModels(trdf, "tree", 20, "train")
crossValidationModels(tstdf, "tree", 20, "train")

crossValidationModels(trdf, "svm", 20, "train")
crossValidationModels(tstdf, "svm", 20, "test")

crossValidationModels(dataset_without_outliers, "knn", 20, "train")

###############################################################################
#                            Bagging                                          #
###############################################################################
trdf <- trainingDataset_withoutOutliers
tstdf <- testingDataset_withoutOutliers

baggingModels=function(df,type){
  # For lda
  if(type=="lda"){
    runModel<-function(df) {
      lda(Letter~.,data=df[sample(1:nrow(df),nrow(df),replace=T),])
    }
  }
  # For qda
  else if(type=="qda"){
    runModel<-function(df) {
      qda(Letter~.,data=df[sample(1:nrow(df),nrow(df),replace=T),])
    }
  }
  # For SVM
  else if(type=="svm"){
    runModel<-function(df) {
      svm(Letter~.,data=df[sample(1:nrow(df),nrow(df),replace=T),])
    }
    lapplyrunmodel <- function(x) runModel(df)
    models<-lapply(1:5,lapplyrunmodel)
    bagging_preds <- lapply(models,FUN=function(M,D=df[,-c(which(names(df)=="Letter"))])predict(M,D))
    bagging_cfm <-lapply(bagging_preds,FUN=function(P,A=df[[c(which(names(df)=="Letter"))]]){
      pred_tbl<-table(A,P)
      pred_cfm<-caret::confusionMatrix(pred_tbl)
      pred_cfm
    })
  }
  # For Tree
  else if(type=="tree"){
    runModel<-function(df) {
      rpart(Letter~.,data=df[sample(1:nrow(df),nrow(df),replace=T),])
    }
    lapplyrunmodel <- function(x) runModel(df)
    models<-lapply(1:100,lapplyrunmodel)
    bagging_preds <- lapply(models,FUN=function(M,D=df[,-c(which(names(df)=="Letter"))])predict(M,D,type="class"))
    bagging_cfm <-lapply(bagging_preds,FUN=function(P,A=df[[c(which(names(df)=="Letter"))]]){
      pred_tbl<-table(A,P)
      pred_cfm<-caret::confusionMatrix(pred_tbl)
      pred_cfm
    })
  }
  else {
    print("type should be ’lda’ ’qda’ ’tree’ 'svm'")
    return()
  }
  
  if(type !="tree" && type !="svm")
  {
    lapplyrunmodel <- function(x) runModel(df)
    models<-lapply(1:100,lapplyrunmodel)
    bagging_preds <- lapply(models,FUN=
                              function(M,D=df[,-c(which(names(df)=="Letter"))])
                                predict(M,D))
    bagging_cfm <-lapply(bagging_preds,FUN=function(P,A=df[[c(which(names(df)=="Letter"))]]){
      pred_tbl<-table(A,P$class)
      pred_cfm<-caret::confusionMatrix(pred_tbl)
      pred_cfm
    })
  }
  
  bagging.perf<-as.data.frame(do.call('rbind',lapply(bagging_cfm,FUN=function(cfm)c(cfm$overall))))
  acc_varEstp <- bagging.perf$Accuracy
  mean_varEstp = signif(mean(acc_varEstp),4)
  var_varEstp = signif(var(acc_varEstp),4)
  varEstp = data.frame(mean_varEstp,var_varEstp)
  names(varEstp) = c("Mean of Accuracies","Variance of Accuracies")
  varianceEstimation <- t(varEstp)
  print("varianceEstimation")
  print(varianceEstimation)
  
  bagging.perf.mean<-apply(bagging.perf[bagging.perf$AccuracyPValue<0.01,-c(6:7)],2,mean)
  print("Mean")
  print(bagging.perf.mean)
  bagging.perf.std<-apply(bagging.perf[bagging.perf$AccuracyPValue<0.01,-c(6:7)],2,sd)
  print("Standard Deviation")
  print(bagging.perf.std)
  bagging.perf.var<-apply(bagging.perf[bagging.perf$AccuracyPValue<0.01,-c(6:7)],2,var)
  print("Variance")
  print(bagging.perf.var)
  for(c in bagging_cfm) {
    confusionMatrixByClass <- c$byClass
    confusion_matrix_sd <- c$byClass
    break
  }
  y <- 0
  z <- c()
  cfm_len <- length(bagging_cfm)
  for(column in colnames(confusionMatrixByClass)) {
    for(row in rownames(confusionMatrixByClass)) {
      for (x in bagging_cfm) {
        my_mat <- as.matrix(x$byClass) 
        y <- y + my_mat[row,column]
        z <- c(z,my_mat[row,column])
      }
      confusionMatrixByClass[row,column] <- y/cfm_len
      confusion_matrix_sd[row, column] <- sd(z)
      y <- 0
      z <- c()
    }
  }
  confusionMatrixByClass
}

baggingModels(trdf, "lda")
baggingModels(tstdf, "lda")

baggingModels(trdf, "qda")
baggingModels(tstdf, "qda")

baggingModels(trdf, "svm")
baggingModels(tstdf, "svm")

baggingModels(trdf, "tree")
baggingModels(tstdf, "tree")

###############################################################################
#                       KNN Bagging                                           #
###############################################################################
trdf <- trainingDataset_withoutOutliers
tstdf <- testingDataset_withoutOutliers

label <- trdf$Letter
trdf_knn = trdf[-which(names(trdf)=="Letter")]
tstdf_knn = tstdf[-which(names(tstdf)=="Letter")]

trclass_knn=factor(trdf[which(names(trdf)=="Letter")])
tstclass_knn=factor(tstdf[which(names(tstdf)=="Letter")])

runModel<-function(df1, df2) {
  knn(df1[-which(names(df1)=="Letter")],df2[-which(names(df2)=="Letter")],label, k = 1)
}

lapplyrunmodel <- function(x) runModel(trdf, tstdf)
models<-lapply(1:5,lapplyrunmodel)
bagging_cfm <- lapply(models, FUN=function(P) {
  pred_cfm<-caret::confusionMatrix(table(P,tstdf$Letter))
  pred_cfm
})

bagging.perf<-as.data.frame(do.call('rbind',lapply(bagging_cfm,FUN=function(cfm)c(cfm$overall))))

acc_varEstp <- bagging.perf$Accuracy
mean_varEstp = signif(mean(acc_varEstp),4)
var_varEstp = signif(var(acc_varEstp),4)
varEstp = data.frame(mean_varEstp,var_varEstp)
names(varEstp) = c("Mean of Accuracies","Variance of Accuracies")
varianceEstimation <- t(varEstp)
print("varianceEstimation")
print(varianceEstimation)

bagging.perf.mean <-apply(bagging.perf[bagging.perf$AccuracyPValue<0.01,-c(6:7)],2,mean)
print("Mean")
print(bagging.perf.mean)
bagging.perf.std<-apply(bagging.perf[bagging.perf$AccuracyPValue<0.01,-c(6:7)],2,sd)
print("Standard Deviation")
print(bagging.perf.std)
bagging.perf.var<-apply(bagging.perf[bagging.perf$AccuracyPValue<0.01,-c(6:7)],2,sd)
print("Variance")
print(bagging.perf.var)

for(c in bagging_cfm) {
  confusionMatrixByClass <- c$byClass
  confusion_matrix_sd <- c$byClass
  break
}
y <- 0
z <- c()
cfm_len <- length(bagging_cfm)
for(column in colnames(confusionMatrixByClass)) {
  for(row in rownames(confusionMatrixByClass)) {
    for (x in bagging_cfm) {
      my_mat <- as.matrix(x$byClass) 
      y <- y + my_mat[row,column]
      z <- c(z,my_mat[row,column])
    }
    confusionMatrixByClass[row,column] <- y/cfm_len
    confusion_matrix_sd[row, column] <- sd(z)
    y <- 0
    z <- c()
  }
}
confusionMatrixByClass


###############################################################################
#                        Random Forest                                        #
###############################################################################

trdf <- trainingDataset_withoutOutliers
tstdf <- testingDataset_withoutOutliers

rf_model<-randomForest(Letter~.,data=trdf)
rf_pred<-predict(rf_model,tstdf[,-c(which(names(tstdf)=="Letter"))])
rf_mtab<-table(tstdf$Letter,rf_pred)
rf_cmx<-caret::confusionMatrix(rf_mtab)
rf_cmx

(rf_accuracy<-sum(diag(rf_mtab))/sum(rf_mtab))

rf_cmx$overall

rf_cmx$byClass

###############################################################################
#                       Random Forest 50 Trees                                #
###############################################################################

rf50_model<-randomForest(Letter~.,data=trdf,ntree=50)
rf50_pred<-predict(rf50_model,tstdf[,-c(which(names(tstdf)=="Letter"))])
rf50_mtab<-table(tstdf$Letter,rf50_pred)
rf50_cmx<-caret::confusionMatrix(rf50_mtab)
rf50_mtab

(rf50_accuracy<-sum(diag(rf50_mtab))/sum(rf50_mtab))

rf50_cmx$overall

rf50_cmx$byClass

###############################################################################
#                       Random Forest 100 Trees                              #
###############################################################################
# Number of trees - 100
rf100_model<-randomForest(Letter~.,data=trdf,ntree=100)
rf100_pred<-predict(rf100_model,tstdf[,-c(which(names(tstdf)=="Letter"))])
rf100_mtab<-table(tstdf$Letter,rf100_pred)
rf100_cmx<-caret::confusionMatrix(rf100_mtab)
rf100_mtab

(rf100_accuracy<-sum(diag(rf100_mtab))/sum(rf100_mtab))

rf100_cmx$overall

rf100_cmx$byClass


###############################################################################
#                       Random Forest 1000 Trees                              #
###############################################################################
# Number of trees - 1000
rf1000_model<-randomForest(Letter~.,data=trdf,ntree=1000)
rf1000_pred<-predict(rf1000_model,tstdf[,-c(which(names(tstdf)=="Letter"))])
rf1000_mtab<-table(tstdf$Letter,rf1000_pred)
rf1000_cmx<-caret::confusionMatrix(rf1000_mtab)
rf1000_mtab

(rf1000_accuracy<-sum(diag(rf1000_mtab))/sum(rf1000_mtab))

rf1000_cmx$overall

rf1000_cmx$byClass


###############################################################################
#                             Boosting GBM                                    #
###############################################################################
trdf <- trainingDataset_withoutOutliers
tstdf <- testingDataset_withoutOutliers

gbm_table_3<-gbm(Letter~.,data=trdf,
                 distribution="multinomial",
                 n.trees=1000,
                 shrinkage=0.01,
                 interaction.depth=3,
                 n.minobsinnode=10,
                 verbose=T,
                 keep.data=F)
gbm_predict<-predict(gbm_table_3,
                     tstdf[,-which(names(tstdf)=="Letter")],
                     type="response",
                     gbm_table_3$n.trees)

labels = colnames(gbm_predict)[apply(gbm_predict, 1, which.max)]

result = data.frame(tstdf$Letter, labels)

cm = confusionMatrix(tstdf$Letter, as.factor(labels))

cm


