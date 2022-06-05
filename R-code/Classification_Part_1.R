library(MASS) 
library(dplyr)
library(tidyverse)
library(caret)
library(pROC)
library(e1071) # SVM
library(rpart) #Tree
library(class) # KNN
library(tree) # Tree
library(rpart.plot) #Tree Plot

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
#                                   LDA                                       #
###############################################################################

# Fit the model - Learning Phase
lda.model.withoutOutliers <- lda(Letter~., data = trainingDataset_withoutOutliers)
# Make predictions for training data set
predictions.withoutOutliers <- lda.model.withoutOutliers %>% predict(trainingDataset_withoutOutliers)
# Train rate - Learning
lda.withoutOutliers.rate<- mean(predictions.withoutOutliers$class == trainingDataset_withoutOutliers$Letter)

print(paste("LDA Train Rate =",round(lda.withoutOutliers.rate*100, 4)))


# Learnign Phase - Confusion Matrix
lda_table_training <- table(list(predicted=predictions.withoutOutliers$class, observed=trainingDataset_withoutOutliers$Letter))
lda_cfm_training <- confusionMatrix(lda_table_training)

print("Learning Phase Confusion Matrix")
lda_cfm_training

# Accuracy of predictions with train data
lda_acc_tr=round(lda_cfm_training$overall[["Accuracy"]],4)
print(paste("Classification accuracy of learning phase =",lda_acc_tr*100))

##LDA - Generalization Phase
lda.model.test.withoutOutliers <- lda(Letter~., data = testingDataset_withoutOutliers)
# Make predictions for Test data set
predictions.test.withoutOutliers <- lda.model.test.withoutOutliers %>% predict(testingDataset_withoutOutliers)
# Test error
lda.withoutOutliers.test.error <- mean(predictions.test.withoutOutliers$class == testingDataset_withoutOutliers$Letter)

print(paste("LDA Test Rate =",round(lda.withoutOutliers.test.error*100, 4)))

# Confusion Matrix
lda_table_testing <- table(list(predicted=predictions.test.withoutOutliers$class, observed=testingDataset_withoutOutliers$Letter))
lda_cfm_testing <- confusionMatrix(lda_table_testing)
print("Generalization Phase Confusion Matrix")
lda_cfm_testing

# Accuracy of predictions with test data
lda_acc_tst=round(lda_cfm_testing$overall[["Accuracy"]],4)

print(paste("Classification accuracy of Generalization phase =",lda_acc_tst*100))

## LDA Checking for over fitting
# Check for over-fitting. Criteria: Accuracy change from train to test > 25%
lda_model_isOF=abs((lda_acc_tr-lda_acc_tst)/lda_acc_tr)
if(lda_model_isOF>0.25) print("Model is over-fitting") else print("Model is not over-fitting")

lda_model_isOF=round(lda_model_isOF,4)
print(paste("Accuracy drop from training data to test data is",lda_model_isOF*100,"%"))

### LDA Model Performance Metrics
print("LDA Learning-Phase Performance Parameters:")
lda_PM_tr= lda_cfm_training$byClass[,c("Balanced Accuracy", "Precision", "Sensitivity", "Specificity", "Recall")]
lda_PMavg_tr=round(apply(lda_PM_tr,2,mean),4)
print("Macro Averages:")
t(lda_PMavg_tr)

lda_prob_tr = predict(lda.model.withoutOutliers, trainingDataset_withoutOutliers[,-which(names(trainingDataset_withoutOutliers)=="Letter")], type="prob")
lda_AUC_tr = multiclass.roc(trainingDataset_withoutOutliers[,which(names(trainingDataset_withoutOutliers)=="Letter")],lda_prob_tr$posterior, percent=TRUE)
print(paste("LAD Learning-Phase AUC:",round(lda_AUC_tr$auc,4)))

lda_ROC_tr <- lda_AUC_tr[['rocs']]
ROC_num=paste("1/",as.character(2),sep="")
par(pty = "s")
plot.roc(lda_ROC_tr[[ROC_num]][[2]], col=2, main="ROC curve(Learning Phase)",
         legacy.axes=TRUE, xlab="False Positive Percentage(1-Specificity)", ylab="True Positive Percentage(Sensitivity)",lwd=4)
for(i in 3:10) {
  ROC_num=paste("1/",as.character(i),sep="")
  lines.roc(lda_ROC_tr[[ROC_num]][[2]],col=i)
}
legend("bottomright", legend=c('1/2','1/3','1/4','1/5','1/6','1/7','1/8','1/9','1/10'), col=2:10, lwd=2)

##LDA - Generalization Phase
print("LDA Generalization-Phase Performance Parameters:")
lda_PM_tst= lda_cfm_testing$byClass[,c("Balanced Accuracy", "Precision", "Sensitivity", "Specificity", "Recall")]

lda_PMavg_tst=round(apply(lda_PM_tst,2,mean),4)

print("Macro Averages:")
t(lda_PMavg_tst)

lda_prob_tst = predict(lda.model.test.withoutOutliers, testingDataset_withoutOutliers[,-which(names(testingDataset_withoutOutliers)=="Letter")], type="prob")

lda_AUC_tst = multiclass.roc(testingDataset_withoutOutliers[,which(names(testingDataset_withoutOutliers)=="Letter")],lda_prob_tst$posterior, percent=TRUE)

print(paste("LAD Generalization-Phase AUC:",round(lda_AUC_tst$auc,4)))

lda_ROC_tst <- lda_AUC_tst[['rocs']]
ROC_num=paste("1/",as.character(2),sep="")
par(pty = "s")
plot.roc(lda_ROC_tst[[ROC_num]][[2]], col=2, main="ROC curve(Generalization Phase)",
         legacy.axes=TRUE, xlab="False Positive Percentage(1-Specificity)", ylab="True Positive Percentage(Sensitivity)",lwd=4)

for(i in 3:10) {
  ROC_num=paste("1/",as.character(i),sep="")
  lines.roc(lda_ROC_tst[[ROC_num]][[2]],col=i)
}
legend("bottomright", legend=c('1/2','1/3','1/4','1/5','1/6','1/7','1/8','1/9','1/10'), col=2:10, lwd=2)

print("Generalization Confusion Matrix")
lda_cfm_testing

## Variance Estimation
# Variance Estimation for Learning Phase
varEst_tridx=sample(1:nrow(trainingDataset_withoutOutliers), 0.9*nrow(trainingDataset_withoutOutliers), replace=F)
# Define training data for variance estimation
varEst_trdf=trainingDataset_withoutOutliers[varEst_tridx,] 
varEst_tstdf=trainingDataset_withoutOutliers[-varEst_tridx,]

varEst=function(trdf,tstdf,percent,type){
  target_idx=which(colnames(trdf)=="Letter")
  acc_varEstp=c(); # Initialize a variable to store the accuracies computed in the loop
  for(i in 1:10){
    varEstp_tridx=sample(1:nrow(trdf), percent/100*nrow(trdf), replace=F) # Take samples, percent% of the d
    varEstp_trdf=trdf[varEstp_tridx,]
    # For lda
    if(type=="lda"){
      lda_model_varEstp= lda(Letter~., data = varEstp_trdf) #Train a Logistic model
      pred_varEstp= predict(lda_model_varEstp, tstdf[,-target_idx]) # Predict with varia
      pred_varEstp = pred_varEstp$class
    }
    # For qda
    else if(type=="qda"){
      lda_model_varEstp= qda(Letter~., data = varEstp_trdf) #Train a Logistic model
      pred_varEstp= predict(lda_model_varEstp, tstdf[,-target_idx]) # Predict with varia
      pred_varEstp = pred_varEstp$class
    }
    # For SVM
    else if(type=="svm"){
      svm_model_varEstp = svm(Letter~., varEstp_trdf)
      pred_varEstp = predict(svm_model_varEstp, tstdf[,-target_idx])
    }
    # For Tree
    else if(type=="tree"){
      tree_model_varEstp = rpart(Letter~., varEstp_trdf, method = "class")
      pred_varEstp = predict(tree_model_varEstp, tstdf[,-target_idx], type="class")
    }
    # For KNN
    else if(type=="knn"){
      trclass=factor(varEstp_trdf[,target_idx])
      tstclass=factor(tstdf[,target_idx])
      pred_varEstp=knn(varEstp_trdf[,-target_idx], tstdf[,-target_idx], trclass, k = 15, prob=TRUE)
    }
    else {
      print("type should be ’lda’ ’qda’ ’tree’ 'svm' or ’knn’")
      return()
    }
    u_varEstp=union(pred_varEstp, tstdf[,target_idx]) # Avoids issues when number of classes are not equal
    t_varEstp=table(factor(pred_varEstp, u_varEstp), factor(tstdf[,target_idx],u_varEstp))
    mn_cfm_varEstp=confusionMatrix(t_varEstp) # Confusion Matrix
    mn_acc_varEstp=mn_cfm_varEstp$overall[["Accuracy"]] # Accuracy of predictions
    acc_varEstp=c(acc_varEstp,mn_acc_varEstp) # Store
  }
  mean_varEstp = signif(mean(acc_varEstp),4)
  var_varEstp = signif(var(acc_varEstp),4)
  varEstp = data.frame(mean_varEstp,var_varEstp)
  names(varEstp) = c("Mean of Accuracies","Variance of Accuracies")
  return(t(varEstp))
}

# Variance estimation using 30% of the data
lda_varEst30=varEst(varEst_trdf, varEst_tstdf, 30, type="lda") 
# Variance estimation using 60% of the data
lda_varEst60=varEst(varEst_trdf, varEst_tstdf, 60, type="lda") 
# Variance estimation using 100% of the data
lda_varEst100=varEst(varEst_trdf, varEst_tstdf, 100, type="lda") 

print("LDA Variance Estimation using 30% of data:")
lda_varEst30

print("LDA Variance Estimation using 60% of data:")
lda_varEst60

print("LDA Variance Estimation using 100% of data:")
lda_varEst100

###############################################################################
#                     LDA with Correlated Variables removed                   #
###############################################################################

dataset_without_outliers <- dataset[-c(outliers_row),]
dataWithoutCorrelated_1 = dataset_without_outliers

# Understanding of correlated variables was based on Pearson’s Coefficient and based on the code in EDA
dataWithoutCorrelated_1 = dataWithoutCorrelated_1[,-which(names(dataWithoutCorrelated_1)=="V3")]
dataWithoutCorrelated_1 = dataWithoutCorrelated_1[,-which(names(dataWithoutCorrelated_1)=="V4")]
dataWithoutCorrelated_1 = dataWithoutCorrelated_1[,-which(names(dataWithoutCorrelated_1)=="V5")]
dataWithoutCorrelated_1 = dataWithoutCorrelated_1[,-which(names(dataWithoutCorrelated_1)=="V6")]
dataWithoutCorrelated_1 = dataWithoutCorrelated_1[,-which(names(dataWithoutCorrelated_1)=="V12")]

randomized__withoutCorrelation=dataWithoutCorrelated_1[sample(1:nrow(dataWithoutCorrelated_1),nrow(dataWithoutCorrelated_1)),]
tridx_withoutCorrelation = sample(1:nrow(dataWithoutCorrelated_1),0.7*nrow(dataWithoutCorrelated_1),replace=F)
trainingDataset_withoutCorrelation = randomized__withoutCorrelation[tridx_withoutCorrelation,] 
testingDataset_withoutCorrelation = randomized__withoutCorrelation[-tridx_withoutCorrelation,]


# Fit the model - Learning Phase
lda.model.withoutOutliers <- lda(Letter~., data = trainingDataset_withoutCorrelation)
# Make predictions for training data set
predictions.withoutOutliers <- lda.model.withoutOutliers %>% predict(trainingDataset_withoutCorrelation)
# Train error
lda.withoutOutliers.error <- mean(predictions.withoutOutliers$class == trainingDataset_withoutCorrelation$Letter)

print(paste("LDA Train Rate =",round(lda.withoutOutliers.error*100, 4)))

# Confusion Matrix
lda_table_training <- table(list(predicted=predictions.withoutOutliers$class, observed=trainingDataset_withoutCorrelation$Letter))

print("LDA Learning Phase Confusion Matrix")
lda_cfm_training <- confusionMatrix(lda_table_training)


print("Learning Phase Confusion Matrix")
lda_cfm_training

# Accuracy of predictions with train data
lda_acc_tr=round(lda_cfm_training$overall[["Accuracy"]],4)

print(paste("Classification accuracy of learning phase =",lda_acc_tr))

##LDA - Generalization Phase
lda.model.test.withoutOutliers <- lda(Letter~., data = testingDataset_withoutCorrelation)
# Make predictions for training data set
predictions.test.withoutOutliers <- lda.model.test.withoutOutliers %>% predict(testingDataset_withoutCorrelation)
# Train error
lda.withoutOutliers.test.error <- mean(predictions.test.withoutOutliers$class == testingDataset_withoutCorrelation$Letter)

print(paste("LDA Test Rate =",round(lda.withoutOutliers.test.error*100, 4)))
# Confusion Matrix
lda_table_testing <- table(list(predicted=predictions.test.withoutOutliers$class, observed=testingDataset_withoutCorrelation$Letter))

print("LDA Generalization Phase Confusion Matrix")
lda_cfm_testing <- confusionMatrix(lda_table_testing)

print("Generalization Phase Confusion Matrix")
lda_cfm_testing

# Accuracy of predictions with train data
lda_acc_tst=round(lda_cfm_testing$overall[["Accuracy"]],4)

print(paste("Classification accuracy of Generalization phase =",lda_acc_tst))

###############################################################################
#                     LDA Predictor Interaction                               #
###############################################################################
# New fit with predictors interaction
# this step takes time
lda2.fit <- lda(Letter~.*.+.:.:., trainingDataset_withoutOutliers)
#  Make prediction
lda2.train.pred <- predict(lda2.fit, trainingDataset_withoutOutliers)
lda2.train.rate <- mean(lda2.train.pred$class == trainingDataset_withoutOutliers$Letter)
print(paste("LDA Predictor Interaction Learning Phase Train Rate: ",round(lda2.train.rate*100,4)))

# Confusion Matrix
lda_table_training <- table(list(predicted=lda2.train.pred$class, observed=trainingDataset_withoutOutliers$Letter))
lda_cfm_training <- confusionMatrix(lda_table_training)
print("Learning Phase Confusion Matrix")
lda_cfm_training

# Accuracy of predictions with test data
lda_acc_train=round(lda_cfm_training$overall[["Accuracy"]],4)
print(paste("Classification accuracy of Generalization phase =",lda_acc_train*100))

#  Make prediction
lda2.test.pred <- predict(lda2.fit, testingDataset_withoutOutliers)
lda2.test.class <- lda2.test.pred$class
# Test error
lda2.test.rate <- mean(lda2.test.class == testingDataset_withoutOutliers$Letter)
print(paste("LDA Predictor Interaction Generalization Phase Test Rate: ",round(lda2.test.rate*100, 4)))

# Generalization Phase Confusion Matrix
lda_table_test <- table(list(predicted=lda2.test.class, observed=testingDataset_withoutOutliers$Letter))
lda_cfm_test <- confusionMatrix(lda_table_test)
print("Generalization Phase Confusion Matrix")
lda_cfm_test


## LDA Checking for over fitting
# Check for over-fitting. Criteria: Accuracy change from train to test > 25%
lda_model_isOF=abs((lda2.train.rate-lda2.test.rate)/lda2.train.rate)
if(lda_model_isOF>0.25) print("Model is over-fitting") else print("Model is not over-fitting")

# Accuracy of predictions with test data
lda2_acc_test=round(lda_cfm_test$overall[["Accuracy"]],4)
print(paste("Classification accuracy of Generalization phase =",round(lda2_acc_test*100, 4)))

lda2_AUC_train = multiclass.roc(trainingDataset_withoutOutliers[,which(names(trainingDataset_withoutOutliers)=="Letter")],lda2.train.pred$posterior, percent=TRUE)
print(paste("LDA Learning-Phase AUC:",round(lda2_AUC_train$auc,4)))

lda2_ROC_train <- lda2_AUC_train[['rocs']]
ROC_num=paste("1/",as.character(2),sep="")
par(pty = "s")
plot.roc(lda2_ROC_train[[ROC_num]][[2]], col=2, main="ROC curve(Learning Phase)",
         legacy.axes=TRUE, xlab="False Positive Percentage(1-Specificity)", ylab="True Positive Percentage(Sensitivity)",lwd=4)

for(i in 3:10) {
  ROC_num=paste("1/",as.character(i),sep="")
  lines.roc(lda2_ROC_train[[ROC_num]][[2]],col=i)
}
legend("bottomright", legend=c('1/2','1/3','1/4','1/5','1/6','1/7','1/8','1/9','1/10'), col=2:10, lwd=2)


lda2_AUC_test = multiclass.roc(testingDataset_withoutOutliers[,which(names(testingDataset_withoutOutliers)=="Letter")],lda2.test.pred$posterior, percent=TRUE)
print(paste("LDA Generalization-Phase AUC:",round(lda2_AUC_test$auc,4)))

lda2_ROC_test <- lda2_AUC_test[['rocs']]
ROC_num=paste("1/",as.character(2),sep="")
par(pty = "s")
plot.roc(lda2_ROC_test[[ROC_num]][[2]], col=2, main="ROC curve(Generalization Phase)",
         legacy.axes=TRUE, xlab="False Positive Percentage(1-Specificity)", ylab="True Positive Percentage(Sensitivity)",lwd=4)

for(i in 3:10) {
  ROC_num=paste("1/",as.character(i),sep="")
  lines.roc(lda2_ROC_test[[ROC_num]][[2]],col=i)
}
legend("bottomright", legend=c('1/2','1/3','1/4','1/5','1/6','1/7','1/8','1/9','1/10'), col=2:10, lwd=2)

###############################################################################
#                                   QDA                                       #
###############################################################################
# Fit the model - Learning Phase
qda.model.withoutOutliers <- qda(Letter~., data = trainingDataset_withoutOutliers)
# Make predictions for training data set
qda.predictions.withoutOutliers <- qda.model.withoutOutliers %>% predict(trainingDataset_withoutOutliers)
# Train Rate
qda.withoutOutliers.rate <- mean(qda.predictions.withoutOutliers$class == trainingDataset_withoutOutliers$Letter)
print(paste("QDA Train Rate: ",round(qda.withoutOutliers.rate*100,4)))

# Confusion Matrix
qda_table_training <- table(list(predicted=qda.predictions.withoutOutliers$class, observed=trainingDataset_withoutOutliers$Letter))
qda_cfm_training <- confusionMatrix(qda_table_training)
print("Learning Phase Confusion Matrix")
qda_cfm_training

# Accuracy of predictions with train data
qda_acc_tr=round(qda_cfm_training$overall[["Accuracy"]],4)
print(paste("Classification accuracy of learning phase =",qda_acc_tr))

##QDA - Generalization Phase
qda.model.test.withoutOutliers <- qda(Letter~., data = testingDataset_withoutOutliers)
# Make predictions for training data set
qda.predictions.test.withoutOutliers <- qda.model.test.withoutOutliers %>% predict(testingDataset_withoutOutliers)
# Train error
qda.withoutOutliers.test.error <- mean(qda.predictions.test.withoutOutliers$class == testingDataset_withoutOutliers$Letter)
print(paste("QDA Test Rate: ",round(qda.withoutOutliers.test.error*100,4)))

# Confusion Matrix
qda_table_testing <- table(list(predicted=qda.predictions.test.withoutOutliers$class, observed=testingDataset_withoutOutliers$Letter))
qda_cfm_testing <- confusionMatrix(qda_table_testing)
print("QDA Generalization Phase Confusion Matrix")
qda_cfm_testing

# Accuracy of predictions with train data
qda_acc_tst=round(qda_cfm_testing$overall[["Accuracy"]],4)
print(paste("Classification accuracy of Generalization phase =",qda_acc_tst))

## QDA Checking for over fitting
# Check for over-fitting. Criteria: Accuracy change from train to test > 25%
qda_model_isOF=abs((qda_acc_tr-qda_acc_tst)/qda_acc_tr)
qda_model_isOF=round(qda_model_isOF,4)
print(paste("Accuracy drop from training data to test data is",qda_model_isOF*100,"%"))
if(lda_model_isOF>0.25) print("Model is over-fitting") else print("Model is not over-fitting")


### QDA Model Performance Metrics
print("QDA Learning-Phase Performance Parameters:")
qda_PM_tr= qda_cfm_training$byClass[,c("Balanced Accuracy", "Precision", "Sensitivity", "Specificity", "Recall")]

qda_PMavg_tr=round(apply(qda_PM_tr,2,mean),4)

print("Macro Averages:")
t(qda_PMavg_tr)

qda_prob_tr = predict(qda.model.withoutOutliers, trainingDataset_withoutOutliers[,-which(names(trainingDataset_withoutOutliers)=="Letter")], type="prob")
qda_AUC_tr = multiclass.roc(trainingDataset_withoutOutliers[,which(names(trainingDataset_withoutOutliers)=="Letter")],qda_prob_tr$posterior, percent=TRUE)
print(paste("QDA Learning-Phase AUC:",round(qda_AUC_tr$auc,4)))

qda_ROC_tr <- qda_AUC_tr[['rocs']]
ROC_num=paste("1/",as.character(2),sep="")
par(pty = "s")
plot.roc(qda_ROC_tr[[ROC_num]][[2]], col=2, main="ROC curve(Learning Phase)",
         legacy.axes=TRUE, xlab="False Positive Percentage(1-Specificity)", ylab="True Positive Percentage(Sensitivity)",lwd=4)
for(i in 3:10) {
  ROC_num=paste("1/",as.character(i),sep="")
  lines.roc(qda_ROC_tr[[ROC_num]][[2]],col=i)
}
legend("bottomright", legend=c('1/2','1/3','1/4','1/5','1/6','1/7','1/8','1/9','1/10'), col=2:10, lwd=2)

##QDA - Generalization Phase
print("QDA Generalization-Phase Performance Parameters:")
qda_PM_tst= qda_cfm_testing$byClass[,c("Balanced Accuracy", "Precision", "Sensitivity", "Specificity", "Recall")]
qda_PMavg_tst=round(apply(qda_PM_tst,2,mean),4)
print("Macro Averages:")
t(qda_PMavg_tst)

qda_prob_tst = predict(qda.model.test.withoutOutliers, testingDataset_withoutOutliers[,-which(names(testingDataset_withoutOutliers)=="Letter")], type="prob")
qda_AUC_tst = multiclass.roc(testingDataset_withoutOutliers[,which(names(testingDataset_withoutOutliers)=="Letter")],qda_prob_tst$posterior, percent=TRUE)
print(paste("QDA Generalization-Phase AUC:",round(qda_AUC_tst$auc,4)))

qda_ROC_tst <- qda_AUC_tst[['rocs']]
ROC_num=paste("1/",as.character(2),sep="")
par(pty = "s")
plot.roc(qda_ROC_tst[[ROC_num]][[2]], col=2, main="ROC curve(Generalization Phase)",
         legacy.axes=TRUE, xlab="False Positive Percentage(1-Specificity)", ylab="True Positive Percentage(Sensitivity)",lwd=4)
for(i in 3:10) {
  ROC_num=paste("1/",as.character(i),sep="")
  lines.roc(qda_ROC_tst[[ROC_num]][[2]],col=i)
}
legend("bottomright", legend=c('1/2','1/3','1/4','1/5','1/6','1/7','1/8','1/9','1/10'), col=2:10, lwd=2)

qda_cfm_testing

## Variance Estimation
# Variance Estimation for Learning Phase
varEst_tridx=sample(1:nrow(trainingDataset_withoutOutliers), 0.9*nrow(trainingDataset_withoutOutliers), replace=F)
# Define training data for variance estimation
varEst_trdf=trainingDataset_withoutOutliers[varEst_tridx,] 
varEst_tstdf=trainingDataset_withoutOutliers[-varEst_tridx,]

# Variance estimation using 30% of the data
qda_varEst30=varEst(varEst_trdf, varEst_tstdf, 30, type="qda") 
# Variance estimation using 60% of the data
qda_varEst60=varEst(varEst_trdf, varEst_tstdf, 60, type="qda") 
# Variance estimation using 100% of the data
qda_varEst100=varEst(varEst_trdf, varEst_tstdf, 100, type="qda") 

print("QDA Variance Estimation using 30% of data:")
qda_varEst30

print("QDA Variance Estimation using 60% of data:")
qda_varEst60

print("QDA Variance Estimation using 100% of data:")
qda_varEst100

 

###############################################################################
#                                   SVM                                       #
###############################################################################
# Fit SVM
svm.fit <- svm(Letter~., trainingDataset_withoutOutliers)
# Predict using train data (Learning Phase)
svm_pred_tr=predict(svm.fit, trainingDataset_withoutOutliers[, -which(names(trainingDataset_withoutOutliers)=="Letter")])

svm.train.rate <- mean(svm_pred_tr == trainingDataset_withoutOutliers$Letter)
print(paste("SVM Train Rate: ",round(svm.train.rate*100,4)))

# Confusion Matrix for train data
svm_cfm_tr=confusionMatrix(table(trainingDataset_withoutOutliers$Letter,svm_pred_tr))
print("SVM Learning Phase Confusion Matrix")
svm_cfm_tr

svm_acc_tr=round(svm_cfm_tr$overall[["Accuracy"]],4)
print(paste("SVM Learning Phase Accuracy =",round(svm_acc_tr*100,4)))

### SVM Generalization Phase
# Predict using test data (Generalization Phase)
svm_pred_tst=predict(svm.fit, testingDataset_withoutOutliers[, -which(names(testingDataset_withoutOutliers)=="Letter")])
svm.pred.test.rate <- mean(svm_pred_tst == testingDataset_withoutOutliers$Letter)

print(paste("SVM Test Rate: ",round(svm.pred.test.rate*100,4)))

# Confusion Matrix for test data
svm_cfm_testing=confusionMatrix(table(testingDataset_withoutOutliers$Letter,svm_pred_tst))
print("SVM Learning Phase Confusion Matrix")
svm_cfm_testing

# Accuracy
svm_acc_tst=round(svm_cfm_testing$overall[["Accuracy"]],4) 
print(paste("SVM Generalization Phase Accuracy =",round(svm_acc_tst*100,4)))

# Check for over-fitting. Criteria: Accuracy change from train to test > 25%
svm_model_isOF=abs((svm_acc_tr-svm_acc_tst)/svm_acc_tr)
svm_model_isOF=round(svm_model_isOF,4)
print(paste("Accuracy drop from training data to test data is",svm_model_isOF*100,"%"))
if(svm_model_isOF>0.25) print("Model is over-fitting") else print("Model is not over-fitting")

###############################################################################
#                    SVM Model Performance Metrics                            #
###############################################################################
svm_PM_tr = svm_cfm_tr$byClass[, c("Balanced Accuracy", "Precision", "Sensitivity", "Specificity", "Recall")]
print("SVM Learning-Phase Performance Parameters:")
svm_PM_tr

svm_PM_avg_tr=round(apply(svm_PM_tr,2,mean),4)
print("Macro Averages:")
t(svm_PM_avg_tr)

svm_prob_tr = predict(svm.fit, 
                      trainingDataset_withoutOutliers[,-which(names(trainingDataset_withoutOutliers)=="Letter")], )
svm_AUC_tr= multiclass.roc(trainingDataset_withoutOutliers$Letter, as.numeric(svm_prob_tr))
print(paste("SVM Learning-Phase AUC:", round(svm_AUC_tr$auc, 4)))

svm_AUC_tr <- svm_AUC_tr$rocs
plot.roc(svm_AUC_tr[[1]], col=2, main="ROC curve(Learning Phase)",
         legacy.axes=TRUE,lwd=4)
for(i in 2:10){
  lines.roc(svm_AUC_tr[[i]],col=i)
}
legend("bottomright", legend=c('1', '2','3','4','5','6','7','8','9','10'), col=2:10, lwd=2)


##### SVM generalization-Phase Performance Matrics
svm_PM_tst=svm_cfm_testing$byClass[,c("Balanced Accuracy", "Precision", "Sensitivity", "Specificity", "Recall")]
print("SVM Generalization-Phase Performance Parameters:")
svm_PM_tst
svm_PM_avg_tst=round(apply(svm_PM_tst,2,mean),4)
svm_PM_avg_tst

svm_prob_tst = predict(svm.fit, 
                       testingDataset_withoutOutliers[,-which(names(testingDataset_withoutOutliers)=="Letter")], )
svm_AUC_tst= multiclass.roc(testingDataset_withoutOutliers$Letter, as.numeric(svm_prob_tst))
print(paste("SVM Generalization-Phase AUC:", round(svm_AUC_tst$auc, 4)))
svm_AUC_tst <- svm_AUC_tst$rocs
plot.roc(svm_AUC_tst[[1]])
for(i in 2:10){
  lines.roc(svm_AUC_tst[[i]],col=i)
}
legend("bottomright", legend=c('1', '2','3','4','5','6','7','8','9','10'), col=2:10, lwd=2)

svm_cfm_testing

svm_varEst30 = varEst(varEst_trdf, varEst_tstdf, 30, type="svm")
svm_varEst60 = varEst(varEst_trdf, varEst_tstdf, 60, type="svm")
svm_varEst100 = varEst(varEst_trdf, varEst_tstdf, 100, type="svm")

print("SVM Variance Estimation using 30% of data:")
svm_varEst30

print("SVM Variance Estimation using 60% of data:")
svm_varEst60

print("SVM Variance Estimation using 100% of data:")
svm_varEst100

###############################################################################
#               SVM Without Correlated Variables                              #
###############################################################################
# Fit SVM
svm.withoutCorrelation.fit <- svm(Letter~., trainingDataset_withoutCorrelation)
# Predict using train data (Learning Phase)
svm_pred_withoutcorrelation_tr=predict(svm.withoutCorrelation.fit, trainingDataset_withoutCorrelation[, -which(names(trainingDataset_withoutCorrelation)=="Letter")])
svm.train.rate <- mean(svm_pred_withoutcorrelation_tr == trainingDataset_withoutCorrelation$Letter)
print(paste("SVM Learning Phase: ",round(svm.train.rate*100, 4)))

# Fit SVM
svm.withoutCorrelation.fit.test <- svm(Letter~., testingDataset_withoutCorrelation)
# Predict using train data (Generalization Phase)
svm_pred_withoutcorrelation_test=predict(svm.withoutCorrelation.fit.test, testingDataset_withoutCorrelation[, -which(names(trainingDataset_withoutCorrelation)=="Letter")])
svm.test.rate <- mean(svm_pred_withoutcorrelation_test == testingDataset_withoutCorrelation$Letter)
print(paste("SVM Generalization Phase: ",round(svm.test.rate*100, 4)))


# Confusion Matrix for train data
svm_cfm2_tr=confusionMatrix(table(trainingDataset_withoutCorrelation$Letter,svm_pred_withoutcorrelation_tr))
svm_acc2_tr=round(svm_cfm2_tr$overall[["Accuracy"]],4)
print(paste("SVM Learning Phase Accuracy =",round(svm_acc2_tr*100,4)))

# Confusion Matrix for test data
svm_cfm2_tst=confusionMatrix(table(testingDataset_withoutCorrelation$Letter,svm_pred_withoutcorrelation_test))
svm_acc2_tst=round(svm_cfm2_tst$overall[["Accuracy"]],4)
print(paste("SVM Generalization Phase Accuracy =",round(svm_acc2_tst*100,4)))

svm_model2_isOF=abs((svm_acc2_tr-svm_acc2_tst)/svm_acc2_tr)
svm_model2_isOF=round(svm_model2_isOF,4)
print(paste("Accuracy drop from training data to test data is",svm_model2_isOF*100,"%"))
if(svm_model2_isOF>0.25) print("Model is over-fitting") else print("Model is not over-fitting")



###############################################################################
#                                  TREE                                       #
###############################################################################

# Calculations can become complex when there are many class labels
#Generally, it gives low prediction accuracy for a dataset as compared to other machine learning algorithms.
# Decision-tree learners can create over-complex trees that do not generalize the data well
#The problem of learning an optimal decision tree is known to be 
#NP-complete under several aspects of optimality and even for simple concepts. 

tree_model = tree(Letter~., data = trainingDataset_withoutOutliers)
summary(tree_model)
# Make prediction
tree_pred_tr=predict(tree_model, trainingDataset_withoutOutliers[, -which(names(trainingDataset_withoutOutliers)=="Letter")], type="class")

plot(tree_model)
text(tree_model,pretty=0)

# Train rate
tree.train.rate <- mean(tree_pred_tr == trainingDataset_withoutOutliers$Letter)
print(paste("Tree Train Rate: ",round(tree.train.rate*100,4)))

# Using rpart (more verbosity)
tree_model1 = rpart(Letter~., data = trainingDataset_withoutOutliers, method = 'class')
rpart.plot(tree_model1, box.palette = "azure2")

printcp(tree_model1)

tree_pred1_tr=predict(tree_model1, trainingDataset_withoutOutliers[, -which(names(trainingDataset_withoutOutliers)=="Letter")],type="class")

tree_cfm_tr=confusionMatrix(table(trainingDataset_withoutOutliers$Letter,tree_pred1_tr))
print("Tree Learning Phase Confusion Matrix")
tree_cfm_tr

tree_acc_tr=round(tree_cfm_tr$overall[["Accuracy"]],4) # Accuracy
print(paste("Tree Learning Phase Accuracy =",round(tree_acc_tr*100, 4)))


###############################################################################
#                        Tree Generalization Phase                            #
###############################################################################

tree_pred1_tst= predict(tree_model1, testingDataset_withoutOutliers[, -which(names(testingDataset_withoutOutliers)=="Letter")], type = 'class')

tree_cfm1_tst=confusionMatrix(table(testingDataset_withoutOutliers[, which(names(testingDataset_withoutOutliers)=="Letter")],tree_pred1_tst)) # Confusion Matrix

print("Tree Generalization-Phase Confusion Matrix")
tree_cfm1_tst

tree_acc1_tst=round(tree_cfm1_tst$overall[["Accuracy"]],4) # Accuracy
print(paste("Tree Generalization-Phase Accuracy =",round(tree_acc1_tst*100, 4)))

## Checking for over-fitting
# Check for over-fitting. Criteria: Accuracy change from train to test > 25%
tree_model1_isOF=abs((tree_acc_tr-tree_acc1_tst)/tree_acc_tr)
tree_model1_isOF=round(tree_model1_isOF,4)
print(paste("Accuracy drop from training data to test data is",tree_model1_isOF*100,"%"))

if(tree_model1_isOF>0.25) print("Model is over-fitting") else print("Model is not over-fitting")

###############################################################################
#                   Tree Model Performance Metrics                            #
###############################################################################
#Tree Learning Phase
tree_PM_tr = tree_cfm_tr$byClass[,c("Balanced Accuracy", "Precision", "Sensitivity", "Specificity", "Recall")]
print("Tree Learning-Phase Performance Parameters:")
tree_PM_tr

# replace null with 0 to calculate average
tree_PM_tr[is.na(tree_PM_tr)] <- 0
tree_PMavg_tr=round(apply(tree_PM_tr,2,mean),4)
tree_PMavg_tr

tree_model2 = rpart(Letter~., data = trainingDataset_withoutOutliers, method = 'class')
tree_class2_tr = predict(tree_model2, trainingDataset_withoutOutliers[, -which(names(trainingDataset_withoutOutliers)=="Letter")], type='class')
tree_AUC2_tr = multiclass.roc(trainingDataset_withoutOutliers$Letter, as.numeric(tree_class2_tr))
print(paste("tree Learning-Phase AUC:",round(tree_AUC2_tr$auc,4)))

tree_AUC2_rocs_tr = tree_AUC2_tr$rocs
plot.roc(tree_AUC2_rocs_tr[[1]])

for(i in 2:10){
  lines.roc(tree_AUC2_rocs_tr[[i]],col=i)
}
legend("bottomright", legend=c('1', '2','3','4','5','6','7','8','9','10'), col=2:10, lwd=2)

## Tree Generalization Phase - AUC & ROC
tree_PM2_tst = tree_cfm1_tst$byClass[,c("Balanced Accuracy", "Precision", "Sensitivity", "Specificity", "Recall")]
tree_test_model2 = rpart(Letter~., data = testingDataset_withoutOutliers, method = 'class')
tree_prob2_tst=predict(tree_test_model2, testingDataset_withoutOutliers[, -which(names(testingDataset_withoutOutliers)=="Letter")], type = 'class')
tree_AUC2_tst=multiclass.roc(testingDataset_withoutOutliers$Letter, as.numeric(tree_prob2_tst), percent = TRUE)
print(paste("tree Generalization-Phase AUC:",round(tree_AUC2_tst$auc,4)))

tree_PM2_tst[is.na(tree_PM2_tst)] <- 0
tree_PMavg_tst=round(apply(tree_PM2_tst,2,mean),4)
tree_PMavg_tst


tree_AUC2_rocs_tst <- tree_AUC2_tst$rocs
plot.roc(tree_AUC2_rocs_tst[[1]])
for(i in 2:10){
  lines.roc(tree_AUC2_rocs_tst[[i]],col=i)
}
legend("bottomright", legend=c('1', '2','3','4','5','6','7','8','9','10'), col=2:10, lwd=2)

## Tree Variance Estimation
tree_varEst30=varEst(varEst_trdf, varEst_tstdf, 30, type="tree") # Variance estimation using 30% of the data
tree_varEst60=varEst(varEst_trdf, varEst_tstdf, 60, type="tree") # Variance estimation using 60% of the data
tree_varEst100=varEst(varEst_trdf, varEst_tstdf, 100, type="tree") # Variance estimation using 100% of the data

print("Tree Variance Estimation using 30% of data:")
tree_varEst30

print("Tree Variance Estimation using 60% of data:")
tree_varEst60

print("Tree Variance Estimation using 100% of data:")
tree_varEst100

###############################################################################
#                                     KNN                                     #
###############################################################################
label <- trainingDataset_withoutOutliers$Letter
# Explore different K values
error.rate <- numeric(10)
for(i in 1:10){
  knn.pred <- knn(trainingDataset_withoutOutliers[, -which(names(trainingDataset_withoutOutliers)=="Letter")],
                  testingDataset_withoutOutliers[, -which(names(testingDataset_withoutOutliers)=="Letter")],label, k = i)
  error.rate[i] <- 1-mean(knn.pred == testingDataset_withoutOutliers$Letter)
}
# Plot error rates
plot(1:10, error.rate,"b", pch = 20, col = "red", xlab = "K", ylab = "Error Rate")

# Make prediction with K = 1
prediction <- knn(trainingDataset_withoutOutliers[, -which(names(trainingDataset_withoutOutliers)=="Letter")],
                  testingDataset_withoutOutliers[, -which(names(testingDataset_withoutOutliers)=="Letter")],label, k = 1)
table(prediction,testingDataset_withoutOutliers$Letter)
knn.test.rate <- mean(prediction==testingDataset_withoutOutliers$Letter)
print(paste("KNN Test Rate: ",round(knn.test.rate*100,4)))

# Same with Cross Validation
knn.cv.pred <- knn.cv(dataset_without_outliers[, -which(names(dataset_without_outliers)=="Letter")], dataset_without_outliers$Letter, k = 1)
knn.cv.test.rate <- mean(knn.cv.pred == dataset_without_outliers$Letter)
print(paste("KNN Test Rate: ",round(knn.cv.test.rate*100,4)))

trdf_knn=trainingDataset_withoutOutliers[, -which(names(trainingDataset_withoutOutliers)=="Letter")]
tstdf_knn=testingDataset_withoutOutliers[, -which(names(testingDataset_withoutOutliers)=="Letter")]
trclass_knn=factor(trainingDataset_withoutOutliers[, which(names(trainingDataset_withoutOutliers)=="Letter")])
tstclass_knn=factor(testingDataset_withoutOutliers[, which(names(testingDataset_withoutOutliers)=="Letter")])
knn_pred=knn(trdf_knn,tstdf_knn,trclass_knn, k = 1, prob=TRUE)
# Predict using test data (Generalization Phase)
# Confusion Matrix for test data
knn_cfm_tst=confusionMatrix(table(tstclass_knn,knn_pred))
knn_cfm_tst

knn_acc_tst=round(knn_cfm_tst$overall[["Accuracy"]],4)
print(paste("kNN Generalization Phase Accuracy =",knn_acc_tst))

###kNN Model Performance Metrics
knn_PM_tst=knn_cfm_tst$byClass[, c("Balanced Accuracy", "Precision", "Sensitivity", "Specificity", "Recall")]
print("kNN Generalization-Phase Performance Parameters:")
knn_PM_tst

knn_prob_tst=attr(knn_pred,"prob")
knn_AUC_tst=multiclass.roc(tstclass_knn, as.ordered(knn_pred))
print(paste("kNN Generalization-Phase AUC:",round(knn_AUC_tst$auc,4)))

# ROC curves
knn_ROC_tst=knn_AUC_tst$rocs
plot.roc(knn_ROC_tst[[1]], col=1)
for(i in 2:10){
  lines.roc(knn_ROC_tst[[i]],col=i)
}
legend("bottomright", legend=c('1', '2','3','4','5','6','7','8','9','10'), col=2:10, lwd=2)

knn_varEst30=varEst(varEst_trdf, varEst_tstdf, 30, type="knn") # 30% of data
knn_varEst60=varEst(varEst_trdf, varEst_tstdf, 60, type="knn") # 60% of data
knn_varEst100=varEst(varEst_trdf, varEst_tstdf, 100, type="knn") # 100% of data

print("kNN Variance Estimation using 30% of data:")
knn_varEst30

print("kNN Variance Estimation using 60% of data:")
knn_varEst60

print("kNN Variance Estimation using 100% of data:")
knn_varEst100

