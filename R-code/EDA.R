library(corrplot)
library(ggplot2)
library(cowplot)

datasetFile = "letter-recognition.data"
dataset = read.csv(datasetFile, header=FALSE, sep=",")

dim(dataset)

# Some rows from the data set
head(dataset)

# View column names in the data set
colnames(dataset)

# Change the dependent variable column name to Letter
colnames(dataset)[1] <- "Letter"
colnames(dataset)

# position of the class variable
which(names(dataset)=='Letter')

# convert to the data set table
datasetTable = table(dataset$Letter)
datasetTable

names(datasetTable)

# check for type of classification
print(ifelse(length(datasetTable)==2, "Binary Classification", "MultiClassClassification"))

# convert letters to number
dataset$Letter = factor(dataset$Letter,
                              levels = c("A", "B","C","D","E","F","G","H","I","J","K","L","M","N","O","P",
                                         "Q","R","S","T","U","V","W","X","Y","Z"),
                              labels = c(1, 2, 3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26))

datasetTable = table(dataset$Letter)
datasetTable
# bar plot to show letter distribution
barplot(datasetTable, main="Letter Distribution", xlab="Letters", ylab="Count of Letters", col=c("darkblue","darkblue"))

# histogram plot for each feature variable
z1 <- ggplot(data=dataset,aes(V2)) + geom_histogram(breaks=seq(0,15,by=1),color="black",fill="red") + labs(x="V2",y="Count")
z2 <- ggplot(data=dataset,aes(V3)) + geom_histogram(breaks=seq(0,15,by=1),color="black",fill="red") + labs(x="V3",y="Count")
z3 <- ggplot(data=dataset,aes(V4)) + geom_histogram(breaks=seq(0,15,by=1),color="black",fill="red") + labs(x="V4",y="Count")
z4 <- ggplot(data=dataset,aes(V5)) + geom_histogram(breaks=seq(0,15,by=1),color="black",fill="red")+ labs(x="V5",y="Count")
plot_grid(z1, z2, z3, z4, ncol = 2, nrow = 2)

z5 <- ggplot(data=dataset,aes(V6)) + geom_histogram(breaks=seq(0,15,by=1),color="black",fill="red") + labs(x="V6",y="Count")
z6 <- ggplot(data=dataset,aes(V7)) + geom_histogram(breaks=seq(0,15,by=1),color="black",fill="red") + labs(x="V7",y="Count")
z7 <- ggplot(data=dataset,aes(V8)) + geom_histogram(breaks=seq(0,15,by=1),color="black",fill="red") + labs(x="V8",y="Count")
z8 <- ggplot(data=dataset,aes(V9)) + geom_histogram(breaks=seq(0,15,by=1),color="black",fill="red")+ labs(x="V9",y="Count")
plot_grid(z5, z6, z7, z8, ncol = 2, nrow = 2)


z9 <- ggplot(data=dataset,aes(V10)) + geom_histogram(breaks=seq(0,15,by=1),color="black",fill="red") + labs(x="V10",y="Count")
z10 <- ggplot(data=dataset,aes(V11)) + geom_histogram(breaks=seq(0,15,by=1),color="black",fill="red") + labs(x="V11",y="Count")
z11 <- ggplot(data=dataset,aes(V12)) + geom_histogram(breaks=seq(0,15,by=1),color="black",fill="red") + labs(x="V12",y="Count")
z12 <- ggplot(data=dataset,aes(V13)) + geom_histogram(breaks=seq(0,15,by=1),color="black",fill="red")+ labs(x="V13",y="Count")
plot_grid(z9, z10, z11, z12, ncol = 2, nrow = 2)


z13 <- ggplot(data=dataset,aes(V14)) + geom_histogram(breaks=seq(0,15,by=1),color="black",fill="red") + labs(x="V14",y="Count")
z14 <- ggplot(data=dataset,aes(V15)) + geom_histogram(breaks=seq(0,15,by=1),color="black",fill="red") + labs(x="V15",y="Count")
z15 <- ggplot(data=dataset,aes(V16)) + geom_histogram(breaks=seq(0,15,by=1),color="black",fill="red") + labs(x="V16",y="Count")
z16 <- ggplot(data=dataset,aes(V17)) + geom_histogram(breaks=seq(0,15,by=1),color="black",fill="red")+ labs(x="V17",y="Count")
plot_grid(z13, z14, z15, z16, ncol = 2, nrow = 2)


# box plot for each feature variable
p1 <- ggplot(dataset,aes(x="",y=V2)) + geom_boxplot(fill = "darkolivegreen1", color = "black",width=0.5) + theme(axis.title.y=element_blank()) + labs(x="V2")
p2 <- ggplot(dataset,aes(x="",y=V3)) + geom_boxplot(fill = "darkolivegreen1", color = "black")+ theme(axis.title.y=element_blank()) + labs(x="V3")
p3 <- ggplot(dataset,aes(x="",y=V4)) + geom_boxplot(fill = "darkolivegreen1", color = "black")+ theme(axis.title.y=element_blank()) + labs(x="V4")
p4 <- ggplot(dataset,aes(x="",y=V5)) + geom_boxplot(fill = "darkolivegreen1", color = "black")+ theme(axis.title.y=element_blank()) + labs(x="V5")
p5 <- ggplot(dataset,aes(x="",y=V6)) + geom_boxplot(fill = "darkolivegreen1", color = "black",width=0.5) + theme(axis.title.y=element_blank()) + labs(x="V6")
p6 <- ggplot(dataset,aes(x="",y=V7)) + geom_boxplot(fill = "darkolivegreen1", color = "black")+ theme(axis.title.y=element_blank()) + labs(x="V7")
p7 <- ggplot(dataset,aes(x="",y=V8)) + geom_boxplot(fill = "darkolivegreen1", color = "black")+ theme(axis.title.y=element_blank()) + labs(x="V8")
p8 <- ggplot(dataset,aes(x="",y=V9)) + geom_boxplot(fill = "darkolivegreen1", color = "black")+ theme(axis.title.y=element_blank()) + labs(x="V9")
plot_grid(p1, p2, p3, p4,p5,p6,p7,p8,ncol = 8, nrow = 1)

p9 <- ggplot(dataset,aes(x="",y=V10)) + geom_boxplot(fill = "darkolivegreen1", color = "black",width=0.5) + theme(axis.title.y=element_blank()) + labs(x="V10")
p10 <- ggplot(dataset,aes(x="",y=V11)) + geom_boxplot(fill = "darkolivegreen1", color = "black")+ theme(axis.title.y=element_blank()) + labs(x="V11")
p11 <- ggplot(dataset,aes(x="",y=V12)) + geom_boxplot(fill = "darkolivegreen1", color = "black")+ theme(axis.title.y=element_blank()) + labs(x="V12")
p12 <- ggplot(dataset,aes(x="",y=V13)) + geom_boxplot(fill = "darkolivegreen1", color = "black")+ theme(axis.title.y=element_blank()) + labs(x="V13")
p13 <- ggplot(dataset,aes(x="",y=V14)) + geom_boxplot(fill = "darkolivegreen1", color = "black",width=0.5) + theme(axis.title.y=element_blank()) + labs(x="V14")
p14 <- ggplot(dataset,aes(x="",y=V15)) + geom_boxplot(fill = "darkolivegreen1", color = "black")+ theme(axis.title.y=element_blank()) + labs(x="V15")
p15 <- ggplot(dataset,aes(x="",y=V16)) + geom_boxplot(fill = "darkolivegreen1", color = "black")+ theme(axis.title.y=element_blank()) + labs(x="V16")
p16 <- ggplot(dataset,aes(x="",y=V17)) + geom_boxplot(fill = "darkolivegreen1", color = "black")+ theme(axis.title.y=element_blank()) + labs(x="V17")
plot_grid(p9, p10, p11, p12,p13, p14, p15, p16, ncol = 8, nrow = 1)

summary(dataset)



names(datasetTable)
# check for null in the dataset
sum(is.na(dataset))

dataFeatureVariable = dataset[,-which(names(dataset)=='Letter')]
summary(dataFeatureVariable)

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

nrow(dataset)
# check if data set balanced
if(any(datasetTable<(nrow(dataset)*1/26*1/13) | any(datasetTable>(nrow(dataset)*1/26*13)))) {
  3 + imbalanced_classes=names(datasetTable)[datasetTable<(nrow(dataset)*1/26*1/13) | 
                                               datasetTable>(nrow(dataset)*1/26*13)]
  print("Imbalanced Classes:")
  print(imbalanced_classes)
}else { 
  print("Data set is Balanced")
}

std_dev_data=round(apply(dataset[,-T],2,sd),4)
mean_data=round(apply(dataset[,-T],2,mean),4)
par(mfrow=c(2,1),mar=c(2,4,2,2))
plot(mean_data)
plot(std_dev_data)

# pearson coefficient
k1 = cor(dataset[sapply(dataset,is.numeric)],method="pearson")
corrplot(k1,method="number")
corrplot(k1,method="pie")

# check if constant predictor exists
const_pred = unlist(lapply(colnames(dataset),FUN=function(x) {
  TBL=table(dataset[[x]])
  ifelse(length(names(TBL)) < 2, -1*x,x)}
))
print(ifelse(any(const_pred<0),"Constant Predictors Exist","No Constant Predictors"))

# check for correlated predictors
cordata=cor(dataset[,-which(names(dataset)=='Letter')]) 
print(ifelse(any(abs(cordata[cordata!=1])>0.5),"Correlated Predictors Exist","
No Correlated Predictors"))
cor_index=which(abs(cordata)>0.5 & abs(cordata)!=1, arr.ind = T) 
cor_index=cor_index[!duplicated(cbind(pmax(cor_index[,1], cor_index[,2]), pmin(cor_index[,1], cor_index[,2]))),] 
tbl_cor_index=table(cor_index[,1])
cor_index_num=length(tbl_cor_index) # Number of correlated predictors
print(paste("Number of correlated predictors = ",cor_index_num))

names(tbl_cor_index)
cor_attributes=as.numeric(names(tbl_cor_index))

# show correlated variables
corelated_variables = ''
for(i in cor_attributes) {
  corelated_variables <- paste(corelated_variables, colnames(dataset[i]))
}
corelated_variables

# Divide into training & testing data set
set.seed(43)
randomized=dataset[sample(1:nrow(dataset),nrow(dataset)),]
tridx=sample(1:nrow(dataset),0.7*nrow(dataset),replace=F)
trainingDataset = randomized[tridx,] 
testingDataset = randomized[-tridx,]
# confirm by checking if the data set split is consistent throughout
table(dataset$Letter)/nrow(dataset)
table(trainingDataset$Letter)/nrow(trainingDataset)
table(testingDataset$Letter)/nrow(testingDataset)
