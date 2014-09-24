Using Random Forest Classification to Predict Activity Recorded from Fitness Device
========================================================
By: Polong Lin

## Background

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement – a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, my goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website here: http://groupware.les.inf.puc-rio.br/har (see the section on the Weight Lifting Exercise Dataset). 

## Goal: Predicting which of the five exercises was performed
The goal of this project is to predict the manner in which participants performed the exercise. This is the "classe" variable in the training set. Any of the other variables are available to predict with. In the end, I will also use my prediction model to predict 20 different test cases. 

## Loading the dataset


```r
library(caret)
library(randomForest)
train <- read.csv("pml-training.csv", stringsAsFactors = FALSE)
test <- read.csv("pml-testing.csv", stringsAsFactors = FALSE)
```


## Pre-processing
Many variables contained mostly NA's or strings of characters. Those were removed after manual inspection, resulting in 54 remaining variables. There were many observations of each variable (n = 19622). Because random forest processing can be quite time-consuming, I subsetted only 5000 rows for the classification to process more quickly.


```r
set.seed(13243)

# Sample 3000 rows out of original training data
train.sample <- sample(nrow(train), 5000)
train.subset <- train[train.sample, ]

# Remove irrelevant, unnecessary columns
train.dropna <- train.subset[, colSums(is.na(train.subset)) < 100]  #Columns with >100 NA values
train.dropchr <- train.dropna[, !sapply(train.dropna, is.character)]  #Columns of 'character'' type
train.final <- train.dropchr[, 4:56]  #Remove other irrelevant columns through manual inspection
train.final$classe <- factor(train.subset$classe)  #Re-insert classe column, convert to factor

# createDataPartition: Subset 20% of training data for cross-validation
trainIndex <- createDataPartition(y = train.final$classe, p = 0.8, list = FALSE)
train1 <- train.final[trainIndex, ]
train2 <- train.final[-trainIndex, ]
```



## Running a Random Forest classification tree
Here, I run a random forest algorithm to train on 80% of the training data (4002 observations). For cross-validation, I have reserved the remaining 20% (998 observations).  

Below, you can see the output for the final model, "modFit1".


```r
# Random forest on train1
modFit1 <- train(train1$classe ~ ., data = train1, method = "rf")

modFit1$finalModel
```

```
## 
## Call:
##  randomForest(x = x, y = y, mtry = param$mtry) 
##                Type of random forest: classification
##                      Number of trees: 500
## No. of variables tried at each split: 27
## 
##         OOB estimate of  error rate: 1.37%
## Confusion matrix:
##      A   B   C   D   E class.error
## A 1151   0   0   0   0     0.00000
## B   12 732   8   1   0     0.02789
## C    0  11 682   3   0     0.02011
## D    0   1  13 638   0     0.02147
## E    0   1   1   4 744     0.00800
```


## Expectations for out-of-sample error
As indicated in the output for the model above, the OOB estimate of error rate was 1.37%. In-sample error estimates are almost always too optimistic, as the estimates may be incorporating the noise from the in-sample set. Out-sample estimates of error are typically larger than in-sample estimates. Thus, I expect that the out-sample error rate will be around 1.37%, but most likely higher than 1.37%. A much higher error rate would indicate that the model is overfitting, or that there is something very different between the training set and the cross-validation dataset.

## Cross-validation
Here, I test the random forest model, "modFit1" on the remaining partitioned data, "train2", that contains 998 observations. We then report the confusion matrix showing how the model's predictions compared with the actual classes from the cross-validation dataset.


```r
predictions <- predict(modFit1, newdata = train2)

confusionMatrix(predictions, train2$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction   A   B   C   D   E
##          A 287   5   0   0   0
##          B   0 182   2   0   0
##          C   0   1 171   1   0
##          D   0   0   0 162   1
##          E   0   0   0   0 186
## 
## Overall Statistics
##                                         
##                Accuracy : 0.99          
##                  95% CI : (0.982, 0.995)
##     No Information Rate : 0.288         
##     P-Value [Acc > NIR] : <2e-16        
##                                         
##                   Kappa : 0.987         
##  Mcnemar's Test P-Value : NA            
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             1.000    0.968    0.988    0.994    0.995
## Specificity             0.993    0.998    0.998    0.999    1.000
## Pos Pred Value          0.983    0.989    0.988    0.994    1.000
## Neg Pred Value          1.000    0.993    0.998    0.999    0.999
## Prevalence              0.288    0.188    0.173    0.163    0.187
## Detection Rate          0.288    0.182    0.171    0.162    0.186
## Detection Prevalence    0.293    0.184    0.173    0.163    0.186
## Balanced Accuracy       0.996    0.983    0.993    0.996    0.997
```


## Estimating out-of-sample error


In the confusion matrix above, observe that the accuracy of the model is 0.99. Subtracting this value from 1 gives us the error rate, 0.01, or 1.002%. Compared to the in-sample error rate of 1.37%, we can observe that the out-of-sample error rate of 1.002% was actually slightly lower! This suggests that our prediction model is doing well - it is not overfitting the training dataset, and is able to predict a new dataset very well.

Overall, this model appears to have performed very well - 99% of the cross-validation dataset was classified correctly!

## Predictions of Test dataset
Below are my final predictions for the test set of 20 cases.

```r
predictions.test <- predict(modFit1, newdata = test)
data.frame(Predictions = predictions.test)
```

```
##    Predictions
## 1            B
## 2            A
## 3            A
## 4            A
## 5            A
## 6            E
## 7            D
## 8            B
## 9            A
## 10           A
## 11           B
## 12           C
## 13           B
## 14           A
## 15           E
## 16           E
## 17           A
## 18           B
## 19           B
## 20           B
```


