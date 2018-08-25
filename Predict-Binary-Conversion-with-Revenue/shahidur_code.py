#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 23 21:09:01 2018

@author: sheikhshahidurrahman
"""

import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
from numpy import array
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_predict
import matplotlib as mpl
mpl.style.use('ggplot')


############################### Data loading and manipulation #######################
#### Read the data

ssDataModelNA = pd.read_csv(filepath_or_buffer = "/Users/sheikhshahidurrahman/Documents/DS/Stepstone/ssDataModelNA.csv")
ssDataModelNA.head()
ssDataModelNA.columns
ssDataModelNA.drop("ID", axis = 1)

## Train Data
considerCol = [i for i in ssDataModelNA.columns if i not in ["ID","Target_Sold","Target_Sales_ORNA"]]
x_trainName = ssDataModelNA.loc[:, considerCol]
y_trainName = ssDataModelNA.loc[:,ssDataModelNA.columns == 'Target_Sold']

## Convert to array
x_train = array(x_trainName)
y_train = array(y_trainName)
y_train = y_train.reshape(len(y_train),)
assert(len(x_train)==len(y_train))




######################## Modelling ############################

#### Gaussian Naive Bayes
gnb_clf = GaussianNB()

## K-Fold Cross validation
skfolds = StratifiedKFold(n_splits = 10, random_state=11)

for train_index, test_index in skfolds.split(x_train, y_train):
    clone_clf = clone(gnb_clf)
    x_train_folds = x_train[train_index]
    y_train_folds = y_train[train_index]
    x_test_folds = x_train[test_index]
    y_test_folds = y_train[test_index]
    clone_clf.fit(x_train_folds, y_train_folds)
    y_pred = clone_clf.predict(x_test_folds)
    print(confusion_matrix(y_test_folds, y_pred))
    

# =============================================================================
# [[1843  169]
#  [ 209  104]] # the result is not that good
# =============================================================================


#### RandomForestClassifier
rf_clf = RandomForestClassifier(n_estimators=100
                                , criterion="entropy"
                                , max_depth=None
                                , min_samples_split=2
                                , min_samples_leaf=1
                                , min_weight_fraction_leaf=0.0
                                , max_features="log2"
                                , max_leaf_nodes=None
                                , min_impurity_decrease=0.0
                                , min_impurity_split=None
                                , bootstrap=True
                                , oob_score=False
                                , n_jobs=4
                                , random_state=22
                                , verbose=0
                                , warm_start=False
                                , class_weight=None)

## K-Fold Cross validation
skfolds = StratifiedKFold(n_splits = 10, random_state=22)

for train_index, test_index in skfolds.split(x_train, y_train):
    clone_clf = clone(rf_clf)
    x_train_folds = x_train[train_index]
    y_train_folds = y_train[train_index]
    x_test_folds = x_train[test_index]
    y_test_folds = y_train[test_index]
    clone_clf.fit(x_train_folds, y_train_folds)
    y_pred = clone_clf.predict(x_test_folds)
    print(confusion_matrix(y_test_folds, y_pred))

# =============================================================================
# [[1990   22]
#  [  43  270]] # the result is quite better than the previous
# =============================================================================


#### GradientBoostingClassifier
gdc_clf = GradientBoostingClassifier(loss="deviance"
                                     , learning_rate=0.1
                                     , n_estimators=50
                                     , subsample=1.0
                                     , criterion="friedman_mse"
                                     , min_samples_split=2
                                     , min_samples_leaf=1
                                     , min_weight_fraction_leaf=0.0
                                     , max_depth=20
                                     , min_impurity_decrease=0.0
                                     , min_impurity_split=None
                                     , init=None
                                     , random_state=33
                                     , max_features="auto"
                                     , verbose=0
                                     , max_leaf_nodes=None
                                     , warm_start=False
                                     , presort="auto")    

## K-Fold Cross validation
skfolds = StratifiedKFold(n_splits = 10, random_state=33)

for train_index, test_index in skfolds.split(x_train, y_train):
    clone_clf = clone(gdc_clf)
    x_train_folds = x_train[train_index]
    y_train_folds = y_train[train_index]
    x_test_folds = x_train[test_index]
    y_test_folds = y_train[test_index]
    clone_clf.fit(x_train_folds, y_train_folds)
    y_pred = clone_clf.predict(x_test_folds)
    print(confusion_matrix(y_test_folds, y_pred))

# =============================================================================
# [[1968   44]
#  [  50  263]] # Very close to the RF model and is good
# =============================================================================

#### LinearDiscriminantAnalysis()
lda_clf = LinearDiscriminantAnalysis()

## K-Fold Cross validation
skfolds = StratifiedKFold(n_splits = 10, random_state=44)

for train_index, test_index in skfolds.split(x_train, y_train):
    clone_clf = clone(lda_clf)
    x_train_folds = x_train[train_index]
    y_train_folds = y_train[train_index]
    x_test_folds = x_train[test_index]
    y_test_folds = y_train[test_index]
    clone_clf.fit(x_train_folds, y_train_folds)
    y_pred = clone_clf.predict(x_test_folds)
    print(confusion_matrix(y_test_folds, y_pred))

# =============================================================================
# [[1906  106]
#  [  96  217]] # Not very good performance to be considered
# =============================================================================


#### Neural Network : MLP
mlp_clf = MLPClassifier(hidden_layer_sizes=(150, 5)
                        , activation="relu"
                        , solver="adam"
                        , alpha=0.001
                        , batch_size="auto"
                        , learning_rate="constant"
                        , learning_rate_init=0.001
                        , power_t=0.5
                        , max_iter=5000
                        , shuffle=True
                        , random_state=None
                        , tol=0.0001
                        , verbose=False
                        , warm_start=False
                        , momentum=0.9
                        , nesterovs_momentum=True
                        , early_stopping=False
                        , validation_fraction=0.1
                        , beta_1=0.9
                        , beta_2=0.999
                        , epsilon=1e-08)

## K-Fold Cross validation
skfolds = StratifiedKFold(n_splits = 10, random_state=55)

for train_index, test_index in skfolds.split(x_train, y_train):
    clone_clf = clone(mlp_clf)
    x_train_folds = x_train[train_index]
    y_train_folds = y_train[train_index]
    x_test_folds = x_train[test_index]
    y_test_folds = y_train[test_index]
    clone_clf.fit(x_train_folds, y_train_folds)
    y_pred = clone_clf.predict(x_test_folds)
    print(confusion_matrix(y_test_folds, y_pred))

# =============================================================================
# [[1912  100]
#  [ 247   66]] # poor performance as compared to the other models
# =============================================================================


#### KNeighborsClassifier(5)
knn_clf = KNeighborsClassifier(n_neighbors=3
                               , weights="uniform"
                               , algorithm="kd_tree"
                               , leaf_size=30
                               , p=1.8
                               , metric="minkowski"
                               , metric_params=None
                               , n_jobs=4)

## K-Fold Cross validation
skfolds = StratifiedKFold(n_splits = 10, random_state=66)

for train_index, test_index in skfolds.split(x_train, y_train):
    clone_clf = clone(knn_clf)
    x_train_folds = x_train[train_index]
    y_train_folds = y_train[train_index]
    x_test_folds = x_train[test_index]
    y_test_folds = y_train[test_index]
    clone_clf.fit(x_train_folds, y_train_folds)
    y_pred = clone_clf.predict(x_test_folds)
    print(confusion_matrix(y_test_folds, y_pred))

# =============================================================================
# [[1921   91]
#  [ 107  206]] # poor performance as compared to Tree models
# =============================================================================


######################## Model Selection by ROC Curve ####################

## plot for ROC curve for model comparison

def plot_roc_curve(fpr, tpr, label = None):
    plt.plot(fpr, tpr, linewidth = 2, label = label)
    plt.plot([0,1],[0,1],color='darkgrey', linestyle='dashed')
    plt.axis([0,1,0,1])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    pass

# Gaussian NB
gnbScore = cross_val_predict(gnb_clf, x_train, y_train, cv = 10, method = "predict_proba")
gnbScore = gnbScore[:,1]
gnbFPR, gnbTPR, gnbThres = roc_curve(y_train, gnbScore)

# Random Forest
rfScore = cross_val_predict(rf_clf, x_train, y_train, cv = 10, method = "predict_proba")
rfScore = rfScore[:,1]
rfFPR, rfTPR, rfThres = roc_curve(y_train, rfScore)

# Gradient Boosting
gdcScore = cross_val_predict(gdc_clf, x_train, y_train, cv = 10, method = "predict_proba")
gdcScore = gdcScore[:,1]
gdcFPR, gdcTPR, gdcThres = roc_curve(y_train, gdcScore)

# Linear Disciminant 
ldaScore = cross_val_predict(lda_clf, x_train, y_train, cv = 10, method = "predict_proba")
ldaScore = ldaScore[:,1]
ldaFPR, ldaTPR, ldaThres = roc_curve(y_train, ldaScore)

# K-Nearest Neighbor
knnScore = cross_val_predict(knn_clf, x_train, y_train, cv = 10, method = "predict_proba")
knnScore = knnScore[:,1]
knnFPR, knnTPR, knnThres = roc_curve(y_train, knnScore)

## Plot all the model ROC for comparison
# ROC curve will give us idea about AUC of different models to choose from.
fig = plt.figure(figsize=(7.195, 6.195), dpi=100)
plt.plot(gnbFPR, gnbTPR, color='dodgerblue', linestyle='dashed', label = "Gaussian NB")
plt.plot(gdcFPR, gdcTPR, color='limegreen', linestyle='dashed', label = "Gradient Boosting")
plt.plot(ldaFPR, ldaTPR, color='magenta', linestyle='dashed', label = "Linear Discriminant Analysis")
plt.plot(knnFPR, knnTPR, color='orange', linestyle='dashed', label = "K-Nearest Neighbours")
plot_roc_curve(rfFPR, rfTPR, "Random Forest")
plt.legend(loc = "lower right")
plt.show()

fig.savefig("/Users/sheikhshahidurrahman/Documents/DS/Stepstone/modelComparison.png"
            , dpi = 1000)
# Random Forest shows the best result out of all the models selected


############################ Final Model Selected ########################

#### RandomForestClassifier
rf_clf = RandomForestClassifier(n_estimators=100
                                , criterion="entropy"
                                , max_depth=None
                                , min_samples_split=2
                                , min_samples_leaf=1
                                , min_weight_fraction_leaf=0.0
                                , max_features="log2"
                                , max_leaf_nodes=None
                                , min_impurity_decrease=0.0
                                , min_impurity_split=None
                                , bootstrap=True
                                , oob_score=False
                                , n_jobs=4
                                , random_state=22
                                , verbose=0
                                , warm_start=False
                                , class_weight=None)

## K-Fold Cross validation
skfolds = StratifiedKFold(n_splits = 10, random_state=22)

## Precision Recall and % sales variable initialization
recallTrain = []
precisionTrain = []
recall = []
precision = []
percentTotal = []
thresholda = []

confusionMatrix = np.zeros((2,2))
confusionMatrixProb = np.zeros((2,2))
confusionMatrixProbTrain = np.zeros((2,2))

for train_index, test_index in skfolds.split(x_train, y_train):
    clone_clf = clone(rf_clf)
    x_train_folds = x_train[train_index]
    y_train_folds = y_train[train_index]
    x_test_folds = x_train[test_index]
    y_test_folds = y_train[test_index]
    clone_clf.fit(x_train_folds, y_train_folds)
    y_pred = clone_clf.predict(x_test_folds)
    cm = confusion_matrix(y_test_folds, y_pred)
    confusionMatrix = confusionMatrix + cm
    
    # Probability based approach; as we're asked to give results at 40% data to sales
    # it's better to approach the problem with prediction probability
    a = clone_clf.classes_
    # Train
    y_probTrain = clone_clf.predict_proba(x_train_folds)
    y_probTrain = y_probTrain[:,a.tolist().index(1)]
    y_probTrain = np.where(y_probTrain>=.1, 1, 0)
    cmProbTrain = confusion_matrix(y_train_folds, y_probTrain)
    confusionMatrixProbTrain = confusionMatrixProbTrain + cmProbTrain
    # Test
    y_prob = clone_clf.predict_proba(x_test_folds)
    y_prob = y_prob[:,a.tolist().index(1)]
    y_prob = np.where(y_prob>=.1, 1, 0)
    cmProb = confusion_matrix(y_test_folds, y_prob)
    confusionMatrixProb = confusionMatrixProb + cmProb
    
## Feature importance from training "Gain"
print(clone_clf.feature_importances_)
featureImportance = [(i,j) for i,j in enumerate(clone_clf.feature_importances_)]
fiIndex = [i for i,j in sorted(featureImportance, reverse = True, key = lambda x:x[1])]
fiValue = [j for i,j in sorted(featureImportance, reverse = True, key = lambda x:x[1])]
fiCol = [i for i in x_trainName.columns[fiIndex]]

fi = pd.DataFrame(list(zip(fiCol,fiValue)),columns=["Feature","Gain"])
fi = fi.sort_values(by='Gain', ascending=False).head(10)

## Plot feature importance
fig3 = plt.figure(figsize=(10, 6.195), dpi=100)
plt.barh(fi.loc[:,"Feature"], fi.loc[:,"Gain"], color='yellowgreen', align='center')
     #, ascending=True)
plt.xlabel("Feature Importance(Gain)")
plt.gca().invert_yaxis()
fig3.savefig("/Users/sheikhshahidurrahman/Documents/DS/Stepstone/Feature Importance.png"
            , dpi = 1000)


## Precision Recall and % to sales plot data preparation
confusionMatrix = confusionMatrix/10
confusionMatrixProb = confusionMatrixProb/10
confusionMatrixProbTrain = confusionMatrixProbTrain/10

thresholda.append(.9)
recallTrain.append(confusionMatrixProbTrain[1,1]/(confusionMatrixProbTrain[1,1]+confusionMatrixProbTrain[1,0]))
recall.append(confusionMatrixProb[1,1]/(confusionMatrixProb[1,1]+confusionMatrixProb[1,0]))
percentTotal.append((confusionMatrixProb[0,1] + confusionMatrixProb[1,1])/np.sum(confusionMatrixProb))
precisionTrain.append(confusionMatrixProbTrain[1,1]/(confusionMatrixProbTrain[1,1]+confusionMatrixProbTrain[0,1]))
precision.append(confusionMatrixProb[1,1]/(confusionMatrixProb[1,1]+confusionMatrixProb[0,1]))

print(thresholda)
print(recallTrain)
print(recall)
print(percentTotal)
print(precisionTrain)
print(precision)


## Plot the Precision Recall and % to Sales on the test data set
def plot_final(x, y, label = None):
    plt.plot(x, y, linewidth = 1.5, color = "tomato", linestyle = "dashed", label = label)
    #plt.plot([0,1],[0,1],'k--')
    plt.axis([0,1,0,1])
    plt.xlabel("Threshold")
    plt.ylabel("Value")
    pass

fig2 = plt.figure(figsize=(7.195, 6.195), dpi=100)
# plt.plot(thresholda, recallTrain, "b:", label = "Recall Train")
plt.plot(thresholda, recall, color = "limegreen", linestyle = "dashed", label = "Recall Test")
# plt.plot(thresholda, precisionTrain, "g:", label = "Precision Train")
plot_final(thresholda, precision, "Precision Test")
plt.plot(thresholda, percentTotal, color = "dodgerblue", linestyle = "dashed", label = "% to Sales") # 
plt.legend(loc = "upper right")
plt.show()

fig2.savefig("/Users/sheikhshahidurrahman/Documents/DS/Stepstone/modelPerformance.png"
            , dpi = 1000)


################################ End ############################

