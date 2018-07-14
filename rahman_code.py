#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  8 17:01:30 2018

@author: sheikhshahidurrahman
"""

import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
from numpy import array
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_predict
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier

############################### Data loading and manipulation #######################
#### Read the data
twoStepDataFinal = pd.read_csv(filepath_or_buffer = "/Users/sheikhshahidurrahman/Documents/DS/ds_dp_assessment/twoStepDataFinal.csv")
givenTestData = pd.read_csv(filepath_or_buffer = "/Users/sheikhshahidurrahman/Documents/DS/ds_dp_assessment/givenTestData.csv")
twoStepDataFinal.head()
twoStepDataFinal.columns
givenTestData.columns

# Date manipulation : Drop columns
dropColumns = ["X", "Date", "Keyword_ID", "Ad_group_ID", "Campaign_ID", "Account_ID"
               , "Device_ID", "Match_type_ID", "Revenue", "Clicks", "Conversions"
               , "rpc", "rpcBinary", "indexBinary", "campaignIndex"
               , "dateIndex", "keyADIndexX", "keyADConversionIndex"
               ] # , "dateIndexRevenue", "keyADConversionIndex" , "dateIndexConversion" , "accountIndicator" , "keyADIndexX"

dateColumns = [i for i in twoStepDataFinal.columns if i.startswith('Date=')]

allDropCol = dropColumns + dateColumns

len(allDropCol) 
#considerColumns = [i for i in twoStepDataFinal.columns if i not in allDropCol]
#len(considerColumns)

consideredColumns = ["Match_type_ID.95725474456", "Match_type_ID.872544605608"
                     , "Match_type_ID.894413617560", "Device_ID.298643508640"
                     , "Device_ID.848779586902", "Device_ID.1077718730738"
                     , "Account_ID.151664859558", "Account_ID.164144662657"
                     , "Account_ID.212779990172", "Account_ID.221354172146"
                     , "Account_ID.256188843610", "Account_ID.341124366337"
                     , "Account_ID.412971074791", "Account_ID.573604300663"
                     , "Account_ID.575525143937", "Account_ID.602182847798"
                     , "Account_ID.604905316813", "Account_ID.654870334100"
                     , "Account_ID.719583196582", "Account_ID.861287123742"
                     , "Account_ID.866124423689", "Account_ID.981453654147"
                     , "campaignIndex.0", "campaignIndex.1", "campaignIndex.4"
                     , "campaignIndex.6", "campaignIndex.8", "campaignIndex.10"
                     , "campaignIndex.20", "campaignIndex.40", "campaignIndex.60"
                     , "campaignIndex.80", "campaignIndex.100", "dayFromDate.Friday"
                     , "dayFromDate.Monday", "dayFromDate.Saturday", "dayFromDate.Sunday"
                     , "dayFromDate.Thursday", "dayFromDate.Tuesday", "dayFromDate.Wednesday"
                     , "keyADIndexX", "keyADConversionIndex.1"
                     , "keyADConversionIndex.2", "keyADConversionIndex.3"
                     , "keyADConversionIndex.4", "keyADConversionIndex.5"
                     , "keyADConversionIndex.6", "keyADConversionIndex.7"
                     , "keyADConversionIndex.8", "keyADConversionIndex.9"
                     , "keyADConversionIndex.10", "keyADConversionIndex.15"
                     , "keyADConversionIndex.25", "keyADConversionIndex.50"
                     , "keyADConversionIndex.100", "keyADConversionIndex.125"
                     , "keyADConversionIndex.250"]

## Train Data
x_train = twoStepDataFinal[consideredColumns]
x_train.fillna(0)
y_train = twoStepDataFinal.loc[:,twoStepDataFinal.columns == 'rpcBinary']
y_train.fillna(0)

## Test Data
xx = givenTestData.loc[givenTestData['indexBinary'] == 1]
x_prediction = xx[consideredColumns]
x_prediction.fillna(0)

## Convert to array
x_train = array(x_train)
y_train = array(y_train)
y_train = y_train.reshape(len(y_train),)
assert(len(x_train)==len(y_train))

x_prediction = array(x_prediction)




######################## Modelling ############################
#### Model with SGD
sgd_clf = SGDClassifier(loss="hinge"
                        , penalty="elasticnet"
                        , alpha=0.01
                        , l1_ratio=0.15
                        , fit_intercept=True
                        , max_iter=1000
                        , tol=None
                        , shuffle=True
                        , verbose=0
                        , epsilon=0.1
                        , n_jobs=1
                        , random_state=1
                        , learning_rate="optimal"
                        , eta0=0.0
                        , class_weight=None
                        , warm_start=False
                        , average=False
                        , n_iter=None)

skfolds = StratifiedKFold(n_splits = 10, random_state=11)
skfolds1 = StratifiedKFold(n_splits = 3, random_state=22)

for train_index, test_index in skfolds.split(x_train, y_train):
    clone_clf = clone(sgd_clf)
    x_train_folds = x_train[train_index]
    y_train_folds = y_train[train_index]
    x_test_folds = x_train[test_index]
    y_test_folds = y_train[test_index]
    for train_index1, test_index1 in skfolds1.split(x_test_folds, y_test_folds):
        clone_clf = clone(sgd_clf)
        x_train_folds1 = x_test_folds[train_index1]
        y_train_folds1 = y_test_folds[train_index1]
        x_test_folds1 = x_test_folds[test_index1]
        y_test_folds1 = y_test_folds[test_index1]
        clone_clf.fit(x_train_folds1, y_train_folds1)
        y_pred1 = clone_clf.predict(x_test_folds1)
        print(confusion_matrix(y_test_folds1, y_pred1))
        pass

## data is sparse with 0.9424998 negative and 0.05750018 positive
## any naive classier would give the 0.9424998 results


sgd_clf = SGDClassifier(random_state = 1)
y_pred = cross_val_predict(sgd_clf, x_train, y_train, cv = 3)

confusion_matrix(y_train, y_pred)
array([[2579999,       0],
       [ 157401,       0]])


#### Decision tree

tree_clf = DecisionTreeClassifier(max_depth=10)
tree_clf.fit(x_train, y_train)
y_pred = tree_clf.predict(x_train)

confusion_matrix(y_train, y_pred)
array([[2579773,     226],
       [ 156917,     484]])


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
                                , random_state=111
                                , verbose=0
                                , warm_start=False
                                , class_weight=None)

skfolds = StratifiedKFold(n_splits = 1, random_state=11)
skfolds1 = StratifiedKFold(n_splits = 1, random_state=22)

for train_index, test_index in skfolds.split(x_train, y_train):
    clone_clf = clone(rf_clf)
    x_train_folds = x_train[train_index]
    y_train_folds = y_train[train_index]
    x_test_folds = x_train[test_index]
    y_test_folds = y_train[test_index]
    i = 0
    for train_index1, test_index1 in skfolds1.split(x_test_folds, y_test_folds):
        clone_clf = clone(rf_clf)
        x_train_folds1 = x_test_folds[train_index1]
        y_train_folds1 = y_test_folds[train_index1]
        x_test_folds1 = x_test_folds[test_index1]
        y_test_folds1 = y_test_folds[test_index1]
        clone_clf.fit(x_train_folds1, y_train_folds1)
        y_pred1 = clone_clf.predict(x_test_folds1)
        print(confusion_matrix(y_test_folds1, y_pred1))
        i = i + 1
        if i==3:
            break
    break
    pass

rf_clf.fit(x_train, y_train)
y_pred = rf_clf.predict(x_train)

confusion_matrix(y_train, y_pred)




#### AdaBoost Classifier
ada_clf = AdaBoostClassifier(base_estimator=None
                   , n_estimators=100
                   , learning_rate=1.0
                   , algorithm="SAMME.R"
                   , random_state=333)

skfolds = StratifiedKFold(n_splits = 20, random_state=11)
skfolds1 = StratifiedKFold(n_splits = 3, random_state=22)

for train_index, test_index in skfolds.split(x_train, y_train):
    clone_clf = clone(ada_clf)
    x_train_folds = x_train[train_index]
    y_train_folds = y_train[train_index]
    x_test_folds = x_train[test_index]
    y_test_folds = y_train[test_index]
    i = 0
    for train_index1, test_index1 in skfolds1.split(x_test_folds, y_test_folds):
        clone_clf = clone(ada_clf)
        x_train_folds1 = x_test_folds[train_index1]
        y_train_folds1 = y_test_folds[train_index1]
        x_test_folds1 = x_test_folds[test_index1]
        y_test_folds1 = y_test_folds[test_index1]
        clone_clf.fit(x_train_folds1, y_train_folds1)
        y_pred1 = clone_clf.predict(x_test_folds1)
        print(confusion_matrix(y_test_folds1, y_pred1))
        i = i + 1
        if i==3:
            break
    break
    pass

# =============================================================================
# Confusion matrix for 10 estimators
# [[1289251     749]
#  [  76215    2486]] # increase estimators
# 
# =============================================================================




#### GradientBoostingClassifier
gdc_clf = GradientBoostingClassifier(loss="deviance"
                                     , learning_rate=0.1
                                     , n_estimators=10
                                     , subsample=1.0
                                     , criterion="friedman_mse"
                                     , min_samples_split=2
                                     , min_samples_leaf=1
                                     , min_weight_fraction_leaf=0.0
                                     , max_depth=3
                                     , min_impurity_decrease=0.0
                                     , min_impurity_split=None
                                     , init=None
                                     , random_state=444
                                     , max_features="auto"
                                     , verbose=0
                                     , max_leaf_nodes=None
                                     , warm_start=False
                                     , presort="auto")    

skfolds = StratifiedKFold(n_splits = 2, random_state=44)

for train_index, test_index in skfolds.split(x_train, y_train):
    clone_clf = clone(gdc_clf)
    x_train_folds = x_train[train_index]
    y_train_folds = y_train[train_index]
    x_test_folds = x_train[test_index]
    y_test_folds = y_train[test_index]
    clone_clf.fit(x_train_folds, y_train_folds)
    y_pred = clone_clf.predict(x_test_folds)
    print(confusion_matrix(y_test_folds, y_pred))
    pass




#### KNeighborsClassifier(5)
knn_clf = KNeighborsClassifier(n_neighbors=5
                               , weights="uniform"
                               , algorithm="kd_tree"
                               , leaf_size=30
                               , p=.8
                               , metric="minkowski"
                               , metric_params=None
                               , n_jobs=4)

skfolds = StratifiedKFold(n_splits = 20, random_state=44)
skfolds1 = StratifiedKFold(n_splits = 3, random_state=44)

for train_index, test_index in skfolds.split(x_train, y_train):
    clone_clf = clone(knn_clf)
    x_train_folds = x_train[train_index]
    y_train_folds = y_train[train_index]
    x_test_folds = x_train[test_index]
    y_test_folds = y_train[test_index]
    i=0
    for train_index1, test_index1 in skfolds1.split(x_test_folds, y_test_folds):
        clone_clf = clone(knn_clf)
        x_train_folds1 = x_test_folds[train_index1]
        y_train_folds1 = y_test_folds[train_index1]
        x_test_folds1 = x_test_folds[test_index1]
        y_test_folds1 = y_test_folds[test_index1]
        clone_clf.fit(x_train_folds1, y_train_folds1)
        y_pred1 = clone_clf.predict(x_test_folds1)
        print(confusion_matrix(y_test_folds1, y_pred1))
        i=i+1
        if i==3:
            break
    break
    pass




#### Gaussian Naive Bias
gnb_clf = GaussianNB()

skfolds = StratifiedKFold(n_splits = 20, random_state=55)
skfolds1 = StratifiedKFold(n_splits = 3, random_state=55)

for train_index, test_index in skfolds.split(x_train, y_train):
    clone_clf = clone(gnb_clf)
    x_train_folds = x_train[train_index]
    y_train_folds = y_train[train_index]
    x_test_folds = x_train[test_index]
    y_test_folds = y_train[test_index]
    #clone_clf.fit(x_train_folds, y_train_folds)
    #y_pred = clone_clf.predict(x_test_folds)
    #print(confusion_matrix(y_test_folds, y_pred))
    i=0
    for train_index1, test_index1 in skfolds1.split(x_test_folds, y_test_folds):
        clone_clf = clone(gnb_clf)
        x_train_folds1 = x_test_folds[train_index1]
        y_train_folds1 = y_test_folds[train_index1]
        x_test_folds1 = x_test_folds[test_index1]
        y_test_folds1 = y_test_folds[test_index1]
        clone_clf.fit(x_train_folds1, y_train_folds1)
        y_pred1 = clone_clf.predict(x_test_folds1)
        print(confusion_matrix(y_test_folds1, y_pred1))
        i=i+1
        if i==3:
            break
    break
    pass

# =============================================================================
# [[40066  2934]
#  [ 1931   692]] # better than previous models
# =============================================================================




#### LinearDiscriminantAnalysis()
lda_clf = LinearDiscriminantAnalysis()

skfolds = StratifiedKFold(n_splits = 20, random_state=66)
skfolds1 = StratifiedKFold(n_splits = 3, random_state=66)

for train_index, test_index in skfolds.split(x_train, y_train):
    clone_clf = clone(lda_clf)
    x_train_folds = x_train[train_index]
    y_train_folds = y_train[train_index]
    x_test_folds = x_train[test_index]
    y_test_folds = y_train[test_index]
    i=0
    for train_index1, test_index1 in skfolds1.split(x_test_folds, y_test_folds):
        clone_clf = clone(lda_clf)
        x_train_folds1 = x_test_folds[train_index1]
        y_train_folds1 = y_test_folds[train_index1]
        x_test_folds1 = x_test_folds[test_index1]
        y_test_folds1 = y_test_folds[test_index1]
        clone_clf.fit(x_train_folds1, y_train_folds1)
        y_pred1 = clone_clf.predict(x_test_folds1)
        print(confusion_matrix(y_test_folds1, y_pred1))
        i=i+1
        if i == 3:
            break
    break

# =============================================================================
# [[42538   462]
#  [ 2369   255]] # not good enough yet
# =============================================================================




#### QuadraticDiscriminantAnalysis
qda_clf = QuadraticDiscriminantAnalysis()

skfolds = StratifiedKFold(n_splits = 20, random_state=77)
skfolds1 = StratifiedKFold(n_splits = 3, random_state=77)

for train_index, test_index in skfolds.split(x_train, y_train):
    clone_clf = clone(qda_clf)
    x_train_folds = x_train[train_index]
    y_train_folds = y_train[train_index]
    x_test_folds = x_train[test_index]
    y_test_folds = y_train[test_index]
    for train_index1, test_index1 in skfolds1.split(x_test_folds, y_test_folds):
        clone_clf = clone(qda_clf)
        x_train_folds1 = x_test_folds[train_index1]
        y_train_folds1 = y_test_folds[train_index1]
        x_test_folds1 = x_test_folds[test_index1]
        y_test_folds1 = y_test_folds[test_index1]
        clone_clf.fit(x_train_folds1, y_train_folds1)
        y_pred1 = clone_clf.predict(x_test_folds1)
        print(confusion_matrix(y_test_folds1, y_pred1))
        pass


for i in range(len(x_train.columns)):
    print(i)
    print(x_train.iloc[:,i].unique())



x_train.columns[34]

pd.unique(x_train.keyADIndexX)



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

skfolds = StratifiedKFold(n_splits = 20, random_state=88)
skfolds1 = StratifiedKFold(n_splits = 3, random_state=88)

for train_index, test_index in skfolds.split(x_train, y_train):
    clone_clf = clone(mlp_clf)
    x_train_folds = x_train[train_index]
    y_train_folds = y_train[train_index]
    x_test_folds = x_train[test_index]
    y_test_folds = y_train[test_index]
    i=0
    for train_index1, test_index1 in skfolds1.split(x_test_folds, y_test_folds):
        clone_clf = clone(mlp_clf)
        x_train_folds1 = x_test_folds[train_index1]
        y_train_folds1 = y_test_folds[train_index1]
        x_test_folds1 = x_test_folds[test_index1]
        y_test_folds1 = y_test_folds[test_index1]
        clone_clf.fit(x_train_folds1, y_train_folds1)
        y_pred1 = clone_clf.predict(x_test_folds1)
        print(confusion_matrix(y_test_folds1, y_pred1))
        i=i+1
        if i==3:
            break
    break        
    pass



############################ Mutual Voting from classifiers ############################
#### Mutual voting :

skfoldsMutual = StratifiedKFold(n_splits = 20, random_state=88)
skfoldsMutual1 = StratifiedKFold(n_splits = 3, random_state=88)

i=0
for train_index1, test_index1 in skfoldsMutual.split(x_train,y_train):
    x_train_folds1 = x_train[train_index1]
    y_train_folds1 = y_train[train_index1]
    x_test_folds1 = x_train[test_index1]
    y_test_folds1 = y_train[test_index1]
    for train_index, test_index in skfoldsMutual1.split(x_test_folds1,y_test_folds1):
        x_train_folds = x_test_folds1[train_index]
        y_train_folds = y_test_folds1[train_index]
        x_test_folds = x_test_folds1[test_index]
        y_test_folds = y_test_folds1[test_index]
        
        # Random Forest
        clone_rf = clone(rf_clf)
        clone_rf.fit(x_train_folds,y_train_folds)
        y_pred_rf = clone_rf.predict(x_test_folds)
        
        # NB
        clone_gnb = clone(gnb_clf)
        clone_gnb.fit(x_train_folds,y_train_folds)
        y_pred_nb = clone_gnb.predict(x_test_folds)
        
        # knn_clf
        clone_knn = clone(knn_clf)
        clone_knn.fit(x_train_folds,y_train_folds)
        y_pred_knn = clone_knn.predict(x_test_folds)
        
        # ada_clf
        clone_ada = clone(ada_clf)
        clone_ada.fit(x_train_folds,y_train_folds)
        y_pred_ada = clone_ada.predict(x_test_folds)
        
        # lda_clf
        clone_lda = clone(lda_clf)
        clone_lda.fit(x_train_folds,y_train_folds)
        y_pred_lda = clone_lda.predict(x_test_folds)
        
        i=i+1
        if i==1:
            break
    break

y_pred_overall = y_pred_rf + y_pred_nb + y_pred_knn + y_pred_ada + y_pred_lda
len(y_pred_overall)
np.unique(y_pred_overall)
y_pred_overall = np.where(y_pred_overall>=1, 1, 0)
np.unique(y_pred_overall)
len(y_pred_overall)

print(confusion_matrix(y_test_folds, y_pred_overall))
# =============================================================================
# [[36066  6934]
#  [ 1631   992]] # better than previous models
# =============================================================================




############################ Final Prediction ################################
# Random Forest
clone_rf = clone(rf_clf)
clone_rf.fit(x_train,y_train)
y_pred_rf = clone_rf.predict(x_prediction)
#y_pred_rf = clone_rf.predict(x_train)
#confusion_matrix(y_train, y_pred_rf)

# NB
clone_gnb = clone(gnb_clf)
clone_gnb.fit(x_train,y_train)
y_pred_nb = clone_gnb.predict(x_prediction)
#y_pred_nb = clone_gnb.predict(x_train)
#confusion_matrix(y_train, y_pred_nb)

# =============================================================================
# # knn_clf
# clone_knn = clone(knn_clf)
# clone_knn.fit(x_train,y_train)
# #y_pred_knn = clone_knn.predict(x_prediction)
# y_pred_knn = clone_knn.predict(x_train)
# confusion_matrix(y_train, y_pred_nb)
# =============================================================================

# ada_clf
clone_ada = clone(ada_clf)
clone_ada.fit(x_train,y_train)
y_pred_ada = clone_ada.predict(x_prediction)
#y_pred_ada = clone_ada.predict(x_train)
#confusion_matrix(y_train, y_pred_ada)

# lda_clf
clone_lda = clone(lda_clf)
clone_lda.fit(x_train,y_train)
y_pred_lda = clone_lda.predict(x_prediction)
#confusion_matrix(y_train, y_pred_lda)

y_pred_overall = y_pred_rf + y_pred_nb + y_pred_ada + y_pred_lda
len(y_pred_overall)
np.unique(y_pred_overall)
np.bincount(y_pred_overall)
y_pred_overall = np.where(y_pred_overall>=1, 1, 0)
np.unique(y_pred_overall)
len(y_pred_overall)
np.bincount(y_pred_overall)

type(x_prediction)
type(y_pred_overall)

x_predictionOut = xx
x_predictionOut["predictionIndex"] = y_pred_overall.tolist()
np.bincount(x_predictionOut["predictionIndex"])

x_predictionOut.to_csv(path_or_buf = "/Users/sheikhshahidurrahman/Documents/DS/ds_dp_assessment/x_predictionOut.csv")









