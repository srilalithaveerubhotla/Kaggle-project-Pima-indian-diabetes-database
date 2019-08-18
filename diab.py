#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 13 11:20:00 2018

"""
# import packages

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import RFE
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.svm import SVR
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

# Preprocessing and data exploration

    
diab=pd.read_csv("/home/Videos/datasets/diabetes.csv")

diab.head()
diab.info
diab.dtypes
diab.isnull().sum()


diab['Outcome'].value_counts().plot.bar()
diab.hist(figsize=(12,12))
    
sns.heatmap(diab.corr(),annot=True, fmt=".2f")
plt.scatter(diab.Age,diab.BMI,label="stars",marker="*",s=30)

plt.xlabel('Age')
plt.ylabel('BMI')

y=diab['Outcome']
    
X=diab[['Age','DiabetesPedigreeFunction','Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI']]

svc = SVC(kernel="linear")
    
rfecv = RFECV(estimator=svc, step=1, cv=StratifiedKFold(2),
                  scoring='accuracy')
rfecv.fit(X, y)
    
print("Optimal number of features : %d" % rfecv.n_features_)
    
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (nb of correct classifications)")
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
plt.show()
    
estimator = SVR(kernel="linear")
selector = RFE(estimator, 5, step=1)
selector = selector.fit(X, y)
   
   
# Data Modelling and scores



split_percentage = 0.8
split = int(split_percentage*len(diab))
 
# Train data set
X_train = X[:split]
y_train = y[:split]
 
# Test data set
X_test = X[split:]
y_test = y[split:]

print(X_train)
print(X_test)


    
parameters = [{'kernel': ['rbf'],
               'gamma': [1e-4, 1e-3, 0.01, 0.1, 0.2, 0.5],
                'C': [1, 10, 100, 1000]},
              {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]
print("# Tuning hyper-parameters")
print()

clf = GridSearchCV(svm.SVC(decision_function_shape='ovr'), parameters, cv=5)
clf.fit(X_train, y_train)

print("Best parameters set found on development set:")
print()
print(clf.best_params_)
print()
print("Grid scores on training set:")
print()
means = clf.cv_results_['mean_test_score']
stds = clf.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, clf.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r"
          % (mean, std * 2, params))
print()
print("Detailed classification report:")
print()
print("The model is trained on the full development set.")
print("The scores are computed on the full evaluation set.")
print()
y_true, y_pred = y_test, clf.predict(X_test)
print(classification_report(y_true, y_pred))
print()

     


 
