# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 10:08:21 2019

@author: Arsalan Ashraf
"""


import numpy as np
import pandas as pd
from sklearn.model_selection import cross_validate
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier


#types = {"Temperature" :float,"Humidity" :float,"Light" :float,"CO2" :float,"HumidityRatio" :float,"Occupancy" :int}
names = ["Temperature","Humidity","Light","CO2","HumidityRatio","Occupancy"]
dataset = pd.read_csv("datatraining.txt", usecols=names) #dtypes = types #


#array = dataset.values
X_train = dataset.iloc[:,2:5]

Y_train = dataset["Occupancy"]
print(X_train)
dataset = pd.read_csv("datatest.txt",usecols=names)
dataset.append(pd.read_csv("datatest2.txt",usecols=names))


X_test = dataset.iloc[:,2:5]
Y_test = dataset["Occupancy"]


clf = DecisionTreeClassifier()
clf.fit(X_train, Y_train)
predictions = clf.predict(X_test)
print(accuracy_score(Y_test, predictions))
print(confusion_matrix(Y_test, predictions))
print(classification_report(Y_test, predictions))
