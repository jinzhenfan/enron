# -*- coding: utf-8 -*-
"""
Created on Thu Jul 14 20:10:34 2016

@author: 进击的樊
"""

from sklearn.ensemble import AdaBoostClassifier
from outlier_cleaner import outlierCleaner

def OurlierRemovalRegression(features_train, labels_train):
    clf0 = AdaBoostClassifier()
    clf0 = clf0.fit(features_train, labels_train)
    predictions=clf0.predict(features_train)
    cleaned_data = outlierCleaner(predictions, features_train, labels_train)
    cleaned_data=list(zip(*cleaned_data))
    return cleaned_data[1], cleaned_data[2]