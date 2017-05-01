# -*- coding: utf-8 -*-
"""
Created on Fri Jul 15 20:31:24 2016

@author: 进击的樊
"""
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel

def featureSelection(features, labels):
    lsvc=LinearSVC(C=0.01, penalty="l1", dual=False).fit(features, labels)
    model = SelectFromModel(lsvc, prefit=True)
    features = model.transform(features)
    mask= model.get_support(indices=False)
    return features, mask