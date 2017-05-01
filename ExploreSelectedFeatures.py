# -*- coding: utf-8 -*-
"""
Created on Sat Jul 16 19:52:26 2016

@author: 进击的樊
"""
#!/usr/bin/python
## A classification algorithm to select Person of Interest from real enron data
import sys
import pickle
sys.path.append("../tools/")
import matplotlib.pyplot
from feature_format import featureFormat, targetFeatureSplit
#from my_outlier_removal_regression import OurlierRemovalRegression

FILE_NAME = "final_project_dataset.pkl"
RAW_FEATURES_LIST = ['poi','salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus', \
'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses',\
 'exercised_stock_options', 'long_term_incentive', 'restricted_stock', 'director_fees',\
 'to_messages', 'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi',\
 'shared_receipt_with_poi']
###'poi', 'director_fees', 'to_messages', 'from_messages',\
### 'from_this_person_to_poi', 'shared_receipt_with_poi'
data_dict = pickle.load( open("final_project_dataset.pkl", "rb") )
    ###
    ### Remove an outlier
#print('DUNCAN JOHN H\n', data_dict['DUNCAN JOHN H'])
data_dict.pop("TOTAL",0)
features = ["poi", "salary", "director_fees","to_messages", 'from_messages',\
 'from_this_person_to_poi', 'shared_receipt_with_poi']
data = featureFormat(data_dict, features)
def plotFeature(index,featureName):
    for point in data:
        poi = point[0]
        y = point[index]
        matplotlib.pyplot.scatter(poi, y )
    
    matplotlib.pyplot.xlabel("poi")
    matplotlib.pyplot.ylabel(featureName)
    matplotlib.pyplot.show()

for i in range(2,7):
    plotFeature(i,features[i])
