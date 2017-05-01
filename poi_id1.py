#!/usr/bin/python
## A classification algorithm to select Person of Interest from real enron data
import sys
import pickle
sys.path.append("../tools/")


from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import train_test_split
from tester import dump_classifier_and_data
from feature_format import featureFormat, targetFeatureSplit
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
import numpy as np
from sklearn.decomposition import PCA

#from my_outlier_removal_regression import OurlierRemovalRegression

FILE_NAME = "final_project_dataset.pkl"
RAW_FEATURES_LIST = ['poi','salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus', \
'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses',\
 'exercised_stock_options', 'long_term_incentive', 'restricted_stock', 'director_fees',\
 'to_messages', 'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi',\
 'shared_receipt_with_poi']
###Load dataset
def openFile(FILE_NAME):
    ### Load the dictionary containing the dataset
    with open(FILE_NAME, "rb") as data_file:
        data_dict = pickle.load(data_file)
    ###
    ### Remove an outlier
    print('DUNCAN JOHN H\n', data_dict['DUNCAN JOHN H'])
    data_dict.pop("TOTAL",0)
    print("\nNo. of features before selection:",len(RAW_FEATURES_LIST))
    return data_dict
###
### Extract features and labels from dataset  
def formatData(data_dict):
    my_dataset = data_dict
    data = featureFormat(my_dataset, RAW_FEATURES_LIST, sort_keys = True)
    labels, features = targetFeatureSplit(data)
    print("Size of total dataset", len(features)) 
    return labels, features, my_dataset
###   
###select features   
def featureSelection(features, labels):
    lsvc=LinearSVC(C=0.01, penalty="l1", dual=False).fit(features, labels)
    model = SelectFromModel(lsvc, prefit=True)
    features = model.transform(features)
    mask= model.get_support(indices=False)
    features_list=list(RAW_FEATURES_LIST[i] for i,a in enumerate(RAW_FEATURES_LIST) if i>0 and mask[i-1]==True)
    features_list.insert(0,"poi")    
    print("No. of features after selection:",len(features_list))
    print("\nSelected Features:\n", features_list,"\n")
    return features, labels, features_list
###

### classification
def classification(features_train, features_test, labels_train, labels_test):
    ### Try a varity of classifiers to achieve better than .3 precision and recall
    clf = AdaBoostClassifier()
    #clf = GaussianNB() 
    #clf = KNeighborsClassifier(n_neighbors=15) 
    #clf = RandomForestClassifier()
    clf = clf.fit(features_train, labels_train)
    return clf

def evaluation(clf, labels, features):
    labels_predict=clf.predict(features)
    print("precision", precision_score(labels, labels_predict, average='binary'))
    print("recall", recall_score(labels, labels_predict, average='binary')) 
    print("f1_score", f1_score(labels, labels_predict, average='binary'))


def main():
    ###Load data
    data_dict = openFile(FILE_NAME)
    ###format dataset
    labels, features, my_dataset  = formatData(data_dict)
    ###select features
    ###features, labels, features_list = featureSelection(features, labels)
    ###pca 
    pca = PCA(n_components=4)
    pca.fit_transform(features, labels)
    ###split training and testing set
    features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)
    #print("Size of test dataset", len(features_test))
    clf=classification(features_train, features_test, labels_train, labels_test)
    print("accuracy", clf.score(features_test, labels_test, sample_weight=None))
    evaluation(clf, labels, features)
    #bar_chart(("true_positives","false_positives","true_negatives","false_negatives"), evaluation)
    ### Dump classifier, dataset, and features_list in .pkl files for validating results
    dump_classifier_and_data(clf, my_dataset, RAW_FEATURES_LIST)

if __name__ == '__main__':
    main()
#features_train, labels_train = OurlierRemovalRegression(features_train, labels_train)

### Pipelines: http://scikit-learn.org/stable/modules/pipeline.html



