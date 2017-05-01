#!/usr/bin/python
## A classification algorithm to select Person of Interest from real enron data
import sys
import pickle
sys.path.append("../tools/")

from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel
from feature_format import featureFormat, targetFeatureSplit
from sklearn.cross_validation import train_test_split
from tester import dump_classifier_and_data
from my_outlier_removal_regression import OurlierRemovalRegression
from sklearn.ensemble import AdaBoostClassifier

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "rb") as data_file:
    data_dict = pickle.load(data_file)
###
### Remove an outlier
print('DUNCAN JOHN H\n', data_dict['DUNCAN JOHN H'])
data_dict.pop("TOTAL",0)
###
###List all features for feature selection
features_list = ['poi','salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus', \
'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses',\
 'exercised_stock_options', 'long_term_incentive', 'restricted_stock', 'director_fees',\
 'to_messages', 'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi',\
 'shared_receipt_with_poi'] 
###print("Raw feature list of",len(features_list), "features: \n", " , ".join(features_list[1:])+"\n", )
print("\nNo. of features before selection:",len(features_list))
### Extract features and labels from dataset
my_dataset = data_dict
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)
###
###Feature Selection to reduce dimensionality of the data.
### linear_model.LogisticRegression and svm.LinearSVC for classification. 
### Use C to control number of selections

lsvc=LinearSVC(C=0.01, penalty="l1", dual=False).fit(features, labels)
model = SelectFromModel(lsvc, prefit=True)
features = model.transform(features)
mask= model.get_support(indices=False)
###
new_features_list=set(features_list[i] for i,a in enumerate(features_list) if i>0 and mask[i-1]==True)
print("No. of features after selection:",len(new_features_list))
print("\nSelected Features:\n", new_features_list,"\n")
###split datasets for testing and training
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)
###
    
### Clean away the 10% of points that have the largest residual errors 
print("Size of total dataset", len(features))    
print("Size of test dataset", len(features_test))
print("Size of train dataset before removing 10% outliers", len(features_train))
features_train, labels_train = OurlierRemovalRegression(features_train, labels_train)
print("Size of train dataset after removing 10% outliers", len(features_train))
### Try a varity of classifiers to achieve better than .3 precision and recall
### Pipelines: http://scikit-learn.org/stable/modules/pipeline.html
#clf = GaussianNB() 
#clf = KNeighborsClassifier(n_neighbors=15) 
#clf = RandomForestClassifier()
clf = AdaBoostClassifier()  
clf = clf.fit(features_train, labels_train)
print("accuracy after outlier removal:", clf.score(features_test, labels_test, sample_weight=None))
### Dump classifier, dataset, and features_list in .pkl files for validating results.

dump_classifier_and_data(clf, my_dataset, features_list)