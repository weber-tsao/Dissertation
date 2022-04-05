### Import important packages ###
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
from sklearn import svm
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV, StratifiedKFold,ShuffleSplit
from sklearn.metrics import auc, plot_roc_curve, accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB,BernoulliNB,MultinomialNB,CategoricalNB
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from xgboost import plot_importance
from xgboost import plot_tree
from arbiter_PUF import*
from XOR_PUF import*
from lightweight_PUF import*
from feedforward_PUF import*
from interpose_PUF import*
from LFSR_simulated import*
from Puf_resilience import*
from Puf_delay import*
from general_model import*
from general_model2 import*
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")

ss = StratifiedKFold(n_splits=5)
# Decision Tree
knn = svm.SVC(C=100, kernel='rbf')
##APUF
#arbiter_puf = arbiter_PUF()
#data, data_label = arbiter_puf.load_data(68, 5000, 11, 123)
##XOR
#xor_puf = XOR_PUF()
#data, data_label = xor_puf.load_data(68, 5000, 6, 13,256,22,77,89,90, 11)
##FF APUF
#feedforward_puf = feedforward_PUF()
#data, data_label = feedforward_puf.load_data(68, 5000, 6, 32, 60, 256, 22, 77, 89, 90, 367, 23)
##Gneral Model
general_model = general_model()
data, data_label = general_model.load_data(1, 1, 1, 0, 0)

'''X_train, X_testVal, y_train, y_testVal = train_test_split(data, data_label, test_size=.25, random_state=66)
X_test, X_val, y_test, y_val = train_test_split(X_testVal, y_testVal, test_size=.5, random_state=24)
evals_result ={}
eval_s = [(X_train, y_train),(X_val, y_val)]
xgboostModel = XGBClassifier(
    booster='gbtree', colsample_bytree=1.0,
              eval_metric='error', gamma=0.6,
              learning_rate=0.3, max_depth=4,
              min_child_weight=20, n_estimators=300, subsample=0.8, tree_method='gpu_hist'
    )
xgboostModel.fit(X_train, y_train, eval_set=eval_s, early_stopping_rounds=100, verbose = 0)      
selection = SelectFromModel(xgboostModel, threshold=0.01, prefit=True)
data_r = selection.transform(data)
data_r, data_label = shuffle(data_r, data_label)'''
knn.fit(data, data_label)
dt_result = cross_val_score(knn, data, data_label, cv=ss)
print("Accuracy: %.2f%% (%.2f%%)" % (dt_result.mean()*100, dt_result.std()*100))


## Try unseen data
##APUF
#data_unseen, data_label_unseen = arbiter_puf.load_data(68, 5000, 11, 19)
##XOR
#data_unseen, data_label_unseen = xor_puf.load_data(68, 5000, 6, 13,256,22,77,89,90, 55)
##FF APUF
#data_unseen, data_label_unseen = feedforward_puf.load_data(68, 5000, 6, 32, 60, 256, 22, 77, 89, 90, 367, 334)
##General Model
general_model2 = general_model2()
data_unseen, data_label_unseen = general_model2.load_data(1, 1, 1, 0, 0)
'''data_unseen_reduct = selection.transform(data_unseen)
data_unseen_reduct, data_label_unseen = shuffle(data_unseen_reduct, data_label_unseen)'''
training2 = knn.score(data_unseen, data_label_unseen)
print("-------------------------------")
print('For unseen data')
print('Training accuracy: {}%'.format(training2*100))
testingacc = knn.predict(data_unseen)
cc = f1_score(data_label_unseen,testingacc)
print(cc*100)
cross_val = cross_val_score(knn, data_unseen, data_label_unseen,  cv=ss)
print("cross validation accuracy: %.2f%% (%.2f%%)" % (cross_val.mean()*100, cross_val.std()*100))
cross_val_f1 = cross_val_score(knn, data_unseen, data_label_unseen, scoring="f1", cv=ss)
print("cross validation accuracy F1: %.2f%% (%.2f%%)" % (cross_val_f1.mean()*100, cross_val_f1.std()*100))