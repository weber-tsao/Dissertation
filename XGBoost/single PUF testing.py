### Import important packages ###
import pypuf.simulation, pypuf.io
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
from feedforward_PUF import*
from LFSR_simulated import*
from Puf_resilience import*
from general_model import*
from attack import LRAttack2021
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

### Set running start time ###
start_time = datetime.now()

### Load data ###

# Arbiter PUF

arbiter_puf = arbiter_PUF()
data, data_label, attack_data= arbiter_puf.load_data(68, 5000, 11, 123,0)

# for lr2021
#attack_data, no_use, no_use2, no_use3 = train_test_split(attack_data, data_label, test_size=.20, 
#                                                                    random_state=22)

# for normal attack
#data, data_unseen, data_label, data_label_unseen = train_test_split(data, data_label, test_size=.20, 
#                                                                    random_state=22)

# XOR Arbiter PUF
#xor_puf = XOR_PUF()
#data, data_label, attack_data = xor_puf.load_data(68, 15000, 6, 13,256,22,77,89,90, 11, 0)
#new_data, data_unseen, new_data_label, data_label_unseen = train_test_split(data, data_label, test_size=.20)
#data_unseen, data_label_unseen = xor_puf.load_data(68, 5000, 3, 13,256,22,77,89,90, 55,0)
#data_unseen = np.c_[ data_unseen, np.ones(300)*2 ]

# Feedforward XOR PUF
#f1 = [5,12,26,19,33,49,51,7]
#d1 = [60,61,63,59,58,57,56,55]
#feedforward_puf = feedforward_PUF()
#data, data_label = feedforward_puf.load_data(68, 15000, 6, 32, 61, 256, 22, 77, 89, 90, 367, 23,0)
#data, data_unseen, data_label, data_label_unseen = train_test_split(data, data_label, test_size=.20)
#data_unseen, data_label_unseen = feedforward_puf.load_data(68, 5000, 6, f1, d1, 256, 22, 77, 89, 90, 367, 334) 

# General model
#general_model = general_model()
#data, data_label = general_model.load_data(1, 1, 1, 0, 0, 5000)
#data, data_label = shuffle(data, data_label)
#data, data_unseen, data_label, data_label_unseen = train_test_split(data, data_label, test_size=.20, random_state=66)

#general_model2 = general_model2()
#data_unseen, data_label_unseen = general_model2.load_data(1, 1, 1, 0, 0)
#data_unseen, data_label_unseen = shuffle(data_unseen, data_label_unseen)


########### For lr2021.py
data_label = data_label.astype(np.float)
#print(data_label)

# create CRPs using custom module...
crp = pypuf.io.ChallengeResponseSet(attack_data, data_label)

#print(crp.responses)

attack = LRAttack2021(crp, seed=3, k=1, bs=1000, lr=.001, epochs=100)
model, layer_output = attack.fit()
array = layer_output.numpy()
print("Array = ",array)

#load data again with new delay difference

# Arbiter PUF
new_data, new_data_label, new_attack_data= arbiter_puf.load_data_2021(68, 5000, 11, 123, 0, array)
data, data_unseen, data_label, data_label_unseen = train_test_split(new_data, new_data_label, test_size=.20)

# XOR Arbiter PUF
#new_data, new_data_label, new_attack_data= xor_puf.load_data_2021(68, 15000, 6, 13,256,22,77,89,90, 11, 
#                                                                  0, array)
#data, data_unseen, data_label, data_label_unseen = train_test_split(data, data_label, test_size=.20)

# FeedForward XOR Arbiter PUF
#new_data, new_data_label, new_attack_data= feedforward_puf.load_data_2021(68, 15000, 6, 13,256,22,77,89,90, 11, 
#                                                                  0, array)
#data, data_unseen, data_label, data_label_unseen = train_test_split(data, data_label, test_size=.20)

###########


xgboostModel_test = XGBClassifier(
    booster='gbtree', colsample_bytree=1.0,
              eval_metric='error', gamma=0.8,
              learning_rate=0.01, max_depth=4,
              min_child_weight=10, n_estimators=200, subsample=0.8, tree_method='gpu_hist'
    )
xgboostModel_test.fit(data, data_label) # for normal attack

#xgboostModel_test.fit(new_data, new_data_label) # for lr2021                           

### Cross validation ###
#ss = ShuffleSplit(n_splits=5, test_size=0.25, random_state=3)
ss = StratifiedKFold(n_splits=5)

### Calculate training time ###
end_time = datetime.now()
print('Training time: {}'.format(end_time - start_time))

test_start_time = datetime.now() # Set testing start time

### for normal attack
#results = xgboostModel_test.score(data, data_label)
#print('Training accuracy: {}%'.format(results*100))
#cross_val = cross_val_score(xgboostModel_test, data, data_label,  cv=ss)
#print("cross validation accuracy: %.2f%% (%.2f%%)" % (cross_val.mean()*100, cross_val.std()*100))
#results_f1 = xgboostModel.score(data_reduct, data_label)
#print("cross validation F1 accuracy: %.2f%% (%.2f%%)" % (results_f1.mean()*100, results_f1.std()*100))
###

### for lr2021
results = xgboostModel_test.score(data, data_label)
print('Training accuracy: {}%'.format(results*100))
cross_val = cross_val_score(xgboostModel_test, data, data_label,  cv=ss)
print("cross validation accuracy: %.2f%% (%.2f%%)" % (cross_val.mean()*100, cross_val.std()*100))
###
        
#data_unseen = np.delete(data_unseen, remove_list, axis=1)
#data_unseen_reduct = selection.transform(data_unseen)
data_unseen_reduct, data_label_unseen = shuffle(data_unseen, data_label_unseen)
test_acc = xgboostModel_test.score(data_unseen_reduct, data_label_unseen)
print("---------------------------------------")
print('For unseen data')
print('Testing accuracy: {}%'.format(test_acc*100))
testingacc = xgboostModel_test.predict(data_unseen_reduct)
cc = f1_score(data_label_unseen,testingacc)
print(cc*100)
cross_val = cross_val_score(xgboostModel_test, data_unseen_reduct, data_label_unseen,  cv=ss)
print("cross validation accuracy: %.2f%% (%.2f%%)" % (cross_val.mean()*100, cross_val.std()*100))
cross_val_f1 = cross_val_score(xgboostModel_test, data_unseen_reduct, data_label_unseen, scoring="f1", cv=ss)
print("cross validation accuracy F1: %.2f%% (%.2f%%)" % (cross_val_f1.mean()*100, cross_val_f1.std()*100))


### Calculate testing time ###
test_end_time = datetime.now()
print('Testing time: {}'.format(test_end_time - test_start_time))