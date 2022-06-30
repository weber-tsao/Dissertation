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
import pypuf.attack
import warnings
warnings.filterwarnings("ignore")


arbiter_puf = arbiter_PUF()
data, data_label = arbiter_puf.load_data(68, 5000, 11, 123, 0)
#data, data_unseen, data_label, data_label_unseen = train_test_split(data, data_label, test_size=.20)
#puf = pypuf.simulation.XORArbiterPUF(n=64, k=1, seed=1)
puf = pypuf.simulation.ArbiterPUF(n=64, seed=1)
print(type(data_label))
crps = pypuf.io.ChallengeResponseSet(data, data_label)
print(crps)

#crps = pypuf.io.ChallengeResponseSet.from_simulation(puf, N=5000, seed=2)
#print(crps)

attack = pypuf.attack.LRAttack2021(crps, seed=3, k=1, bs=1000, lr=.001, epochs=100)
attack.fit()

'''(N, n) = data.shape
transform = np.broadcast_to(data, (2, N, n))
print(transform)
print(transform.shape)
transpose_transform = np.transpose(transform, axes=(1,0,2))
print(transpose_transform)
print(transpose_transform.shape)

transpose_transform = np.copy(transpose_transform)

(_, _, n) = transpose_transform.shape
for i in range(n - 2, -1, -1):
   transpose_transform[:, :, i] *= transpose_transform[:, :, i + 1]
    
print(transpose_transform[:, 0, :])
print(transpose_transform[:, 0, :].shape)

data = transpose_transform[:, 0, :]'''

'''
xgboostModel_test = XGBClassifier(
    booster='gbtree', colsample_bytree=1.0,
              eval_metric='error', gamma=0.8,
              learning_rate=0.01, max_depth=5,
              min_child_weight=10, n_estimators=2000, subsample=0.8, tree_method='gpu_hist'
    )
xgboostModel_test.fit(data, data_label)                       

### Cross validation ###
#ss = ShuffleSplit(n_splits=5, test_size=0.25, random_state=3)
ss = StratifiedKFold(n_splits=5)
### Set testing start time ###
results = xgboostModel_test.score(data, data_label)
print('Training accuracy: {}%'.format(results*100))
cross_val = cross_val_score(xgboostModel_test, data, data_label,  cv=ss)
print("cross validation accuracy: %.2f%% (%.2f%%)" % (cross_val.mean()*100, cross_val.std()*100))
#results_f1 = xgboostModel.score(data_reduct, data_label)
#print("cross validation F1 accuracy: %.2f%% (%.2f%%)" % (results_f1.mean()*100, results_f1.std()*100))
        
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
'''

'''#ML = LogisticRegression()
#ML = DecisionTreeClassifier(criterion='entropy', max_depth=8)
#ML = KNeighborsClassifier(n_neighbors=8)
ML = svm.SVC(C=100, kernel='rbf')
#ML = RandomForestClassifier(max_depth=8, n_estimators=100, criterion='entropy')
ML.fit(data, data_label)

#print('Training time: {}'.format(end_time - start_time))
ss = StratifiedKFold(n_splits=5)
results = ML.score(data, data_label)
print('Training accuracy: {}%'.format(results*100))
#cross_val = cross_val_score(xgboostModel, data_reduct, data_label,  cv=ss)
#print("cross validation accuracy: %.2f%% (%.2f%%)" % (cross_val.mean()*100, cross_val.std()*100))

data_unseen, data_label_unseen = shuffle(data_unseen, data_label_unseen)
test_acc = ML.score(data_unseen, data_label_unseen)
print("---------------------------------------")
print('For unseen data')
print('Testing accuracy: {}%'.format(test_acc*100))
testingacc = ML.predict(data_unseen)
cc = f1_score(data_label_unseen,testingacc)
print(cc*100)'''