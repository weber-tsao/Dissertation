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
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

### Set running start time ###
start_time = datetime.now()

### Load data ###
#puf = Puf()
#data, data_label = puf.load_data()
arbiter_puf = arbiter_PUF()
data, data_label = arbiter_puf.load_data(68, 5000, 13, 4)
#data_unseen, data_label_unseen = arbiter_puf.load_data(68, 5000, 13, 4)
#data_unseen = np.c_[ data_unseen, np.ones(5000)*3 ]
#xor_puf = XOR_PUF()
#data, data_label = xor_puf.load_data(68, 5000, 2, 34)
#lightweight_puf = lightweight_PUF()
#data, data_label = lightweight_puf.load_data(68, 68000, 2, 123, 11)
#feedforward_puf = feedforward_PUF()
#data, data_label = feedforward_puf.load_data(68, 5000, 2, 32, 60)
#interpose_puf = interpose_PUF()
#data, data_label = interpose_puf.load_data(68, 24000, 3, 3, 12)
#general_model = general_model()
#data, data_label = general_model.load_data(1, 1, 1, 0, 0)

### Split train, test data for the model ###
X_train, X_testVal, y_train, y_testVal = train_test_split(data, data_label, test_size=.25, random_state=66)
X_test, X_val, y_test, y_val = train_test_split(X_testVal, y_testVal, test_size=.5, random_state=24)
evals_result ={}
eval_s = [(X_train, y_train),(X_val, y_val)]

### Create XGBClassifier model ###
xgboostModel = XGBClassifier(
    booster='gbtree', colsample_bytree=1.0,
              eval_metric='error', gamma=0.6,
              learning_rate=0.3, max_depth=4,
              min_child_weight=20, n_estimators=300, subsample=0.8, tree_method='gpu_hist'
    )
xgboostModel.fit(X_train, y_train, eval_set=eval_s, early_stopping_rounds=100, verbose = 0
                 )
selection = SelectFromModel(xgboostModel, threshold=0.01, prefit=True)
print(xgboostModel.feature_importances_)
data_reduct = selection.transform(data)
data_reduct, data_label = shuffle(data_reduct, data_label)

RFC = DecisionTreeClassifier(criterion='entropy', max_depth=8)

ss = StratifiedKFold(n_splits=5)

### Calculate training time ###
end_time = datetime.now()
print('Training time: {}'.format(end_time - start_time))

### Set testing start time ###
test_start_time = datetime.now()
results = cross_val_score(RFC, data_reduct, data_label, cv=ss)
print("cross validation accuracy: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
results_f1 = cross_val_score(RFC, data_reduct, data_label, scoring="f1", cv=ss)
print("cross validation F1 accuracy: %.2f%% (%.2f%%)" % (results_f1.mean()*100, results_f1.std()*100))