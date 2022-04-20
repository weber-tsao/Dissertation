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

Number_of_PUF = [1,2,3,4,5,6,10,20]
clf_result = pd.DataFrame({#'threshold' : [],
                           #'depth': [],
                           #'n_estimators': [],
                           #'puf_seed' : [],
                           #'train_challenge_seed': [],
                           'Number of APUF': [],
                           'Number of XOR-PUF': [],
                           'Number of FF-XOR-APUF': [],
                           'CRPs number':[],
                           'Test split Accuracy' : [],
                           'Test split F1' : [],
                           'Training time' : [],
                           'Testing time' : []
                           #'Cross_val Accuracy' : [],
                           #'Cross_val sd' : [],
                           #'Cross_val split F1' : [],
                           #'Cross_val F1 sd' : [],
                           })

#for (puf_seed, train_challenge_seed, test_challenge_seed) in zip(puf_seeds, train_challenge_seeds, test_challenge_seeds):
#for thresholds in threshold_val:
#for depth in depth_val:
    #for n_estimators in n_estimators_val:
for NoP in Number_of_PUF:
        ### Set running start time ###
        start_time = datetime.now()
        
        ### Load data ###
        g1 = general_model()
        data, data_label = g1.load_data(NoP, NoP, NoP, 0, 0, int(np.floor(5000/NoP)))
        data, data_label = shuffle(data, data_label)
        data, data_unseen, data_label, data_label_unseen = train_test_split(data, data_label, test_size=.20)
        
        #g2 = general_model2()
        #data_unseen, data_label_unseen = g2.load_data(NoP, NoP, NoP, 0, 0, int(np.floor(5000/(NoP*3))))
        #data_unseen, data_label_unseen = shuffle(data_unseen, data_label_unseen)
        
        ### Split train, test data for the model ###
        X_train, X_testVal, y_train, y_testVal = train_test_split(data, data_label, test_size=.25, random_state=66)
        X_test, X_val, y_test, y_val = train_test_split(X_testVal, y_testVal, test_size=.5, random_state=24)
        evals_result ={}
        eval_s = [(X_train, y_train),(X_val, y_val)]
        
        ### Create XGBClassifier model ###
        xgboostModel = XGBClassifier(
            booster='gbtree', colsample_bytree=1.0,
                      eval_metric='error', gamma=0.8,
                      learning_rate=0.01, max_depth=5,
                      min_child_weight=20, n_estimators=700, subsample=0.8, tree_method='gpu_hist'
            )
        
        xgboostModel.fit(X_train, y_train, eval_set=eval_s, early_stopping_rounds=100, verbose = 0)
        
        selection = SelectFromModel(xgboostModel, threshold=0.01, prefit=True)
        #print(xgboostModel.feature_importances_)
        data_reduct = selection.transform(data)
        data_reduct, data_label = shuffle(data_reduct, data_label)
        xgboostModel_test = XGBClassifier(
            booster='gbtree', colsample_bytree=1.0,
                      eval_metric='error', gamma=0.8,
                      learning_rate=0.01, max_depth=5,
                      min_child_weight=20, n_estimators=700, subsample=0.8, tree_method='gpu_hist'
            )
        xgboostModel_test.fit(data_reduct, data_label)                       
        
        ### Cross validation ###
        #ss = StratifiedKFold(n_splits=5)
        
        ### Calculate training time ###
        end_time = datetime.now()
        #print('Training time: {}'.format(end_time - start_time))
        
        ### Set testing start time ###
        test_start_time = datetime.now()
        results = xgboostModel_test.score(data_reduct, data_label)
        print('Training accuracy: {}%'.format(results*100))
        #cross_val = cross_val_score(xgboostModel, data_reduct, data_label,  cv=ss)
        #print("cross validation accuracy: %.2f%% (%.2f%%)" % (cross_val.mean()*100, cross_val.std()*100))
        
        data_unseen_reduct = selection.transform(data_unseen)
        data_unseen_reduct, data_label_unseen = shuffle(data_unseen_reduct, data_label_unseen)
        test_acc = xgboostModel_test.score(data_unseen_reduct, data_label_unseen)
        print("---------------------------------------")
        print('For unseen data')
        print('Testing accuracy: {}%'.format(test_acc*100))
        testingacc = xgboostModel_test.predict(data_unseen_reduct)
        cc = f1_score(data_label_unseen,testingacc)
        print(cc*100)
        #cross_val = cross_val_score(xgboostModel_test, data_unseen_reduct, data_label_unseen,  cv=ss)
        #print("cross validation accuracy: %.2f%% (%.2f%%)" % (cross_val.mean()*100, cross_val.std()*100))
        #cross_val_f1 = cross_val_score(xgboostModel_test, data_unseen_reduct, data_label_unseen, scoring="f1", cv=ss)
        #print("cross validation accuracy F1: %.2f%% (%.2f%%)" % (cross_val_f1.mean()*100, cross_val_f1.std()*100))
        
        ### Calculate testing time ###
        test_end_time = datetime.now()
        #print('Testing time: {}'.format(test_end_time - test_start_time))
        
        clf_result = clf_result.append({ #'threshold' : 0.01,
                                         #'depth': depth,
                                         #'n_estimators': n_estimators,
                                         #'puf_seed' : 11,
                                         #'train_challenge_seed': 123,
                                         #'test_challenge_seed': 19,
                                         'Number of APUF': NoP,
                                         'Number of XOR-PUF': NoP,
                                         'Number of FF-XOR-APUF': NoP,
                                         'CRPs number': int(np.floor(5000/NoP)),
                                         'Test split Accuracy' : test_acc*100,
                                         'Test split F1' : cc*100,
                                         'Training time' : end_time - start_time,
                                         'Testing time' : test_end_time - test_start_time
                                         },  ignore_index=True)
        
        #clf_result.to_csv(r'C:\Users\weber\OneDrive\Desktop\Dissertation\XGBoost\{}.csv'.format(puf_seed))
clf_result.to_csv(r'C:\Users\weber\OneDrive\Desktop\Dissertation\XGBoost\XGBoost_multi.csv')