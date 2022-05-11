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
from feedforward_PUF import*
from LFSR_simulated import*
from Puf_resilience import*
from general_model import*
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

threshold_val = [0.1,0.01,0.05,0.001,0.005,0.0001,0.0005]
depth_val = [2,3,4,5,6,7,8]
n_estimators_val = [300,400,500,600,700]
crps = range(100,5000,200)
clf_result = pd.DataFrame({#'threshold' : [],
                           #'depth': [],
                           #'n_estimators': [],
                           #'puf_seed' : [],
                           #'train_challenge_seed': [],
                           #'test_challenge_seed': [],
                           'CRPs number':[],
                           'Test split Accuracy' : [],
                           'Test split F1' : [],
                           'Training time' : [],
                           'Testing time' : []
                           })

for crp in crps:
        ### Set running start time ###
        start_time = datetime.now()
        
        ### Load data ###
        #puf = Puf()
        #data, data_label = puf.load_data()
        arbiter_puf = arbiter_PUF()
        data, data_label = arbiter_puf.load_data(68, crp, 11, 123, 0)
        #data_unseen, data_label_unseen = arbiter_puf.load_data(68, crp, 11, 19, 0)
        data, data_unseen, data_label, data_label_unseen = train_test_split(data, data_label, test_size=.20)
        #data, data_unseen, data_label, data_label_unseen = train_test_split(data, data_label, test_size=.20)
        #xor_puf = XOR_PUF()
        #data, data_label = xor_puf.load_data(68, crp, 2, 13,256,22,77,89,90, 11)
        #data, data_unseen, data_label, data_label_unseen = train_test_split(data, data_label, test_size=.20)
        #data_unseen, data_label_unseen = xor_puf.load_data(68, crp, 2, 13,256,22,77,89,90, 55)
        #data_unseen = np.c_[ data_unseen, np.ones(300)*2 ] 
        #lightweight_puf = lightweight_PUF()
        #data, data_label = lightweight_puf.load_data(68, 68000, 2, 123, 11)
        #feedforward_puf = feedforward_PUF()
        #data, data_label = feedforward_puf.load_data(68, 300, 2, 32, 60, 256, 22, 77, 89, 90, 367, 23)
        #data_unseen, data_label_unseen = feedforward_puf.load_data(68, 300, 2, 32, 60, 256, 22, 77, 89, 90, 367, 334)
        #interpose_puf = interpose_PUF()
        #data, data_label = interpose_puf.load_data(68, 24000, 3, 3, 12)    
        #general_model = general_model()
        #data, data_label = general_model.load_data(2, 0, 0, 0, 0)
        #data, data_label = shuffle(data, data_label)
        
        xgboostModel_test = XGBClassifier(
            booster='gbtree', colsample_bytree=1.0,
                      eval_metric='error', gamma=0.8,
                      learning_rate=0.01, max_depth=4,
                      min_child_weight=20, n_estimators=200, subsample=0.8, tree_method='gpu_hist'
            )
        xgboostModel_test.fit(data, data_label)                       
        
        ### Cross validation ###
        ss = StratifiedKFold(n_splits=5)
        
        ### Calculate training time ###
        end_time = datetime.now()
        print('Training time: {}'.format(end_time - start_time))
        
        ### Set testing start time ###
        test_start_time = datetime.now()
        results = xgboostModel_test.score(data, data_label)
        print('Training accuracy: {}%'.format(results*100))
        #cross_val = cross_val_score(xgboostModel, data_reduct, data_label,  cv=ss)
        #print("cross validation accuracy: %.2f%% (%.2f%%)" % (cross_val.mean()*100, cross_val.std()*100))
        
        #data_unseen_reduct = selection.transform(data_unseen)
        data_unseen_reduct, data_label_unseen = shuffle(data_unseen, data_label_unseen)
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
        print('Testing time: {}'.format(test_end_time - test_start_time))
        
        clf_result = clf_result.append({ #'threshold' : 0.01,
                                         #'depth': depth,
                                         #'n_estimators': n_estimators,
                                         #'puf_seed' : 11,
                                         #'train_challenge_seed': 123,
                                         #'test_challenge_seed': 19,
                                         'CRPs number':crp,
                                         'Test split Accuracy' : test_acc*100,
                                         'Test split F1' : cc*100,
                                         'Training time' : end_time - start_time,
                                         'Testing time' : test_end_time - test_start_time
                                         },  ignore_index=True)
        
        #clf_result.to_csv(r'C:\Users\weber\OneDrive\Desktop\Dissertation\XGBoost\{}.csv'.format(puf_seed))
clf_result.to_csv(r'C:\Users\weber\OneDrive\Desktop\Dissertation\XGBoost\CRPs_number_test.csv')