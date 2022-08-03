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
from general_model import*
from datetime import datetime
import warnings

from torch import optim
from AutomaticWeightedLoss import AutomaticWeightedLoss
import torch
import torch.nn as nn
import math
warnings.filterwarnings("ignore")

Number_of_PUF = [5]
clf_result = pd.DataFrame({#'threshold' : [],
                           #'depth': [],
                           #'n_estimators': [],
                           #'puf_seed' : [],
                           #'train_challenge_seed': [],
                           'Number of XORPUF': [],
                           'CRPs number':[],
                           'Test split Accuracy' : []
                           #'Test split F1' : [],
                           #'Training time' : [],
                           #'Testing time' : []
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
        data, data_label, data_label2 = g1.load_data(NoP, 0, 0, 0, 0, int(np.floor(5000/NoP)))
        #data, data_label = shuffle(data, data_label)
        data_train, data_unseen, data_label, data_label_unseen = train_test_split(data, data_label, test_size=.20)
        data_no_use, data_no_use, array, array_unseen = train_test_split(data, data_label2, test_size=.20)
        #g2 = general_model2()
        #data_unseen, data_label_unseen = g2.load_data(0, 0, NoP, 0, 0, int(np.floor(5000/NoP)))
        #data_unseen, data_label_unseen = shuffle(data_unseen, data_label_unseen)
        
        
        ### MTL learning 
        
        N = 5000
        M = 100
        c = 0.5
        p = 0.9
        k = np.random.randn(M)
        input_size = 4000
        feature_size = 69
        shared_layer_size = 65
        tower_h1 = 32
        tower_h2 = 64
        output_size = 1
        LR = 0.001
        epoch = 100
        mb_size = 100
        cost1tr = []
        cost2tr = []
        cost1D = []
        cost2D = []
        cost1ts = []
        cost2ts = []
        costtr = []
        costD = []
        costts = []
        
        class MTLnet(nn.Module):
            def __init__(self):
                super(MTLnet, self).__init__()
        
                self.sharedlayer = nn.Sequential(
                    nn.Linear(feature_size, shared_layer_size),
                    nn.ReLU(),
                    nn.Dropout()
                )
                self.tower1 = nn.Sequential(
                    nn.Linear(shared_layer_size, tower_h1),
                    nn.ReLU(),
                    nn.Dropout(),
                    nn.Linear(tower_h1, tower_h2),
                    nn.ReLU(),
                    nn.Dropout(),
                    nn.Linear(tower_h2, output_size)
                )
                self.tower2 = nn.Sequential(
                    nn.Linear(shared_layer_size, tower_h1),
                    nn.ReLU(),
                    nn.Dropout(),
                    nn.Linear(tower_h1, tower_h2),
                    nn.ReLU(),
                    nn.Dropout(),
                    nn.Linear(tower_h2, output_size)
                )        
        
            def forward(self, x):
                h_shared = self.sharedlayer(x)
                out1 = self.tower1(h_shared)
                out2 = self.tower2(h_shared)
                return out1, out2
        
        def random_mini_batches(XE, R1E, R2E, mini_batch_size = 10, seed = 42): 
            # Creating the mini-batches
            np.random.seed(seed)            
            m = XE.shape[0]                  
            mini_batches = []
            permutation = list(np.random.permutation(m))
            shuffled_XE = XE[permutation,:]
            shuffled_X1R = R1E[permutation]
            shuffled_X2R = R2E[permutation]
            num_complete_minibatches = math.floor(m/mini_batch_size)
            for k in range(0, int(num_complete_minibatches)):
                mini_batch_XE = shuffled_XE[k * mini_batch_size : (k+1) * mini_batch_size, :]
                mini_batch_X1R = shuffled_X1R[k * mini_batch_size : (k+1) * mini_batch_size]
                mini_batch_X2R = shuffled_X2R[k * mini_batch_size : (k+1) * mini_batch_size]
                mini_batch = (mini_batch_XE, mini_batch_X1R, mini_batch_X2R)
                mini_batches.append(mini_batch)
            Lower = int(num_complete_minibatches * mini_batch_size)
            Upper = int(m - (mini_batch_size * math.floor(m/mini_batch_size)))
            if m % mini_batch_size != 0:
                mini_batch_XE = shuffled_XE[Lower : Lower + Upper, :]
                mini_batch_X1R = shuffled_X1R[Lower : Lower + Upper]
                mini_batch_X2R = shuffled_X2R[Lower : Lower + Upper]
                mini_batch = (mini_batch_XE, mini_batch_X1R, mini_batch_X2R)
                mini_batches.append(mini_batch)
            
            return mini_batches
        
        MTL = MTLnet()
        optimizer = torch.optim.Adam(MTL.parameters(), lr=LR)
        loss_func = nn.MSELoss()
        
        
        for it in range(epoch):
            epoch_cost = 0
            epoch_cost1 = 0
            epoch_cost2 = 0
            num_minibatches = int(input_size / mb_size) 
            
            data_m = torch.from_numpy(data_train).float()
            
            data_label1 = torch.from_numpy(np.array(data_label)).float()
            
            data_label2 = torch.from_numpy(np.array(array)).float()
            
            minibatches = random_mini_batches(data_m, data_label1, data_label2, mb_size)
            for minibatch in minibatches:
                XE, YE1, YE2  = minibatch 
                
                Yhat1, Yhat2 = MTL(XE)
                
                l1 = loss_func(Yhat1, YE1.view(-1,1))    
                l2 = loss_func(Yhat2, YE2.view(-1,1))
                loss =  (l1 + l2)/2
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_cost = epoch_cost + (loss / num_minibatches)
                epoch_cost1 = epoch_cost1 + (l1 / num_minibatches)
                epoch_cost2 = epoch_cost2 + (l2 / num_minibatches)
            costtr.append(torch.mean(epoch_cost))
            cost1tr.append(torch.mean(epoch_cost1))
            cost2tr.append(torch.mean(epoch_cost2))
            with torch.no_grad():
                
                data_unseen_m = torch.from_numpy(data_unseen).float()
                
                data_label1_unseen_m = torch.from_numpy(data_label_unseen).float()
                
                data_label2_unseen_m = torch.from_numpy(array_unseen).float()
                Yhat1D, Yhat2D = MTL(data_unseen_m)
                
                l1D = loss_func(Yhat1D, data_label1_unseen_m.view(-1,1))
                l2D = loss_func(Yhat2D, data_label2_unseen_m.view(-1,1))
                cost1D.append(l1D)
                cost2D.append(l2D)
                costD.append((l1D+l2D)/2)
                print('Iter-{}; Total loss: {:.4}'.format(it, loss.item()))
         
            
        with torch.no_grad():
        #plt.plot(np.squeeze(costtr), '-r',np.squeeze(costD), '-b')
        
            plt.plot(costtr, '-r',costD, '-b')
            plt.ylabel('total cost')
            plt.xlabel('iterations (per tens)')
            plt.show() 
            
            #plt.plot(np.squeeze(cost1tr), '-r', np.squeeze(cost1D), '-b')
            plt.plot(cost1tr, '-r', cost1D, '-b')
            plt.ylabel('task 1 cost')
            plt.xlabel('iterations (per tens)')
            plt.show() 
            
            #plt.plot(np.squeeze(cost2tr),'-r', np.squeeze(cost2D),'-b')
            plt.plot(cost2tr,'-r', cost2D,'-b')
            plt.ylabel('task 2 cost')
            plt.xlabel('iterations (per tens)')
            plt.show()
            
            running_accuracy = 0 
            total = 0 
            
            
            data_test, data_test_label, no_use = g1.load_data(NoP, 0, 0, 0, 0, 5000)
            
            for d, label1 in zip(data_test, data_test_label):
                data_tensor = torch.from_numpy(d)
                
                predicted_res, predicted_delay= MTL(data_tensor.float()) 
                #print(predicted_res.round().numpy()[0])
                #print("label:", type(label1))
                #print(predicted_res.round().numpy()[0] == label1)
                #_, predicted = torch.max(predicted_res, 1) 
                total += 1
                running_accuracy += (predicted_res.round().numpy()[0] == label1)
     
            print('inputs is: %d %%' % (100 * running_accuracy / total))    
        '''xgboostModel_test = XGBClassifier(
            booster='gbtree', colsample_bytree=0.8,
                      eval_metric='error', gamma=0.1,
                      learning_rate=0.01, max_depth=2,
                      min_child_weight=10, n_estimators=200, subsample=0.8, tree_method='gpu_hist'
            )
        xgboostModel_test.fit(data, data_label)                       
        
        ### Cross validation ###
        #ss = StratifiedKFold(n_splits=5)
        
        ### Calculate training time ###
        end_time = datetime.now()
        #print('Training time: {}'.format(end_time - start_time))
        
        ### Set testing start time ###
        test_start_time = datetime.now()
        results = xgboostModel_test.score(data, data_label)
        print('Training accuracy: {}%'.format(results*100))
        #cross_val = cross_val_score(xgboostModel, data_reduct, data_label,  cv=ss)
        #print("cross validation accuracy: %.2f%% (%.2f%%)" % (cross_val.mean()*100, cross_val.std()*100))
        
        #data_unseen_reduct = selection.transform(data_unseen)
        #data_unseen_reduct, data_label_unseen = shuffle(data_unseen, data_label_unseen)
        test_acc = xgboostModel_test.score(data_unseen, data_label_unseen)
        print("---------------------------------------")
        print('For unseen data')
        print('Testing accuracy: {}%'.format(test_acc*100))
        testingacc = xgboostModel_test.predict(data_unseen)
        cc = f1_score(data_label_unseen,testingacc)
        print(cc*100)
        #cross_val = cross_val_score(xgboostModel_test, data_unseen_reduct, data_label_unseen,  cv=ss)
        #print("cross validation accuracy: %.2f%% (%.2f%%)" % (cross_val.mean()*100, cross_val.std()*100))
        #cross_val_f1 = cross_val_score(xgboostModel_test, data_unseen_reduct, data_label_unseen, scoring="f1", cv=ss)
        #print("cross validation accuracy F1: %.2f%% (%.2f%%)" % (cross_val_f1.mean()*100, cross_val_f1.std()*100))
        
        ### Calculate testing time ###
        test_end_time = datetime.now()
        #print('Testing time: {}'.format(test_end_time - test_start_time))'''
        
        clf_result = clf_result.append({ #'threshold' : 0.01,
                                         #'depth': depth,
                                         #'n_estimators': n_estimators,
                                         #'puf_seed' : 11,
                                         #'train_challenge_seed': 123,
                                         #'test_challenge_seed': 19,
                                         'Number of XORPUF': NoP,
                                         'CRPs number': int(np.floor(50000/NoP)),
                                         'Test split Accuracy' : (100 * running_accuracy / total)
                                         #'Training time' : end_time - start_time,
                                         #'Testing time' : test_end_time - test_start_time
                                         },  ignore_index=True)
        
        #clf_result.to_csv(r'C:\Users\weber\OneDrive\Desktop\Dissertation\XGBoost\{}.csv'.format(puf_seed))
clf_result.to_csv(r'C:\Users\Asus\Desktop\Uni Year3\Dissertation\XGBoost\Generic_multiple.csv')