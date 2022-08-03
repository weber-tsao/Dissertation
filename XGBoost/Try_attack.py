# -*- coding: utf-8 -*-
"""
Created on Fri Jul  1 10:26:05 2022

@author: Weber
"""

import pypuf.simulation, pypuf.io
from attack import LRAttack2021
from arbiter_PUF import*
from XOR_PUF import*
import numpy as np
from sklearn.model_selection import train_test_split

from torch import optim
from AutomaticWeightedLoss import AutomaticWeightedLoss
import torch
import torch.nn as nn
import math

class LogisticRegression(torch.nn.Module):
     def __init__(self, input_dim, output_dim):
         super(LogisticRegression, self).__init__()
         self.linear = torch.nn.Linear(input_dim, output_dim)
     def forward(self, x):
         outputs = torch.sigmoid(self.linear(x))
         return outputs

class Try_attack:
    
    def __init__(self):
        #self.puf = pypuf.simulation.XORArbiterPUF(n=64, k=2, seed=1)
        #self.puf = pypuf.simulation.ArbiterPUF(n=64, seed=11)
        #self.crps = pypuf.io.ChallengeResponseSet.from_simulation(self.puf, N=50000, seed=123)
        print("start")
        #c = self.crps.challenges
        #r = self.crps.responses
        #r_final = r[0][0]
        #print(type(r_final[0]))
        #attack = LRAttack2021(self.crps, seed=3, k=4, bs=1000, lr=.001, epochs=100)
        #attack.fit()
        ####################
        #model = attack.model
        
        # init PUF instance
        arbiter_puf = arbiter_PUF()
        data, data_label, attack_data = arbiter_puf.load_data(68, 5000, 11, 123,0)
        
        #xor_puf = XOR_PUF()
        #data, data_label, attack_data = xor_puf.load_data(68, 5000, 6, 3,6,2,7,8,9, 1, 0)
        
        
        data = data.astype(np.float)
        data_label = data_label.astype(np.float)
        
        # create CRPs using custom module... 
        crp = pypuf.io.ChallengeResponseSet(attack_data, data_label)
        
        # attack
        attack = LRAttack2021(crp, seed=3, k=1, bs=1000, lr=.001, epochs=100)
        model, layer_output = attack.fit()
        array = layer_output.numpy()
        #print("Layer output = ",array[0:5])
        
        
        data_train, data_unseen, data_label, data_label_unseen = train_test_split(data, data_label, test_size=.20)
        data_no_use, data_no_use, array, array_unseen = train_test_split(data, array, test_size=.20)
        
        ### MTL learning 
        
        N = 5000
        M = 100
        c = 0.5
        p = 0.9
        k = np.random.randn(M)
        input_size = 4000
        feature_size = 68
        shared_layer_size = 64
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
            
            
            #data_test, data_test_label, no_use = xor_puf.load_data(68, 5000, 6, 1,8,2,19,98,34, 1, 0)
            data_test, data_test_label, no_use = arbiter_puf.load_data(68, 5000, 13, 12,0)
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
        
        
if __name__ == "__main__":
    try_attack_object = Try_attack()
        

