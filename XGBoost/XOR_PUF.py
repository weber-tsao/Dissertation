# -*- coding: utf-8 -*-
"""
Created on Thu Nov  4 23:40:37 2021

@author: weber
"""
import pypuf.simulation
import pypuf.io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy import array
from LFSR_simulated import*
from pypuf.simulation import XORArbiterPUF

class XOR_PUF:
    def __init__(self):
        self.LFSR_simulated = LFSR_simulated()

    def load_data(self, stages, data_num, xor_num, cus_seed):
        puf = XORArbiterPUF(n=(stages-4), k=xor_num, seed=44)
        #puf = XORArbiterPUF(n=(stages-4), k=xor_num, seed=21, noisiness=.1)
        lfsrChallenges = random_inputs(n=stages, N=data_num, seed=cus_seed) # LFSR random challenges data
        train_data = []
        train_label = []
        data = []
        data_label = []
        delay_diff = []
        qcut_one_hot = []
        
        test_crps = lfsrChallenges
        
        for i in range(data_num):
            ### data ###
            challenge = test_crps[i]
            
            # obfuscate part
            obfuscateChallenge = self.LFSR_simulated.createObfuscateChallenge(challenge)
            obfuscateChallenge = [-1 if c == 0 else c for c in obfuscateChallenge]
            
            final_delay_diff = puf.val(np.array([obfuscateChallenge]))        
            
            challenge = challenge[4:]
            challenge = [0 if c == -1 else c for c in challenge]  
            
            ### label ###            
            response = self.LFSR_simulated.produceObfuscateResponse(puf, obfuscateChallenge)
            response = np.array(response)
            data_r = 0
            if response == -1:
                data_r = 0
            else:
                data_r = 1
            
            data.append(challenge)
            delay_diff.append(final_delay_diff[0])
            data_label.append([data_r])
           
        data = np.array(data)
        qcut_label = pd.qcut(delay_diff, q=4, labels=["1", "2", "3", "4"])
        
        data_cut = []
        for x in range(len(qcut_label)):
            if qcut_label[x] == "1":
                data_cut.append(np.concatenate((data[x],[1,0,0,0])))
            elif qcut_label[x] == "2":
                data_cut.append(np.concatenate((data[x],[0,1,0,0])))
            elif qcut_label[x] == "3":
                data_cut.append(np.concatenate((data[x],[0,0,1,0])))
            else:
                data_cut.append(np.concatenate((data[x],[0,0,0,1])))
        
        data_cut = np.array(data_cut)
        train_data = data_cut
        train_label = np.array(data_label)
        
        '''### Without qcut and one hot encode
            data.append([final_delay_diff[0]]+obfuscateChallenge+[data_r])
           
        data = np.array(data)
        train_label = data[:,-1]
        train_data = data[:,0:-1]'''
        
        return train_data, train_label
        