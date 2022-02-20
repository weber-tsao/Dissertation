# -*- coding: utf-8 -*-
"""
Created on Thu Nov  4 23:40:37 2021

@author: weber
"""
import pypuf.simulation
import pypuf.io
import numpy as np
import matplotlib.pyplot as plt
from numpy import array
from LFSR_simulated import*
from pypuf.simulation import XORArbiterPUF

class XOR_PUF:
    def __init__(self):
        self.LFSR_simulated = LFSR_simulated()

    def load_data(self, stages, data_num, xor_num, cus_seed):
        puf = XORArbiterPUF(n=(stages-4), k=xor_num, seed=21)
        #puf = XORArbiterPUF(n=(stages-4), k=xor_num, seed=21, noisiness=.1)
        lfsrChallenges = random_inputs(n=stages, N=data_num, seed=cus_seed) # LFSR random challenges data
        train_data = []
        train_label = []
        data = []
        
        test_crps = lfsrChallenges
        
        for i in range(data_num):
            ### data ###
            challenge = test_crps[i]
            
            # obfuscate part
            obfuscateChallenge = self.LFSR_simulated.createObfuscateChallenge(challenge, 0)
            obfuscateChallenge = [-1 if c == 0 else c for c in obfuscateChallenge]
            
            final_delay_diff = puf.val(np.array([obfuscateChallenge]))        
                
            obfuscateChallenge = [0 if c == -1 else c for c in obfuscateChallenge]         
            
            ### label ###            
            response = self.LFSR_simulated.produceObfuscateResponse(puf, obfuscateChallenge)
            response = np.array(response)
            data_r = 0
            if response == -1:
                data_r = 0
            else:
                data_r = 1
            
            data.append([final_delay_diff[0]]+obfuscateChallenge+[data_r])
        
        data = np.array(data)        
        data = np.unique(data,axis=0)
        train_label = data[:,-1]
        train_data = data[:,0:-1]            
        
        return train_data, train_label
        