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
        self.total_bits_num = 68
        self.N = 32000
        self.puf = XORArbiterPUF(n=(self.total_bits_num-4), k=2, seed=21)
        #self.puf = XORArbiterPUF(n=(self.total_bits_num-4), k=4, seed=21, noisiness=.1)
        self.lfsrChallenges = random_inputs(n=self.total_bits_num, N=self.N, seed=123) # LFSR random challenges data
        self.LFSR_simulated = LFSR_simulated()

    def load_data(self):
        data_len = self.N
        train_data = []
        train_label = []
        data = []
        
        test_crps = self.lfsrChallenges
        
        for i in range(data_len):
            ### data ###
            challenge = test_crps[i]
            
            # obfuscate part
            obfuscateChallenge = self.LFSR_simulated.createObfuscateChallenge(challenge, 0)
            obfuscateChallenge = [-1 if c == 0 else c for c in obfuscateChallenge]
            
            final_delay_diff = self.puf.val(np.array([obfuscateChallenge]))        
                
            challenge = [0 if c == -1 else c for c in challenge]            
            
            ### label ###            
            response = self.LFSR_simulated.produceObfuscateResponse(self.puf, obfuscateChallenge)
            response = np.array(response)
            data_r = 0
            if response == -1:
                data_r = 0
            else:
                data_r = 1
            
            data.append([final_delay_diff[0]]+challenge+[data_r])
        
        data = np.array(data)        
        data = np.unique(data,axis=0)
        train_label = data[:,-1]
        train_data = data[:,0:-1]            
        
        return train_data, train_label
        