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
from pypuf.simulation import InterposePUF

class interpose_PUF:
    def __init__(self):
        self.LFSR_simulated = LFSR_simulated()
    
    
    def load_data(self, stages, data_num, xor_num_up, xor_num_down, cus_seed):
        puf = InterposePUF(n=(stages-4), k_up=xor_num_up, k_down=xor_num_down, seed=12)
        #puf = InterposePUF(n=(stages-4), k_up=xor_num_up, k_down=xor_num_down, seed=12, noisiness=.1)
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
            
            final_delay_diff_up = puf.up.val(np.array([obfuscateChallenge]))
                            
            challenge = [0 if c == -1 else c for c in challenge]
            
            ### label ###
            response = self.LFSR_simulated.produceObfuscateResponse(puf, obfuscateChallenge)
            response = np.array(response)
            data_r = 0
            if response == -1:
                data_r = 0
            else:
                data_r = 1
            
            midpoint = (stages-4)//2+1
            downChallenge = obfuscateChallenge[0:midpoint] + [response[0]] + obfuscateChallenge[midpoint:]
            final_delay_diff_down = puf.down.val(np.array([downChallenge]))
            final_delay_diff = final_delay_diff_up+final_delay_diff_down
            data.append([final_delay_diff[0]]+challenge+[data_r])
        
        data = np.array(data)
        data = np.unique(data,axis=0)
        train_label = data[:,-1]
        train_data = data[:,0:-1]            
        
        return train_data, train_label
        