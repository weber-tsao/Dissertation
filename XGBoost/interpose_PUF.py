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
            
            final_delay_diff_up = puf.up.val(np.array([obfuscateChallenge]))
                
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
            
            midpoint = (stages-4)//2+1
            downChallenge = challenge[0:midpoint] + [response[0]] + challenge[midpoint:]
            final_delay_diff_down = puf.down.val(np.array([downChallenge]))
            final_delay_diff = final_delay_diff_up+final_delay_diff_down
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
        
        return train_data, train_label
        