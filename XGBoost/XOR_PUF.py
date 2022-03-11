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
    
    def total_delay_diff(self, challenge, puf):
        challenge = array([challenge])
        last_stage_ind = len(challenge[0])-1
        puf_delay = pypuf.simulation.LTFArray(weight_array=puf.weight_array[:, :last_stage_ind], bias=None, transform=puf.transform)
        stage_delay_diff = puf_delay.val(challenge[:, :last_stage_ind])

        return stage_delay_diff
    
    def load_data(self, stages, data_num, xor_num, puf_seed1, puf_seed2, puf_seed3, puf_seed4, puf_seed5, puf_seed6, cus_seed):
    #def load_data(self, stages, data_num, xor_num, cus_seed):
        puf1 = pypuf.simulation.ArbiterPUF(n=(stages-4), seed=91)
        puf2 = pypuf.simulation.ArbiterPUF(n=(stages-4), seed=123)
        puf3 = pypuf.simulation.ArbiterPUF(n=(stages-4), seed=43)
        puf4 = pypuf.simulation.ArbiterPUF(n=(stages-4), seed=67)
        puf5 = pypuf.simulation.ArbiterPUF(n=(stages-4), seed=90)
        puf6 = pypuf.simulation.ArbiterPUF(n=(stages-4), seed=111)
        '''puf1 = pypuf.simulation.ArbiterPUF(n=(stages-4), seed=puf_seed1)
        puf2 = pypuf.simulation.ArbiterPUF(n=(stages-4), seed=puf_seed2)
        puf3 = pypuf.simulation.ArbiterPUF(n=(stages-4), seed=puf_seed3)
        puf4 = pypuf.simulation.ArbiterPUF(n=(stages-4), seed=puf_seed4)
        puf5 = pypuf.simulation.ArbiterPUF(n=(stages-4), seed=puf_seed5)
        puf6 = pypuf.simulation.ArbiterPUF(n=(stages-4), seed=puf_seed6)'''
        puf_list = [puf1, puf2, puf3, puf4, puf5, puf6]
        #puf = XORArbiterPUF(n=(stages-4), k=xor_num, seed=21, noisiness=.05)
        lfsrChallenges = random_inputs(n=stages, N=data_num, seed=cus_seed) # LFSR random challenges data
        final_delay_diff = 1
        train_data = []
        train_label = []
        data = []
        data_label = []
        delay_diff = []
        qcut_one_hot = []
        
        test_crps = lfsrChallenges
        
        for i in range(data_num):
            ### data ###
            final_delay_diff = 1
            challenge = test_crps[i]
            
            # obfuscate part
            obfuscateChallenge = self.LFSR_simulated.createObfuscateChallenge(challenge)
            obfuscateChallenge = [-1 if c == 0 else c for c in obfuscateChallenge]
            
            for p in range(xor_num):
                final_delay_diffc = self.total_delay_diff(obfuscateChallenge, puf_list[p])
                final_delay_diff = final_delay_diffc[0]*final_delay_diff
                
                ### label ###
                response1 = puf_list[p].eval(np.array([obfuscateChallenge]))
                if response1[0] == -1:
                    response1 = 0
                else:
                    response1 = 1
                if p == 0: 
                    data_r = response1
                else:
                    data_r = data_r^response1
                    
            challenge = challenge[4:]
            challenge = [0 if c == -1 else c for c in challenge]
                                                    
            data.append(challenge)
            delay_diff.append(final_delay_diff)
            data_label.append(data_r)
           
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
        