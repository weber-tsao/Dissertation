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
    
    def load_data(self, stages, data_num, xor_num, cus_seed):
        puf1 = pypuf.simulation.ArbiterPUF(n=(stages-4), seed=91)
        puf2 = pypuf.simulation.ArbiterPUF(n=(stages-4), seed=123)
        puf3 = pypuf.simulation.ArbiterPUF(n=(stages-4), seed=43)
        puf4 = pypuf.simulation.ArbiterPUF(n=(stages-4), seed=67)
        puf5 = pypuf.simulation.ArbiterPUF(n=(stages-4), seed=90)
        puf6 = pypuf.simulation.ArbiterPUF(n=(stages-4), seed=111)
        #puf = XORArbiterPUF(n=(stages-4), k=xor_num, seed=21, noisiness=.05)
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
            
            final_delay_diff1 = self.total_delay_diff(obfuscateChallenge, puf1)
            final_delay_diff2 = self.total_delay_diff(obfuscateChallenge, puf2)
            final_delay_diff3 = self.total_delay_diff(obfuscateChallenge, puf3)
            final_delay_diff4 = self.total_delay_diff(obfuscateChallenge, puf4)
            #final_delay_diff5 = self.total_delay_diff(obfuscateChallenge, puf5)
            #final_delay_diff6 = self.total_delay_diff(obfuscateChallenge, puf6)
            final_delay_diff = final_delay_diff1[0]*final_delay_diff2[0]*final_delay_diff3[0]*final_delay_diff4[0]
            
            challenge = challenge[4:]
            challenge = [0 if c == -1 else c for c in challenge]
            
            ### label ###            
            response1 = puf1.eval(np.array([obfuscateChallenge]))
            if response1[0] == -1:
                response1 = 0
            else:
                response1 = 1
            
            response2 = puf2.eval(np.array([obfuscateChallenge]))
            if response2[0] == -1:
                response2 = 0
            else:
                response2 = 1
            
            response3 = puf3.eval(np.array([obfuscateChallenge]))
            if response3[0] == -1:
                response3 = 0
            else:
                response3 = 1
            
            response4 = puf4.eval(np.array([obfuscateChallenge]))
            if response4[0] == -1:
                response4 = 0
            else:
                response4 = 1
            
            '''response5 = puf5.eval(np.array([obfuscateChallenge]))
            if response5[0] == -1:
                response5 = 0
            else:
                response5 = 1
            
            response6 = puf6.eval(np.array([obfuscateChallenge]))
            if response6[0] == -1:
                response6 = 0
            else:
                response6 = 1'''
            
            data_r = response1^response2^response3^response4
            
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
        