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

class arbiter_PUF:
    def __init__(self):
        self.total_bits_num = 68
        self.N = 6800
        self.puf = pypuf.simulation.ArbiterPUF(n=(self.total_bits_num-4), seed=12)
        #self.puf = pypuf.simulation.ArbiterPUF(n=(self.total_bits_num-4), seed=12, noisiness=.1)
        self.lfsrChallenges = random_inputs(n=self.total_bits_num, N=self.N, seed=123) # LFSR random challenges data
        self.mux_node = []
        self.top_path = []
        self.bottom_path = []
        self.LFSR_simulated = LFSR_simulated()
    
    def puf_path(self, challenge):
        challenge = array([challenge])
        self.mux_node = [x for x in range(len(challenge[0])*2)]
        self.top_path = []
        self.bottom_path = []
        prev = 0
        for i in range(len(challenge[0])):
            count = i*2
            if challenge[0][i] == 1:
                if prev%2 != 0:
                    self.top_path.append(self.mux_node[count+1])
                    self.bottom_path.append(self.mux_node[count])
                    prev = self.mux_node[count+1]
                else:
                    self.top_path.append(self.mux_node[count])
                    self.bottom_path.append(self.mux_node[count+1])
                    prev = self.mux_node[count]
            elif challenge[0][i] == -1:
                if prev%2 != 0:
                    self.top_path.append(self.mux_node[count])
                    self.bottom_path.append(self.mux_node[count+1])
                    prev = self.mux_node[count]
                else:
                    self.top_path.append(self.mux_node[count+1])
                    self.bottom_path.append(self.mux_node[count])
                    prev = self.mux_node[count+1]

        return self.top_path, self.bottom_path
    
    def total_delay_diff(self, challenge):
        challenge = array([challenge])
        last_stage_ind = len(challenge[0])-1
        puf_delay = pypuf.simulation.LTFArray(weight_array=self.puf.weight_array[:, :last_stage_ind], bias=None, transform=self.puf.transform)
        stage_delay_diff = puf_delay.val(challenge[:, :last_stage_ind])

        return stage_delay_diff

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
            
            final_delay_diff = self.total_delay_diff(obfuscateChallenge)
                            
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
        