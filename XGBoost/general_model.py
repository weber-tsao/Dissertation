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
from arbiter_PUF import*
from XOR_PUF import*
from lightweight_PUF import*
from feedforward_PUF import*
from interpose_PUF import*
from LFSR_simulated import*

class general_model:
    def __init__(self):
        self.total_bits_num = 68
        self.N = 41200
        #self.puf = pypuf.simulation.ArbiterPUF(n=(self.total_bits_num-4), seed=12)
        self.puf = XORArbiterPUF(n=(self.total_bits_num-4), k=4, seed=21, noisiness=.1)
        #self.pufx = XORArbiterPUF(n=(self.total_bits_num-4), k=4, seed=34)
        #self.puf = XORFeedForwardArbiterPUF(n=(self.total_bits_num-4), k=6, ff=[(32,60)], seed=1)
        #self.pufl = LightweightSecurePUF(n=(self.total_bits_num-4), k=3, seed=10)
        #self.puf = InterposePUF(n=(self.total_bits_num-4), k_up=4, k_down=4, seed=12)
        self.lfsrChallenges = random_inputs(n=self.total_bits_num, N=self.N, seed=123) # LFSR random challenges data
        #self.xorChallenges = random_inputs(n=self.total_bits_num, N=18500, seed=134)
        self.mux_node = []
        self.top_path = []
        self.bottom_path = []
        self.LFSR_simulated = LFSR_simulated()
        self.diff_index = 0

    def load_data(self):
        arbiter_puf = arbiter_PUF()
        data, data_label = arbiter_puf.load_data()
        xor_puf = XOR_PUF()
        data, data_label = xor_puf.load_data()
        lightweight_puf = lightweight_PUF()
        data, data_label = lightweight_puf.load_data()
        #feedforward_puf = feedforward_PUF()
        #data, data_label = feedforward_puf.load_data()
        #interpose_puf = interpose_PUF()
        #data, data_label = interpose_puf.load_data()
        
        
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
            
            if i <= 8500:
                final_delay_diff = self.total_delay_diff(obfuscateChallenge)
                self.diff_index = 2
                top_path, bottom_path = self.puf_path(obfuscateChallenge)
            else:
                final_delay_diff = self.pufl.val(np.array([obfuscateChallenge]))
                self.diff_index = 4
                top_path, bottom_path = self.puf_path(obfuscateChallenge)
            
            challenge = [0 if c == -1 else c for c in challenge]
            
            ### label ###
            if self.diff_index == 2:
                response = self.LFSR_simulated.produceObfuscateResponse(self.puf, obfuscateChallenge)
            else:
                response = self.LFSR_simulated.produceObfuscateResponse(self.pufl, obfuscateChallenge)
            
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
        