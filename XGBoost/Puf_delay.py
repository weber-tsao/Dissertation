# -*- coding: utf-8 -*-
"""
Created on Thu Nov  4 23:40:37 2021

@author: weber
"""
import pypuf.simulation
import pypuf.io
import numpy as np
from numpy import array, ones
from LFSR_simulated import*

class Puf:
    def __init__(self):
        self.node_num = 9
        self.N = 100
        self.puf = pypuf.simulation.ArbiterPUF(n=self.node_num, seed=431)
        self.crp = pypuf.io.ChallengeResponseSet.from_simulation(self.puf, N=self.N, seed=34)
        self.crp.save('crps.npz')
        self.crp_loaded = pypuf.io.ChallengeResponseSet.load('crps.npz')
        #print(self.crp_loaded[1])
        #self.challenge = array([self.crp_loaded[1][0]])
        self.delay_diff = []
        self.stage_delay_diff = []
        self.mux_node = []
        self.top_path = []
        self.bottom_path = []
        self.dict = {} # Ex. {'0':[0.1, 0.3]} --node, [delay when 1, delay when -1]
        self.LFSR_simulated = LFSR_simulated()
    
    def puf_path(self, challenge):
        #print(challenge)
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
        #print(stage_delay_diff)
        return stage_delay_diff
    
    def concat(self, a, b):
        return int(f"{a}{b}")

    def load_data(self):
        test_crps = self.crp_loaded
        data_len = len(test_crps)
        train_data = []
        train_label = []
        data = []
        
        for i in range(data_len):
            ### data ###
            challenge = list(test_crps[i][0])
            
            ### obfuscate part
            #print(challenge)
            x = self.LFSR_simulated.createObfuscateChallenge(challenge)
            #print(x)
            #print(test_crps[i])
            #print(self.total_delay_diff(list(x)))
            
            final_delay_diff = self.total_delay_diff(test_crps[i][0])
            top_path, bottom_path = self.puf_path(test_crps[i][0])
            
            top_path_num = 0
            bottom_path_num = 0
            for x in range(len(top_path)):
                top_path_num = self.concat(top_path_num, top_path[x])
                bottom_path_num = self.concat(bottom_path_num, bottom_path[x])
                
            
            challenge = [0 if c == -1 else 1 for c in challenge]
            #train_data.append(challenge+top_path+bottom_path)
            
            ### label ###
            response = test_crps[i][1]
            response = response[0]
            data_r = 0
            if response == -1:
                #train_label.append([0])
                data_r = 0
            else: 
                #train_label.append([1])
                data_r = 1
                
            
            data.append([final_delay_diff[0]]+challenge+top_path+bottom_path+[data_r])
            
        #train_data = np.array(train_data)
        #train_label = np.array(train_label)
        
        data = np.array(data)
        data = np.unique(data,axis=0)
        train_label = data[:,-1]
        train_data = data[:,0:-1]
        
        return train_data, train_label