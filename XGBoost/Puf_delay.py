# -*- coding: utf-8 -*-
"""
Created on Thu Nov  4 23:40:37 2021

@author: weber
"""
import pypuf.simulation
import pypuf.io
import numpy as np
from numpy import array, ones

class Puf:
    def __init__(self):
        self.node_num = 10
        self.N = 1000
        self.puf = pypuf.simulation.ArbiterPUF(n=self.node_num, seed=5)
        self.crp = pypuf.io.ChallengeResponseSet.from_simulation(self.puf, N=self.N, seed=2)
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
    
    def puf_path(self, challenge):
        challenge = array([challenge[0]])
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
        
    def each_stage_delay(self, challenge):
        challenge = array([challenge[0]])
        for i in range(len(challenge[0])+1):
            puf_delay = pypuf.simulation.LTFArray(weight_array=self.puf.weight_array[:, :i], bias=None, transform=self.puf.transform)
            stage_delay_diff = puf_delay.val(challenge[:, :i])  # delay diff. after the ith stage
            self.delay_diff.append(stage_delay_diff)
        
        for x in range(len(challenge[0])):
            delay_diff1 = self.delay_diff[x]
            delay_diff2 = self.delay_diff[x+1]
            self.stage_delay_diff.append(delay_diff2-delay_diff1)

        return self.delay_diff, self.stage_delay_diff
    
    def cal_each_mux_delay(self, challenge):
        # look at challenge bits
        challenge = array([challenge[0]])
        top_path, bottom_path = self.puf_path(challenge)
        delay_diff, stage_delay_diff = self.each_stage_delay(challenge)
        #print(delay_diff)
        #print(stage_delay_diff)
            
        for i in range((len(challenge[0])-1), -1, -1):
            count = i*2
            delay_different = stage_delay_diff[i]
            current_top = top_path[i]
            
            if delay_different > 0:
                if current_top%2 == 0: #top is top
                     self.dict[str(count)] = abs(delay_different).tolist()[0]
                     self.dict[str(count+1)] = 0
                else:
                    self.dict[str(count)] = 0
                    self.dict[str(count+1)] = abs(delay_different).tolist()[0]
            elif delay_different < 0:
                if current_top%2 == 0:
                    self.dict[str(count)] = 0
                    self.dict[str(count+1)] = abs(delay_different).tolist()[0]
                else:
                    self.dict[str(count)] = abs(delay_different).tolist()[0]
                    self.dict[str(count+1)] = 0
            else:
                self.dict[str(count)] = 0
                self.dict[str(count-1)] = 0        
            #print(self.dict)
        return self.dict
    
    def cal_top_bottom_delay(self, challenge, topPath, bottomPath):
        challenge = array([challenge[0]])
        top_delay = 0
        bottom_delay = 0
        top_delay_list = []
        bottom_delay_list = []
        delay_dict = self.cal_each_mux_delay(challenge)
        for i in range(len(topPath)):
            top_delay += delay_dict[str(topPath[i])]
            bottom_delay += delay_dict[str(bottomPath[i])]
            top_delay_list.append(delay_dict[str(topPath[i])])
            bottom_delay_list.append(delay_dict[str(bottomPath[i])])
        #print(topPath)
        #print(top_delay_list)
        return top_delay, bottom_delay, top_delay_list, bottom_delay_list
    
    def concat(self, a, b):
        return int(f"{a}{b}")

    def load_data(self):
        test_crps = self.crp_loaded
        data_len = len(test_crps)
        train_data = []
        train_label = []
        
        for i in range(data_len):
            # data
            challenge = list(test_crps[i][0])
            challenge = [0 if c == -1 else 1 for c in challenge]
            top_path, bottom_path = self.puf_path(test_crps[i])
            top_delay, bottom_delay, top_delay_list, bottom_delay_list = self.cal_top_bottom_delay(test_crps[i], top_path, bottom_path)
            #print(top_delay)
            #train_data.append(challenge+[top_delay]+[bottom_delay]+top_path+bottom_path)
            
            res = 1 # int(challenge[0])
            for x in range(len(challenge)):
                res = self.concat(res, challenge[x])
            
            train_data.append([res])
            #train_data.append(challenge)
            
            # label
            response = test_crps[i][1]
            response = response[0]
            if response == -1:
                train_label.append([0])
            else: 
                train_label.append([1])            
            
        train_data = np.array(train_data)
        train_label = np.array(train_label)
        
        return train_data, train_label