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
        self.node_num = 5
        self.N = 10
        self.puf = pypuf.simulation.ArbiterPUF(n=self.node_num, seed=5)
        self.crp = pypuf.io.ChallengeResponseSet.from_simulation(self.puf, N=self.N, seed=2)
        self.crp.save('crps.npz')
        self.crp_loaded = pypuf.io.ChallengeResponseSet.load('crps.npz')
        #print(self.crp_loaded[1])
        self.challenge = array([self.crp_loaded[1][0]])
        self.delay_diff = []
        self.stage_delay_diff = []
        self.mux_node = []
        self.top_path = []
        self.bottom_path = []
        self.dict = {} # Ex. {'0':[0.1, 0.3]} --node, [delay when 1, delay when -1]
    
    def puf_path(self, challenges):
        self.mux_node = [x for x in range(len(challenges[0])*2)]
        self.top_path = []
        self.bottom_path = []
        prev = 0
        for i in range(len(challenges[0])):
            count = i*2
            if challenges[0][i] == 1:
                if prev%2 != 0:
                    self.top_path.append(self.mux_node[count+1])
                    self.bottom_path.append(self.mux_node[count])
                    prev = self.mux_node[count+1]
                else:
                    self.top_path.append(self.mux_node[count])
                    self.bottom_path.append(self.mux_node[count+1])
                    prev = self.mux_node[count]
            elif challenges[0][i] == -1:
                if prev%2 != 0:
                    self.top_path.append(self.mux_node[count])
                    self.bottom_path.append(self.mux_node[count+1])
                    prev = self.mux_node[count]
                else:
                    self.top_path.append(self.mux_node[count+1])
                    self.bottom_path.append(self.mux_node[count])
                    prev = self.mux_node[count+1]
        #print(self.top_path)
        #print(self.bottom_path)
        return self.top_path, self.bottom_path
        
    def each_stage_delay(self):
        for i in range(len(self.challenge[0])+1):
            puf_delay = pypuf.simulation.LTFArray(weight_array=self.puf.weight_array[:, :i], bias=None, transform=self.puf.transform)
            stage_delay_diff = puf_delay.val(self.challenge[:, :i])  # delay diff. after the ith stage
            self.delay_diff.append(stage_delay_diff)
        
        for x in range(len(self.challenge[0])):
            delay_diff1 = self.delay_diff[x]
            delay_diff2 = self.delay_diff[x+1]
            self.stage_delay_diff.append(delay_diff2-delay_diff1)
            
        return self.delay_diff, self.stage_delay_diff
    
    def cal_each_mux_delay(self):
        # look at challenge bits
        top_path, bottom_path = self.puf_path(self.challenge)
        #print(top_path)
        delay_diff, stage_delay_diff = self.each_stage_delay()
        #print(delay_diff)
        #print(stage_delay_diff)
            
        for i in range((len(self.challenge[0])-1), -1, -1):
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
    
    def cal_top_bottom_delay(self, topPath, bottomPath):
        top_delay = 0
        bottom_delay = 0
        top_delay_list = []
        bottom_delay_list = []
        delay_dict = self.cal_each_mux_delay()
        for i in range(len(topPath)):
            top_delay += delay_dict[str(topPath[i])]
            bottom_delay += delay_dict[str(bottomPath[i])]
            top_delay_list.append(delay_dict[str(topPath[i])])
            bottom_delay_list.append(delay_dict[str(bottomPath[i])])
        return top_delay, bottom_delay, top_delay_list, bottom_delay_list
    
    def mask(self, idx, l):
        mask = np.zeros(l)
        mask[idx] = 1
        return np.array(mask, dtype=np.bool)
    
    def load_data(self):
        test_crps = self.crp_loaded
        data_len = len(test_crps)
        # train dataset
        train_data = []
        # label
        train_label = []
        valid_label = []
        test_label = []
        adj_mat_each = []
        
        for i in range(data_len):
            '''challenge = list(test_crps[i][0])
            top_path, bottom_path = self.puf_path(test_crps[i])
            top_delay, bottom_delay = self.cal_top_bottom_delay(top_path, bottom_path)
            train_data.append(challenge+top_path+bottom_path+[top_delay]+[bottom_delay])'''
            # response
            response = test_crps[i][1]
            response = response[0]
            if response == -1:
                train_label.append([0])
            else: 
                train_label.append([1])
            #adj
            top_path, bottom_path = self.puf_path(test_crps[i])
            adj_mat = np.zeros((self.node_num*2, self.node_num*2))
            for x in range(len(top_path)-1, -1, -1):
                adj_mat[top_path[x-1]][top_path[x]] = 1
                adj_mat[top_path[x]][top_path[x]] = 1
                adj_mat[bottom_path[x-1]][bottom_path[x]] = 1
                adj_mat[bottom_path[x]][bottom_path[x]] = 1
            adj_mat_each.append(adj_mat)
                
            #print(adj_mat_each)    
            
            
        for i in range(self.node_num):
            top_path, bottom_path = self.puf_path(test_crps[i])
            top_delay, bottom_delay, top_delay_list, bottom_delay_list = self.cal_top_bottom_delay(top_path, bottom_path)
            train_data.append([top_delay_list[i]]+[top_path[i]]+[0])
            train_data.append([bottom_delay_list[i]]+[bottom_path[i]]+[1])
            
        train_data = np.array(train_data)
        train_label = np.array(train_label)
        #print(train_data)
        #(train_label.shape)
        ###
        idx_test = range(int(self.N*0.6))
        idx_train = range(int(self.N*0.6),int(self.N*0.9))
        idx_val = range(int(self.N*0.9),self.N)
    
        train_mask = self.mask(idx_train, train_label.shape[0])
        val_mask = self.mask(idx_val, train_label.shape[0])
        test_mask = self.mask(idx_test, train_label.shape[0])
    
        y_train = np.zeros(train_label.shape)
        y_val = np.zeros(train_label.shape)
        y_test = np.zeros(train_label.shape)
        y_train[train_mask, :] = train_label[train_mask, :]
        y_val[val_mask, :] = train_label[val_mask, :]
        y_test[test_mask, :] = train_label[test_mask, :]  
        
        return train_data, y_train, y_val, y_test, adj_mat_each
    
if __name__ == "__main__":
    x = Puf()
    z = x.cal_each_mux_delay()
    x.load_data()