# -*- coding: utf-8 -*-
"""
Created on Thu Nov  4 23:40:37 2021

@author: weber
"""
import pypuf.simulation
import pypuf.io
import numpy
from numpy import array, ones

class Puf:
    def __init__(self):
        self.puf = pypuf.simulation.ArbiterPUF(n=5, seed=2)
        self.crp = pypuf.io.ChallengeResponseSet.from_simulation(self.puf, N=5, seed=2)
        self.crp.save('crps.npz')
        self.crp_loaded = pypuf.io.ChallengeResponseSet.load('crps.npz')
        #print(self.crp_loaded[2])
        self.challenge = array([self.crp_loaded[2][0]])
        self.delay_diff = []
        self.stage_delay_diff = []
        self.mux_node = []
        self.top_path = []
        self.bottom_path = []
        self.dict = {} # Ex. {'0':[0.1, 0.3]} --node, [delay when 1, delay when -1]
    
    def puf_path(self):
        self.mux_node = [x for x in range(len(self.challenge[0])*2)]
        prev = 0
        for i in range(len(self.challenge[0])):
            count = i*2
            if self.challenge[0][i] == 1:
                if prev%2 != 0:
                    self.top_path.append(self.mux_node[count+1])
                    self.bottom_path.append(self.mux_node[count])
                    prev = self.mux_node[count+1]
                else:
                    self.top_path.append(self.mux_node[count])
                    self.bottom_path.append(self.mux_node[count+1])
                    prev = self.mux_node[count]
            elif self.challenge[0][i] == -1:
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
        top_path, bottom_path = self.puf_path()
        #print(top_path)
        delay_diff, stage_delay_diff = self.each_stage_delay()
            
        for i in range((len(self.challenge[0])-1), -1, -1):
            count = i*2
            delay_different = stage_delay_diff[i]
            current_top = top_path[i]
            
            if delay_different > 0:
                if current_top%2 == 0: #top is top
                     self.dict[str(count)] = abs(delay_different)
                     self.dict[str(count+1)] = 0
                else:
                    self.dict[str(count)] = 0
                    self.dict[str(count+1)] = abs(delay_different)
            elif delay_different < 0:
                if current_top%2 == 0:
                    self.dict[str(count)] = 0
                    self.dict[str(count+1)] = abs(delay_different)
                else:
                    self.dict[str(count)] = abs(delay_different)
                    self.dict[str(count+1)] = 0
            else:
                self.dict[str(count)] = 0
                self.dict[str(count-1)] = 0        
        
        return self.dict
    
    def reward_related_to_delay(self):
        
        delay_dict = self.cal_each_mux_delay()
        for i in range(len(delay_dict)):
            if delay_dict[(str(i))] == 0:
                delay_dict[(str(i))] = 1
            else:
                delay_dict[(str(i))] = delay_dict[(str(i))] + 1
        #print(len(delay_dict))
        return delay_dict
    

if __name__ == "__main__":
    x = Puf()
    y = x.each_stage_delay()
    #print(y)
    z = x.cal_each_mux_delay()
    #print(z)
    #x.puf_path()
    x.reward_related_to_delay()