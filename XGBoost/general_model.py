# -*- coding: utf-8 -*-
"""
Created on Thu Nov  4 23:40:37 2021

@author: weber
"""
import pypuf.simulation
import pypuf.io
import numpy as np
import random
import matplotlib.pyplot as plt
from numpy import array
from arbiter_PUF import*
from XOR_PUF import*
from lightweight_PUF import*
from feedforward_PUF import*
from interpose_PUF import*

class general_model:
            
    def load_data(self, arbiter_num, xor_num, feedforward_num, lightweight_num, interpose_num):
        total_data = []
        total_label = []
        
        for a in range(arbiter_num):
            random_num = random.randint(1,100)
            arbiter_puf = arbiter_PUF()
            arbiter_data, arbiter_data_label = arbiter_puf.load_data(68, 500, random_num)
            puf_label = np.ones((500, 1))*(a+1)
            arbiter_data = np.concatenate((arbiter_data, puf_label), axis=1)
            total_data.append(arbiter_data)
            total_label.append(arbiter_data_label)
            
        for x in range(xor_num):
            random_num = random.randint(1,100)
            random_xor_num = random.randint(1,6)
            xor_puf = XOR_PUF()
            xor_data, xor_data_label = xor_puf.load_data(68, 500, random_xor_num, random_num)
            puf_label = np.ones((500, 1))*(a+1)
            xor_data = np.concatenate((xor_data, puf_label), axis=1)
            total_data.append(xor_data)
            total_label.append(xor_data_label)
        
        for p in range(arbiter_num+xor_num+feedforward_num+lightweight_num+interpose_num):
            if p == 0:
                train_data = total_data[p]
                train_label = total_label[p]
            else:
                train_data = np.concatenate((train_data, total_data[p]))
                train_label = np.concatenate((train_label, total_label[p]))
        
        return train_data, train_label