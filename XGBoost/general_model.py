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
        total_num = arbiter_num+xor_num+feedforward_num+lightweight_num+interpose_num
        total_num_label = arbiter_num+xor_num+feedforward_num+lightweight_num+interpose_num
        total_data = []
        total_label = []
        train_data = []
        train_label = []
        
        for a in range(arbiter_num):
            #random_num = random.randint(1,1000)
            #random_seed = random.randint(1,1000)
            random_num = [11,23,34,56,7,88,99,534,222,345]
            random_seed = [13,256,22,77,89,90,367,123,555,987]
            arbiter_puf = arbiter_PUF()
            arbiter_data, arbiter_data_label = arbiter_puf.load_data(68, 3000, random_seed[a], random_num[a])
            puf_label = np.ones((3000, 1))*(total_num)
            total_num = total_num-1
            arbiter_data = np.concatenate((arbiter_data, puf_label), axis=1)
            total_data.append(arbiter_data)
            total_label.append(arbiter_data_label)
            
        for x in range(xor_num):
            #random_num = random.randint(1,1000)
            #random_xor_num = random.randint(1,6)
            random_num = [11,23,34,56,7,88,99,534,222,345]
            random_seed1 = [13,256,22,77,89,90,367,123,555,987]
            random_seed2 = [15,25,12,57,9,98,37,13,55,907]
            xor_puf = XOR_PUF()
            xor_data, xor_data_label = xor_puf.load_data(68, 3000, 2, random_seed1[x], random_seed2[x], random_num[x])
            puf_label = np.ones((3000, 1))*(total_num) #Sth wrong here, need to deal with this
            total_num = total_num-1
            xor_data = np.concatenate((xor_data, puf_label), axis=1)
            total_data.append(xor_data)
            total_label.append(xor_data_label)
        
        for p in range(total_num_label):
            if p == 0:
                train_data = total_data[p]
                train_label = total_label[p]
            else:
                train_data = np.concatenate((train_data, total_data[p]))
                train_label = np.concatenate((train_label, total_label[p]))
        
        return train_data, train_label