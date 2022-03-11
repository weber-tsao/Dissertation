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
            arbiter_data, arbiter_data_label = arbiter_puf.load_data(68, 50000, random_seed[a], random_num[a])
            puf_label = np.ones((50000, 1))*(total_num)
            total_num = total_num-1
            arbiter_data = np.concatenate((arbiter_data, puf_label), axis=1)
            total_data.append(arbiter_data)
            total_label.append(arbiter_data_label)
            
        for x in range(xor_num):
            #random_num = random.randint(1,1000)
            random_xor_num = [4,4,4,4,4,4,4,4]#[6,5,4,3,2,6]
            #random_xor_num = random.randint(2,6)
            random_num = [11,23,34,56,7,88,99,534,222,345]
            random_seed1 = [13,256,22,77,89,90,367,123,555,987]
            random_seed2 = [15,25,12,57,9,98,37,13,55,907]
            random_seed3 = [16,26,13,58,10,99,38,19,556,917]
            random_seed4 = [17,27,14,59,11,100,39,20,557,927]
            random_seed5 = [18,28,15,60,12,101,40,21,558,937]
            random_seed6 = [19,29,16,61,13,102,41,22,559,947]
            '''random_seed1 = random.sample(range(0, 1000), 6)
            random_seed2 = random.sample(range(0, 1000), 6)
            random_seed3 = random.sample(range(0, 1000), 6)
            random_seed4 = random.sample(range(0, 1000), 6)
            random_seed5 = random.sample(range(0, 1000), 6)
            random_seed6 = random.sample(range(0, 1000), 6)'''
            #print("xx")
            #print(random_xor_num)
            #print("xx")
            xor_puf = XOR_PUF()
            xor_data, xor_data_label = xor_puf.load_data(68, 50000, random_xor_num[x], random_seed1[x], random_seed2[x], 
                                                         random_seed3[x], random_seed4[x], random_seed5[x], 
                                                         random_seed6[x], random_num[x])
            puf_label = np.ones((50000, 1))*(total_num)
            total_num = total_num-1
            xor_data = np.concatenate((xor_data, puf_label), axis=1)
            total_data.append(xor_data)
            total_label.append(xor_data_label)
        
        for f in range(feedforward_num):
            #random_num = random.randint(1,1000)
            random_xor_num = [4,4,4,4,4,4,4,4]
            #random_xor_num = random.randint(2,6)
            #f1 = random.randint(1,63)
            #f2 = random.randint(1,63)
            random_num = [11,23,34,56,7,88,99,534,222,345]
            random_seed1 = [13,256,22,77,89,90,367,123,555,987]
            random_seed2 = [15,25,12,57,9,98,37,13,55,907]
            random_seed3 = [16,26,13,58,10,99,38,19,556,917]
            random_seed4 = [17,27,14,59,11,100,39,20,557,927]
            random_seed5 = [18,28,15,60,12,101,40,21,558,937]
            random_seed6 = [19,29,16,61,13,102,41,22,559,947]
            '''random_seed1 = random.sample(range(0, 1000), 6)
            random_seed2 = random.sample(range(0, 1000), 6)
            random_seed3 = random.sample(range(0, 1000), 6)
            random_seed4 = random.sample(range(0, 1000), 6)
            random_seed5 = random.sample(range(0, 1000), 6)
            random_seed6 = random.sample(range(0, 1000), 6)'''
            #print("ff")
            #print(random_xor_num)
            #print("ff")
            ff_puf = feedforward_PUF()
            ff_data, ff_data_label = ff_puf.load_data(68, 50000, random_xor_num[f], 32, 60, random_seed1[f], random_seed2[f], 
                                                         random_seed3[f], random_seed4[f], random_seed5[f], 
                                                         random_seed6[f], random_num[f])
            puf_label = np.ones((50000, 1))*(total_num)
            total_num = total_num-1
            ff_data = np.concatenate((ff_data, puf_label), axis=1)
            total_data.append(ff_data)
            total_label.append(ff_data_label)
            
        for l in range(lightweight_num):
            #random_num = random.randint(1,1000)
            random_xor_num = [4,4,4,4,4,4,4,4]#[6,5,4,3,2,6]
            #random_xor_num = random.randint(2,6)
            random_num = [11,23,34,56,7,88,99,534,222,345]
            ls_puf = lightweight_PUF()
            ls_data, ls_data_label = ls_puf.load_data(68, 50000, random_xor_num[l], random_num[l])
            puf_label = np.ones((50000, 1))*(total_num)
            total_num = total_num-1
            ls_data = np.concatenate((ls_data, puf_label), axis=1)
            total_data.append(ls_data)
            total_label.append(ls_data_label)
        
        for p in range(total_num_label):
            if p == 0:
                train_data = total_data[p]
                train_label = total_label[p]
            else:
                train_data = np.concatenate((train_data, total_data[p]))
                train_label = np.concatenate((train_label, total_label[p]))
        
        return train_data, train_label