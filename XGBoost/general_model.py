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

class general_model:
            
    def load_data(self):
        arbiter_puf = arbiter_PUF()
        data, data_label = arbiter_puf.load_data(68, 6800, 123)
        #xor_puf = XOR_PUF()
        #data, data_label = xor_puf.load_data(68, 32000, 2, 123)
        #lightweight_puf = lightweight_PUF()
        #data, data_label = lightweight_puf.load_data(68, 80000, 3, 123)
        #feedforward_puf = feedforward_PUF()
        #data, data_label = feedforward_puf.load_data(68, 68000, 3, 32, 60, 123)
        #interpose_puf = interpose_PUF()
        #data, data_label = interpose_puf.load_data(68, 240000, 3, 3, 123)
            
        data = np.array(data)
        data = np.unique(data,axis=0)
        train_label = data[:,-1]
        train_data = data[:,0:-1]          
        
        return train_data, train_label
        