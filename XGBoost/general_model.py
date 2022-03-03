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

class general_model:
            
    def load_data(self):
        arbiter_puf = arbiter_PUF()
        arbiter_data, arbiter_data_label = arbiter_puf.load_data(68, 500, 123)
        puf_label = np.ones((500, 1))
        arbiter_data = np.concatenate((arbiter_data, puf_label), axis=1)
        xor2_puf = XOR_PUF()
        xor2_data, xor2_data_label = xor2_puf.load_data(68, 500, 2, 45)
        puf_label2 = np.ones((500, 1))*2
        xor2_data = np.concatenate((xor2_data, puf_label2), axis=1)
        xor3_puf = XOR_PUF()
        xor3_data, xor3_data_label = xor3_puf.load_data(68, 500, 3, 70)
        puf_label3 = np.ones((500, 1))*3
        xor3_data = np.concatenate((xor3_data, puf_label3), axis=1)
        xor4_puf = XOR_PUF()
        xor4_data, xor4_data_label = xor4_puf.load_data(68, 500, 4, 64)
        puf_label4 = np.ones((500, 1))*4
        xor4_data = np.concatenate((xor4_data, puf_label4), axis=1)
        xor5_puf = XOR_PUF()
        xor5_data, xor5_data_label = xor5_puf.load_data(68, 500, 5, 11)
        puf_label5 = np.ones((500, 1))*5
        xor5_data = np.concatenate((xor5_data, puf_label5), axis=1)
        xor6_puf = XOR_PUF()
        xor6_data, xor6_data_label = xor6_puf.load_data(68, 500, 6, 39)
        puf_label6 = np.ones((500, 1))*6
        xor6_data = np.concatenate((xor6_data, puf_label6), axis=1)
        
        
        
        
        train_data = np.concatenate((arbiter_data, xor2_data))
        train_data = np.concatenate((train_data, xor3_data))
        train_data = np.concatenate((train_data, xor4_data))
        train_data = np.concatenate((train_data, xor5_data))
        train_data = np.concatenate((train_data, xor6_data))

        train_label = np.concatenate((arbiter_data_label, xor2_data_label))
        train_label = np.concatenate((train_label, xor3_data_label))
        train_label = np.concatenate((train_label, xor4_data_label))
        train_label = np.concatenate((train_label, xor5_data_label))
        train_label = np.concatenate((train_label, xor6_data_label))
        
        return train_data, train_label