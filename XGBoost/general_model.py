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
        arbiter_data, arbiter_data_label = arbiter_puf.load_data(68, 6800, 123)
        xor2_puf = XOR_PUF()
        xor2_data, xor2_data_label = xor2_puf.load_data(68, 32000, 2, 45)
        xor3_puf = XOR_PUF()
        xor3_data, xor3_data_label = xor3_puf.load_data(68, 36800, 3, 70)
        xor4_puf = XOR_PUF()
        xor4_data, xor4_data_label = xor4_puf.load_data(68, 41200, 4, 64)
        lightweight_puf = lightweight_PUF()
        lightweight_data, lightweight_data_label = lightweight_puf.load_data(68, 80000, 3, 435)
        
        train_data = np.concatenate((arbiter_data, xor2_data), axis=0)
        train_data = np.concatenate((train_data, xor3_data), axis=0)
        train_data = np.concatenate((train_data, xor4_data), axis=0)
        train_data = np.concatenate((train_data, lightweight_data), axis=0)

        train_label = np.concatenate((arbiter_data_label, xor2_data_label), axis=0)
        train_label = np.concatenate((train_label, xor3_data_label), axis=0)
        train_label = np.concatenate((train_label, xor4_data_label), axis=0)
        train_label = np.concatenate((train_label, lightweight_data_label), axis=0)
        
        return train_data, train_label