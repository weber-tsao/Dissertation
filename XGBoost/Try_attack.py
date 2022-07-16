# -*- coding: utf-8 -*-
"""
Created on Fri Jul  1 10:26:05 2022

@author: Weber
"""

import pypuf.simulation, pypuf.io
from attack import LRAttack2021
from arbiter_PUF import*
from XOR_PUF import*
import numpy as np



class Try_attack:
    
    def __init__(self):
        #self.puf = pypuf.simulation.XORArbiterPUF(n=64, k=2, seed=1)
        #self.puf = pypuf.simulation.ArbiterPUF(n=64, seed=11)
        #self.crps = pypuf.io.ChallengeResponseSet.from_simulation(self.puf, N=50000, seed=123)
        print("start")
        #c = self.crps.challenges
        #r = self.crps.responses
        #r_final = r[0][0]
        #print(type(r_final[0]))
        #attack = LRAttack2021(self.crps, seed=3, k=4, bs=1000, lr=.001, epochs=100)
        #attack.fit()
        ####################
        #model = attack.model
        
        # init PUF instance
        #arbiter_puf = arbiter_PUF()
        #data, data_label, attack_data = arbiter_puf.load_data(68, 5000, 11, 123,0)
        
        xor_puf = XOR_PUF()
        data, data_label, attack_data = xor_puf.load_data(68, 5000, 2, 3,6,2,7,8,9, 1, 0)
        
        data = data.astype(np.float)
        data_label = data_label.astype(np.float)
        
        # create CRPs using custom module...
        crp = pypuf.io.ChallengeResponseSet(attack_data, data_label)
        
        # attack
        attack = LRAttack2021(crp, seed=3, k=2, bs=1000, lr=.001, epochs=100)
        model, layer_output = attack.fit()
        array = layer_output.numpy()
        print("Layer output = ",array[0:5])
        
if __name__ == "__main__":
    try_attack_object = Try_attack()
        

