# -*- coding: utf-8 -*-
"""
Created on Fri Jul  1 10:26:05 2022

@author: Asus
"""

import pypuf.simulation, pypuf.io
from attack import LRAttack2021



class Try_attack:
    
    def __init__(self):
        self.puf = pypuf.simulation.XORArbiterPUF(n=64, k=4, seed=1)
        #self.puf = pypuf.simulation.ArbiterPUF(n=64, k=1, seed=1)
        self.crps = pypuf.io.ChallengeResponseSet.from_simulation(self.puf, N=50000, seed=2)
        print("start")
        attack = LRAttack2021(self.crps, seed=3, k=4, bs=1000, lr=.001, epochs=100)
        attack.fit()
        ####################
        model = attack.model
        
        
        
if __name__ == "__main__":
    try_attack_object = Try_attack()
        

