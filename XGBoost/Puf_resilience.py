# -*- coding: utf-8 -*-
"""
Created on Thu Nov  4 23:40:37 2021

@author: weber
"""
import pypuf.simulation
from pypuf.io import random_inputs
import pypuf.io
import random
import numpy as np
from numpy import array, ones
import numpy as np
from pylfsr import LFSR

class Puf_resilience:
    
    def self_feedback(self, challenge, puf):
        challenge = np.array(challenge)
        eval_challenge = np.array([challenge])
        chal_len = len(challenge)
        response = puf.eval(eval_challenge)
        challenge = list(challenge)
        challenge = [0 if c == -1 else c for c in challenge]
        if response[0] == -1:
            for i in range(int(chal_len/2)):
                challenge[2*i] = challenge[2*i]^1
        else:
            for i in range(int(chal_len/2)):
                challenge[(2*i+1)] = challenge[(2*i+1)]^1
        return challenge

    def cyclic_shift(self, challenge, puf):     
        obfuscate_Challenge = self.self_feedback(challenge, puf)
        obchal_len = len(obfuscate_Challenge)
        l = int (np.log2(obchal_len))
        m = 0
        for i in range(l):
            m = m + obfuscate_Challenge[i]*np.power(2,i)
        obfuscate_Challenge = np.roll(obfuscate_Challenge,m)
        return obfuscate_Challenge
    
'''if __name__ == "__main__":
    puf_resilience_object = Puf_resilience()
    puf = pypuf.simulation.ArbiterPUF(n=8, seed=21)
    x = puf_resilience_object.cyclic_shift([1,-1,-1,-1,-1,-1,1,1], puf)'''
    