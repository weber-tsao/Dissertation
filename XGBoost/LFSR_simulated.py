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

class LSFR_simulated:
    def __init__(self):
        self.node_num = 5
        self.N = 10
        self.puf = pypuf.simulation.ArbiterPUF(n=self.node_num, seed=431)
        #self.challengesConfig = random_inputs(n=self.node_num+4, N=10, seed=2)
        #print(self.challengesConfig)

    def splitCRPs(self, challenge):
        spited_challenge = []
        obfuscate_bits = []
        spited_challenge.append(challenge[4:])
        obfuscate_bits.append(challenge[0:4])
        
        spited_challenge = np.array(spited_challenge[0])
        obfuscate_bits = np.array(obfuscate_bits[0])

        return spited_challenge, obfuscate_bits

    def createShiftCount(self, obfuscate_bits):     
        # create random base number
        base = random.randrange(2, 1000)
        
        # create count by looking at crp and splited crp bits
        binary_obfus = list(obfuscate_bits)
        binary_obfus = [0 if c == -1 else 1 for c in binary_obfus]
        binary_obfus_str = ''.join(str(x) for x in binary_obfus)
        count = int(binary_obfus_str, 2)
        
        # calculate shift_count
        shift_count = base*count
        
        return shift_count
        
    def createObfuscateChallenge(self, challenge):
        spited_challenge, obfuscate_bits = self.splitCRPs(challenge)
        shift_count = self.createShiftCount(obfuscate_bits)
        
        challenge_state = [0 if c == -1 else 1 for c in list(spited_challenge)]
        fpoly = [5,3] # look at the optimize table to determine
        #print(challenge_state)
        L = LFSR(fpoly=fpoly, initstate=challenge_state, verbose=False)
        L.runKCycle(shift_count)
        print(L.state)
        L.info()
        
        return L.state
    
    def produceObfuscateResponse(self, puf, obfuscate_Challenge):
        response = puf.eval(np.array([obfuscate_Challenge]))
        print(response)
        return response
    
if __name__ == "__main__":
    LSFR_object = LSFR_simulated()
    challengesConfig = random_inputs(n=9, N=10, seed=2)
    obfuscate_Challenge = LSFR_object.createObfuscateChallenge(challengesConfig[0])
    puf = pypuf.simulation.ArbiterPUF(n=5, seed=431)
    LSFR_object.produceObfuscateResponse(puf, obfuscate_Challenge)
