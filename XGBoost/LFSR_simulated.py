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

class LFSR_simulated:
    
    def splitCRPs(self, challenge):
        challenge = np.array(challenge)
        spited_challenge = []
        obfuscate_bits = []
        spited_challenge.append(challenge[4:])
        obfuscate_bits.append(challenge[0:4])
        
        spited_challenge = np.array(spited_challenge[0])
        obfuscate_bits = np.array(obfuscate_bits[0])

        return spited_challenge, obfuscate_bits

    def createShiftCount(self, obfuscate_bits, cus_base):     
        # create random base number
        base = cus_base
        
        # create count by looking at crp and splited crp bits
        binary_obfus = list(obfuscate_bits)
        binary_obfus = [0 if c == -1 else c for c in binary_obfus]
        binary_obfus_str = ''.join(str(x) for x in binary_obfus)
        count = int(binary_obfus_str, 2)
        
        # calculate shift_count
        shift_count = base*count
        
        return shift_count
        
    def createObfuscateChallenge(self, challenge, cus_base):
        spited_challenge, obfuscate_bits = self.splitCRPs(challenge)
        shift_count = self.createShiftCount(obfuscate_bits, cus_base)
        
        challenge_state = [0 if c == -1 else 1 for c in list(spited_challenge)]
        fpoly = [3,4]
        #fpoly = [64,63,61,60]
        L = LFSR(fpoly=fpoly, initstate=challenge_state, verbose=False)
        L.runKCycle(shift_count)
        #print(L.state)
        #L.info()
        #result  = L.test_properties(verbose=2)
        
        return L.state
    
    def produceObfuscateResponse(self, puf, obfuscate_Challenge):
        response = puf.eval(np.array([obfuscate_Challenge]))
        
        return response
    
'''if __name__ == "__main__":
    LFSR_object = LFSR_simulated()
    #challengesConfig = random_inputs(n=9, N=10, seed=2)
    #print(challengesConfig[0])
    obfuscate_Challenge = LFSR_object.createObfuscateChallenge([1,1,1,0,0,1,0,1,0,1,1,1,1,1,1,1])
    #puf = pypuf.simulation.ArbiterPUF(n=5, seed=21)
    #LFSR_object.produceObfuscateResponse(puf, obfuscate_Challenge)'''
