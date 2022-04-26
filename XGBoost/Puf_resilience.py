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
from LFSR_simulated import*

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
    
    def dec_to_bin(self, x):
        
        return int(bin(x)[2:])
    
    def switch_complement(self, challenge):
        challenge = np.array(challenge)
        chal_len = len(challenge)
        half_chal_len = int(chal_len/2)
        firsthalf_challenge = []
        secondhalf_challenge = []
        firsthalf_challenge.append(challenge[0:half_chal_len])
        secondhalf_challenge.append(challenge[half_chal_len:])
        switch_challenge = np.concatenate((secondhalf_challenge, firsthalf_challenge),axis=None)
        
        
        '''binary_obfus = list(switch_challenge)
        binary_obfus = [0 if c == -1 else c for c in binary_obfus]
        binary_obfus_str = ''.join(str(x) for x in binary_obfus)
        decimal = int(binary_obfus_str, 2)
        complement_decimal = ~decimal
        print(complement_decimal)
        #complement_challenge = self.dec_to_bin(complement_decimal)
        complement_challenge= "{0:b}".format(complement_decimal)'''
        
        binary_obfus = list(switch_challenge)
        binary_obfus = [0 if c == -1 else c for c in binary_obfus]
        
        final_challenge = []
        
        for i in range(len(binary_obfus)):
            if binary_obfus[i] == 1:
                final_challenge.append(0)
            else:
               final_challenge.append(1)
        
        return np.array(final_challenge)

    def cyclic_shift(self, challenge, puf):   
        ##LFSR
        self.LFSR_simulated = LFSR_simulated()
        alterChallenge = self.LFSR_simulated.createObfuscateChallenge(challenge, 1)
        alterChallenge = [-1 if c == 0 else c for c in alterChallenge]
        
        ##Another idea
        #alterChallenge = self.switch_complement(challenge)
        #alterChallenge = [-1 if c == 0 else c for c in alterChallenge]
        
        obfuscate_Challenge = self.self_feedback(alterChallenge, puf)
        obchal_len = len(obfuscate_Challenge)
        l = int (np.log2(obchal_len))
        m = 0
        for i in range(l):
            m = m + obfuscate_Challenge[i]*np.power(2,i)
        obfuscate_Challenge = np.roll(obfuscate_Challenge,m)
        
        return obfuscate_Challenge
    
'''if __name__ == "__main__":
    puf_resilience_object = Puf_resilience()
    puf = pypuf.simulation.ArbiterPUF(n=4, seed=21)
    x = puf_resilience_object.cyclic_shift([1,-1,-1,-1,1,1,1,1], puf)
    print(x)
    
    #y = puf_resilience_object.switch_complement([-1,-1,-1,-1,1,1,1,1])
    #print(y)
    '''
    