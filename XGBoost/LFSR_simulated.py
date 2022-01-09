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
        self.N = 8000
        self.puf = pypuf.simulation.ArbiterPUF(n=self.node_num, seed=431)

    def findRandom(self):
         
        # Generate the random number
        num = random.randint(0, 1)
     
        # Return the generated number
        return num
     
    # Function to generate a random
    # binary string of length N
    def generateCRPs(self):
         
        '''# Stores the empty string
        challenge_string = ""
        challenge_r = 0
     
        # Iterate over the range [0, N - 1]
        for i in range(N):
             
            # Store the random number
            x = self.findRandom()
     
            # Append it to the string
            challenge_string += str(x)
        
        challenge_r = int(challenge_string.zfill(N))
        # Print the resulting string
        print(challenge_string.zfill(N))'''
        
        challengesConfig = random_inputs(n=self.node_num+4, N=10, seed=2)
        challenges = []
        obfuscate_config = []
        for i in range(len(challengesConfig)):
            challenges.append(challengesConfig[i][4:])
            obfuscate_config.append(challengesConfig[i][0:4])
        
        challenges = np.array(challenges)
        obfuscate_config = np.array(obfuscate_config)
        responses = self.puf.eval(challenges)
        crps = pypuf.io.ChallengeResponseSet(challenges, responses)
        
        return crps, obfuscate_config

    def obfuscateChallenge(self):
        state = [0,0,0,1,0]
        fpoly = [5,4]
        L = LFSR(fpoly=fpoly,initstate=state, verbose=True)
        L.runKCycle(11)
        print(L.seq)
        L.info()
        
        
        base = random.randrange(10, 1000)
        print(base)
        crps, obfuscate_config = self.generateCRPs()
        binary_obfus = list(obfuscate_config[0])
        binary_obfus = [0 if c == -1 else 1 for c in binary_obfus]
        binary_obfus_str = ''.join(str(x) for x in binary_obfus)
        
        print(int(binary_obfus_str, 2))
        
    
if __name__ == "__main__":
    LSFR_object = LSFR_simulated()
    LSFR_object.generateCRPs()
    LSFR_object.obfuscateChallenge()
