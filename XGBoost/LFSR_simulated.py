# -*- coding: utf-8 -*-
"""
Created on Thu Nov  4 23:40:37 2021

@author: weber
"""
import pypuf.simulation
from pypuf.io import random_inputs
import pypuf.io
import random

class LSFR_simulated:
    def __init__(self):
        self.node_num = 128
        self.N = 8000
        self.puf = pypuf.simulation.ArbiterPUF(n=self.node_num, seed=431)

    def findRandom(self):
         
        # Generate the random number
        num = random.randint(0, 1)
     
        # Return the generated number
        return num
     
    # Function to generate a random
    # binary string of length N
    def generateBinaryString(self, N):
         
        # Stores the empty string
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
        print(challenge_string.zfill(N))
        print(random_inputs(n=64, N=3, seed=2))
        #print(self.puf.eval([11]))
    
if __name__ == "__main__":
    LSFR_object = LSFR_simulated()
    LSFR_object.generateBinaryString(7)
