# -*- coding: utf-8 -*-
"""
Created on Thu Nov  4 23:40:37 2021

@author: weber
"""
import random

class LSFR_simulated:
    def __init__(self):
        self.N = 8000

    def findRandom(self):
         
        # Generate the random number
        num = random.randint(0, 1)
     
        # Return the generated number
        return num
     
    # Function to generate a random
    # binary string of length N
    def generateBinaryString(self, N):
         
        # Stores the empty string
        S = ""
     
        # Iterate over the range [0, N - 1]
        for i in range(N):
             
            # Store the random number
            x = self.findRandom()
     
            # Append it to the string
            S += str(x)
         
        # Print the resulting string
        print(S)
    
if __name__ == "__main__":
    LSFR_object = LSFR_simulated()
    LSFR_object.generateBinaryString(7)
