from isort import file
import pandas as pd
import pypuf.simulation, pypuf.io
from pypuf.simulation import ArbiterPUF
from pypuf.io import random_inputs
import numpy as np
import sys
import os



K = [16, 32]

Num = 180000
for k in K:
    path = os.path.join(os.getcwd(), str(k))
    print(path)
    #make txt file K
    file = open(path,"a")
    all_response =[]
    for x in range(Num):
        x+=1
        puf = ArbiterPUF(n=k, seed=1 )
        crps = pypuf.io.ChallengeResponseSet.from_simulation(puf, N=64, seed = k+x)
       # print(crps)

        full_response = []
        
        
        for response in crps.responses:
            if  response == -1:
                response = 0
            else:
                response = 1

                
            full_response.append(response)

        all_response.append(str(full_response))
      #  all_response = "".join(all_response)

    for response_line in all_response:
        
        line = str(response_line).replace('[', '')        
        line = line.replace(']', '')
        line = line.replace(',', '')
        line = line.replace(' ', '')

        file.write(line)
        file.write("\n")
    
    

        #add full response as line to txt file K
        print(full_response)
        print(len(full_response))

            
                

        #for each response in crps, read the value append in new array.

