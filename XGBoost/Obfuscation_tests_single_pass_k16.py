#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 17:00:50 2022

@author: owen
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 23 19:18:52 2022

"""

import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
    

print("Working dir:", os.getcwd())
    
    

from pypuf.simulation import XORArbiterPUF, ArbiterPUF, XORFeedForwardArbiterPUF
import pypuf.io
from attack.lr2021 import LRAttack2021
from attack.mlp2021 import MLPAttack2021

import pandas as pd
import numpy as np

c_origin_file = '16_stage_APUF_C_Origin.txt'
c_final_file = 'APUF0utput_OWF_w_12_epsilon_a_pat_0_output_K_16.txt'



apuf = ArbiterPUF(n=16, seed=1)

puf_name = "APUF"

import csv, os
with open('obfuscation_scheme_ML_k16_results.csv' ,'w', newline='') as csvfile:   
    writer = csv.writer(csvfile)
    writer.writerow(["PUF", "Attack", "Num CRPs", "Accuracy"])        

    
            
    print("Working on: " + puf_name)
    challenge_data = pd.read_csv(c_origin_file, sep=" ", header=None)
    challenge_data.columns = ["C"]
    
    response_data = pd.read_csv(c_final_file, sep=" ", header=None)
    response_data.columns = ["R"]
    
    length = len(challenge_data.index)
    new_challenge_data = pd.DataFrame()
    split_challenges = []
    
    new_response_data = pd.DataFrame()
    split_responses = []
    rows = []
    c = []
    r = []
    
    for i in range(0, length):
        
        #Check progress
        if i % 20000 == 0:
            print(i)  
        
        line = challenge_data.iloc[i]['C']
        
        n = 256
        split_challenge = [line[i:i + n] for i in range(0, len(line), n)]
        #split_challenges = split_challenges + split_challenge
        
          
        line = str(response_data.iloc[i]['R'])
        res_len = len(line)
        if res_len < 4:
            toadd = 4 - res_len
            for j in range(0, toadd):
                line = '0' + line
                  
        n = 1
        split_response = [line[i:i + n] for i in range(0, len(line), n)]
       # split_responses = split_responses + split_response
    
      
        for j in range(0, 4):
            # row = []
            # row.append(split_challenge[j])
            # row.append(str(split_response[j]))
            # rows.append(row)
            ctemp = []
            for char in split_challenge[j]:
                if char == '0':
                    char = '-1'
                ctemp.append(np.int8(char))
                
            c.append(ctemp)
            
            for char in split_response[j]:
                if char == '0':
                    char = '-1'
                r.append(np.float64(char))  
                
     
    c = np.array(c)
    r = np.array(r)
    
    if q == 0:
        apuf = ArbiterPUF(n=256, seed=1)
        crps = pypuf.io.ChallengeResponseSet.from_simulation(apuf, N=1, seed=2)
        crps.challenges = c
        crps.responses = r
        
        post_LR_attack = LRAttack2021(crps, seed=1, k=1, bs=1000, lr=.001, epochs=100)
        post_LR_attack.fit()
        post_LR_model = post_LR_attack.model
        
        post_MLP_attack = MLPAttack2021(crps, seed=3, net=[2 ** 4, 2 ** 5, 2 ** 4], epochs=400, lr=.001, bs=1000, early_stop=.08)
        post_MLP_attack.fit()
        post_MLP_model = post_MLP_attack.model
        
        LR_acc = str(pypuf.metrics.similarity(apuf, post_LR_model, seed=4))
        MLP_acc = str(pypuf.metrics.similarity(apuf, post_MLP_model, seed=4))
        
        print('APUF Post Attack MLP: ' + MLP_acc)
        print('APUF Post Attack LR: ' + LR_acc)
        writer.writerow(["APUF_16", "LR", str(len(c)), LR_acc])
        writer.writerow(["APUF_16", "MLP", str(len(c)), MLP_acc])
        
    elif q == 1:
        xor_apuf = XORArbiterPUF(n=256, k=3, seed=1)
        crps = pypuf.io.ChallengeResponseSet.from_simulation(xor_apuf, N=1, seed=2)
        crps.challenges = c
        crps.responses = r
        
        post_LR_attack = LRAttack2021(crps, seed=1, k=3, bs=1000, lr=.001, epochs=400)
        post_LR_attack.fit()
        post_LR_model = post_LR_attack.model
        
        post_MLP_attack = MLPAttack2021(crps, seed=3, net=[2 ** 4, 2 ** 5, 2 ** 4], epochs=400, lr=.001, bs=1000, early_stop=.08)
        post_MLP_attack.fit()
        post_MLP_model = post_MLP_attack.model
        
        LR_acc = str(pypuf.metrics.similarity(xor_apuf, post_LR_model, seed=4))
        MLP_acc = str(pypuf.metrics.similarity(xor_apuf, post_MLP_model, seed=4))
        
        print('XOR_APUF Post Attack MLP: ' + MLP_acc)
        print('XOR_APUF Post Attack LR: ' + LR_acc)
        writer.writerow(["XOR_APUF_16", "LR", str(len(c)), LR_acc])
        writer.writerow(["XOR_APUF_16", "MLP", str(len(c)), MLP_acc])
        
    elif q == 2:
        ff_apuf = XORFeedForwardArbiterPUF(n=256, k=3, ff=[(4, 8)], seed=1)
        crps = pypuf.io.ChallengeResponseSet.from_simulation(ff_apuf, N=1, seed=2)
        crps.challenges = c
        crps.responses = r
        
        post_LR_attack = LRAttack2021(crps, seed=1, k=3, bs=1000, lr=.001, epochs=400)
        post_LR_attack.fit()
        post_LR_model = post_LR_attack.model
        
        post_MLP_attack = MLPAttack2021(crps, seed=3, net=[2 ** 4, 2 ** 5, 2 ** 4], epochs=400, lr=.001, bs=1000, early_stop=.08)
        post_MLP_attack.fit()
        post_MLP_model = post_MLP_attack.model
        
        LR_acc = str(pypuf.metrics.similarity(ff_apuf, post_LR_model, seed=4))
        MLP_acc = str(pypuf.metrics.similarity(ff_apuf, post_MLP_model, seed=4))
        
        print('FF_APUF Post Attack MLP: ' + MLP_acc)
        print('FF_APUF Post Attack LR: ' + LR_acc)
        writer.writerow(["FF_APUF_16", "LR", str(len(c)), LR_acc])
        writer.writerow(["FF_APUF_16", "MLP", str(len(c)), MLP_acc])



       
        
                   
                
                 




