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

apuf = ArbiterPUF(n=32, seed=1)
xor_apuf = XORArbiterPUF(n=32, k=3, seed=1)
ff_apuf = XORFeedForwardArbiterPUF(n=32, k=3, ff=[(8, 16)], seed=1)


# #Load PUF CRPs
import pypuf.io

apuf_crps = pypuf.io.ChallengeResponseSet.from_simulation(apuf, N=10000, seed=2)
xor_apuf_crps = pypuf.io.ChallengeResponseSet.from_simulation(xor_apuf, N=100000, seed=2)
ff_apuf_crps = pypuf.io.ChallengeResponseSet.from_simulation(ff_apuf, N=100000, seed=2)


# #Train LR Models
from attack.lr2021 import LRAttack2021
from attack.mlp2021 import MLPAttack2021

apuf_LR_attack = LRAttack2021(apuf_crps, seed=3, k=1, bs=1000, lr=.001, epochs=1000)
apuf_LR_attack.fit()
apuf_LR_model = apuf_LR_attack.model

xor_apuf_LR_attack = LRAttack2021(xor_apuf_crps, seed=3, k=3, bs=1000, lr=.001, epochs=1000)
xor_apuf_LR_attack.fit()
xor_apuf_LR_model = xor_apuf_LR_attack.model

ff_apuf_LR_attack = LRAttack2021(ff_apuf_crps, seed=3, k=3, bs=1000, lr=.001, epochs=1000)
ff_apuf_LR_attack.fit()
ff_apuf_LR_model = ff_apuf_LR_attack.model


#Train MLP Models

apuf_MLP_attack = MLPAttack2021(apuf_crps, seed=3, net=[2 ** 4, 2 ** 5, 2 ** 4], epochs=1000, lr=.001, bs=1000, early_stop=.08)
apuf_MLP_attack.fit()
apuf_MLP_model = apuf_MLP_attack.model

xor_apuf_MLP_attack = MLPAttack2021(xor_apuf_crps, seed=3, net=[2 ** 4, 2 ** 5, 2 ** 4], epochs=1000, lr=.001, bs=1000, early_stop=.08)
xor_apuf_MLP_attack.fit()
xor_apuf_MLP_model = xor_apuf_MLP_attack.model


ff_apuf_MLP_attack = MLPAttack2021(ff_apuf_crps, seed=3, net=[2 ** 4, 2 ** 5, 2 ** 4], epochs=1000, lr=.001, bs=1000, early_stop=.08)
ff_apuf_MLP_attack.fit()
ff_apuf_MLP_model = ff_apuf_MLP_attack.model

#Metrics

import pypuf.metrics
print('LR Attacks\n')
print('APUF: ' + str(pypuf.metrics.similarity(apuf, apuf_LR_model, seed=4)))
print('XOR_APUF: ' + str(pypuf.metrics.similarity(xor_apuf, xor_apuf_LR_model, seed=4)))
print('FF_APUF: ' + str(pypuf.metrics.similarity(ff_apuf, ff_apuf_LR_model, seed=4)) + '\n')

print('MLP Attacks\n')
print('APUF: ' + str(pypuf.metrics.similarity(apuf, apuf_MLP_model, seed=4)))
print('XOR_APUF: ' + str(pypuf.metrics.similarity(xor_apuf, xor_apuf_MLP_model, seed=4)))
print('FF_APUF: ' + str(pypuf.metrics.similarity(ff_apuf, ff_apuf_MLP_model, seed=4)))

print(apuf_crps.challenges.shape)


import pandas as pd
import numpy as np

c_origin_list = ['32_stage_APUF_C_Origin.txt', '32_stage_3XOR_APUF_C_Origin.txt', '32_stage_3XOR_FFPUF_C_Origin.txt']
r_final_list = ['APUF0utput_OWF_w_12_epsilon_a_pat_0_output_K_32.txt', '3XORAPUF0utput_all_w_12_epsilon_a_pat_0_output_K_32.txt', '3XOR_FF_PUF0utput_all_w_12_epsilon_a_pat_0_output_K_32.txt']
apuf = ArbiterPUF(n=32, seed=1)
xor_apuf = XORArbiterPUF(n=32, k=3, seed=2)
ff_apuf = XORFeedForwardArbiterPUF(n=32, k=3, ff=[(4, 8)], seed=1)

puf_names = ["APUF", "XOR_APUF", "FF_APUF"]

import csv, os
with open('obfuscation_scheme_ML_k32_results.csv' ,'w', newline='') as csvfile:   
    writer = csv.writer(csvfile)
    writer.writerow(["PUF", "Attack", "Num CRPs", "Accuracy"])        
        
    for q in range(0, 3):
            
        print("Working on: " + puf_names[q])
        challenge_data = pd.read_csv(c_origin_list[q], sep=" ", header=None)
        challenge_data.columns = ["C"]
        
        response_data = pd.read_csv(r_final_list[q], sep=" ", header=None)
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
            
            n = 1024
            split_challenge = [line[i:i + n] for i in range(0, len(line), n)]
            #split_challenges = split_challenges + split_challenge
            
              
            line = str(response_data.iloc[i]['R'])
            res_len = len(line)
            if res_len < 2:
                toadd = 2 - res_len
                for j in range(0, toadd):
                    line = '0' + line
                      
            n = 1
            split_response = [line[i:i + n] for i in range(0, len(line), n)]
           # split_responses = split_responses + split_response
        
          
            for j in range(0, 2):
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
            apuf = ArbiterPUF(n=1024, seed=1)
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
            writer.writerow(["APUF_32", "LR", str(len(c)), LR_acc])
            writer.writerow(["APUF_32", "MLP", str(len(c)), MLP_acc])
            
        elif q == 1:
            xor_apuf = XORArbiterPUF(n=1024, k=3, seed=1)
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
            writer.writerow(["XOR_APUF_32", "LR", str(len(c)), LR_acc])
            writer.writerow(["XOR_APUF_32", "MLP", str(len(c)), MLP_acc])
            
        elif q == 2:
            ff_apuf = XORFeedForwardArbiterPUF(n=1024, k=3, ff=[(4, 8)], seed=1)
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
            writer.writerow(["FF_APUF_32", "LR", str(len(c)), LR_acc])
            writer.writerow(["FF_APUF_32", "MLP", str(len(c)), MLP_acc])
    
    
    
           
            
                       
                    
                     




