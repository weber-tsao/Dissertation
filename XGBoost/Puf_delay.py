# -*- coding: utf-8 -*-
"""
Created on Thu Nov  4 23:40:37 2021

@author: weber
"""
import pypuf.simulation
import pypuf.io
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from numpy import array, ones
from sklearn import preprocessing
from LFSR_simulated import*
from pypuf.simulation import XORArbiterPUF, XORFeedForwardArbiterPUF, LightweightSecurePUF, InterposePUF

class Puf:
    def __init__(self):
        self.total_bits_num = 68
        self.N = 7200
        self.puf = pypuf.simulation.ArbiterPUF(n=(self.total_bits_num-4), seed=12)
        #self.puf = XORArbiterPUF(n=(self.total_bits_num-4), k=2, seed=21)
        #self.puf = XORFeedForwardArbiterPUF(n=(self.total_bits_num-4), k=6, ff=[(32,60)], seed=1)
        #self.puf = LightweightSecurePUF(n=(self.total_bits_num-4), k=5, seed=10)
        #self.puf = InterposePUF(n=(self.total_bits_num-4), k_up=3, k_down=3, seed=12)
        self.lfsrChallenges = random_inputs(n=self.total_bits_num, N=self.N, seed=177) # LFSR random challenges data
        #self.xorChallenges = random_inputs(n=self.total_bits_num, N=7200, seed=134) 
        #self.zeroArray = list(np.zeros(1))
        #self.crp = pypuf.io.ChallengeResponseSet.from_simulation(self.puf, N=self.N, seed=34)
        #self.crp.save('crps.npz')
        #self.crp_loaded = pypuf.io.ChallengeResponseSet.load('crps.npz')
        self.mux_node = []
        self.top_path = []
        self.bottom_path = []
        self.LFSR_simulated = LFSR_simulated()
        self.lfsr = True
        self.diff2 = []
        self.diff3 = []
        self.diff4 = []
        self.diff_index = 0
    
    def puf_path(self, challenge):
        challenge = array([challenge])
        self.mux_node = [x for x in range(len(challenge[0])*2)]
        self.top_path = []
        self.bottom_path = []
        prev = 0
        for i in range(len(challenge[0])):
            count = i*2
            if challenge[0][i] == 1:
                if prev%2 != 0:
                    self.top_path.append(self.mux_node[count+1])
                    self.bottom_path.append(self.mux_node[count])
                    prev = self.mux_node[count+1]
                else:
                    self.top_path.append(self.mux_node[count])
                    self.bottom_path.append(self.mux_node[count+1])
                    prev = self.mux_node[count]
            elif challenge[0][i] == -1:
                if prev%2 != 0:
                    self.top_path.append(self.mux_node[count])
                    self.bottom_path.append(self.mux_node[count+1])
                    prev = self.mux_node[count]
                else:
                    self.top_path.append(self.mux_node[count+1])
                    self.bottom_path.append(self.mux_node[count])
                    prev = self.mux_node[count+1]

        return self.top_path, self.bottom_path
    
    def total_delay_diff(self, challenge):
        challenge = array([challenge])
        last_stage_ind = len(challenge[0])-1
        puf_delay = pypuf.simulation.LTFArray(weight_array=self.puf.weight_array[:, :last_stage_ind], bias=None, transform=self.puf.transform)
        stage_delay_diff = puf_delay.val(challenge[:, :last_stage_ind])

        return stage_delay_diff
    
    def concat(self, a, b):
        return int(f"{a}{b}")

    def load_data(self):
        data_len = self.N
        train_data = []
        train_label = []
        data = []
        #xortrain_data = []
        #xortrain_label = []
        #xordata = []
        
        if self.lfsr:
            test_crps = self.lfsrChallenges
            #xor_test_crps = self.xorChallenges
        else:
            test_crps = self.crp_loaded
        
        
        for i in range(data_len):
            ### data ###
            if self.lfsr: 
                challenge = test_crps[i]
                #xorchallenge = xor_test_crps[i]
                
                # obfuscate part
                obfuscateChallenge = self.LFSR_simulated.createObfuscateChallenge(challenge)
                obfuscateChallenge = [-1 if c == 0 else c for c in obfuscateChallenge]
                
                #xorobfuscateChallenge = self.LFSR_simulated.createObfuscateChallenge(xorchallenge)
                #xorobfuscateChallenge = [-1 if c == 0 else c for c in xorobfuscateChallenge]
                
                final_delay_diff = self.total_delay_diff(obfuscateChallenge)
                #final_delay_diff = self.pufx.val(np.array([obfuscateChallenge]))
                
                ### For interpose PUF
                #final_delay_diff_up = self.puf.up.val(np.array([obfuscateChallenge]))
                
                ### For general model
                '''if i <= 32000:
                    final_delay_diff= self.puf2.val(np.array([obfuscateChallenge]))
                    self.diff2.append(final_delay_diff)
                    self.diff_index = 2
                elif i > 32000 and i <= 64000:
                    final_delay_diff= self.puf3.val(np.array([obfuscateChallenge]))
                    self.diff3.append(final_delay_diff)
                    self.diff_index = 3
                else:
                    final_delay_diff= self.puf4.val(np.array([obfuscateChallenge]))
                    self.diff4.append(final_delay_diff)
                    self.diff_index = 4'''
                
                
                '''count=0
                for p in self.puf.simulations:
                    count+=1
                    #print(p, p.val(np.array([obfuscateChallenge])))
                print(count)'''
                
                top_path, bottom_path = self.puf_path(obfuscateChallenge)
                #xortop_path, xorbottom_path = self.puf_path(xorobfuscateChallenge)
            else:
                challenge = list(test_crps[i][0])
                final_delay_diff = self.total_delay_diff(test_crps[i][0])
                top_path, bottom_path = self.puf_path(test_crps[i][0])
                
            challenge = [0 if c == -1 else c for c in challenge]
            #xorchallenge = [0 if c == -1 else c for c in xorchallenge]
            
            '''### challenge to parity vector
            parity_vector = []
            temp = 0
            for j in range(len(challenge)):
                if j == 0:
                    temp = 1-2*challenge[0]
                else:
                    for x in range(j):
                        temp *= (1-2*challenge[x])
                parity_vector.append(temp)
            parity_vector = [0 if c == -1 else 1 for c in parity_vector]'''
            
            
            ### label ###
            if self.lfsr:
                '''if self.diff_index == 2:
                    response = self.LFSR_simulated.produceObfuscateResponse(self.puf2, obfuscateChallenge)
                elif self.diff_index == 3:
                    response = self.LFSR_simulated.produceObfuscateResponse(self.puf3, obfuscateChallenge)
                else:
                    response = self.LFSR_simulated.produceObfuscateResponse(self.puf4, obfuscateChallenge)'''
                
                response = self.LFSR_simulated.produceObfuscateResponse(self.puf, obfuscateChallenge)
                response = np.array(response)
                data_r = 0
                if response == -1:
                    data_r = 0
                else:
                    data_r = 1
                    
                '''xorresponse = self.LFSR_simulated.produceObfuscateResponse(self.pufx, xorobfuscateChallenge)
                xorresponse = np.array(xorresponse)
                xordata_r = 0
                if xorresponse == -1:
                    xordata_r = 0
                else:
                    xordata_r = 1'''
            else:
                response = test_crps[i][1]
                response = response[0]
                
                data_r = 0
                if response == -1:
                    data_r = 0
                else:
                    data_r = 1
            
            '''midpoint = (self.total_bits_num-4)//2+1
            obfuscateChallenge = obfuscateChallenge[0:midpoint] + [response[0]] + obfuscateChallenge[midpoint:] 
            final_delay_diff_down = self.puf.down.val(np.array([obfuscateChallenge]))
            final_delay_diff = final_delay_diff_up+final_delay_diff_down'''
            #data.append(self.zeroArray+challenge+top_path+bottom_path+[data_r])
            data.append([final_delay_diff[0]]+top_path+bottom_path+challenge+[data_r])
            #xordata.append([xorfinal_delay_diff[0]]+xortop_path+xorbottom_path+xorchallenge+[xordata_r])
            #data.append(parity_vector)
        
        data = np.array(data)
        #xordata = np.array(xordata)
        
        #min_max_scaler = preprocessing.MinMaxScaler()
        #data = min_max_scaler.fit_transform(data)
        data = np.unique(data,axis=0)
        train_label = data[:,-1]
        train_data = data[:,0:-1]
        #xordata = np.unique(xordata,axis=0)
        #xortrain_label = xordata[:,-1]
        #xortrain_data = xordata[:,0:-1]
        #plt.hist(train_label, bins=50)
        #plt.show()
        '''num = 0
        for i in range(len(train_label)):
            if train_label[i] == 1:
                num += 1
        print(num)'''
            
        
        return train_data, train_label
        