import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np
import time
import random
from sklearn.metrics import f1_score
from collections import defaultdict

import pypuf.simulation
import pypuf.io
'''puf = pypuf.simulation.ArbiterPUF(n=10, seed=1)
crp = pypuf.io.ChallengeResponseSet.from_simulation(puf, N=500, seed=2)
crp.save('crps.npz')
#print(crp[0][1])'''

num_nodes = 20
num_feats = 10
feat_data = np.zeros((num_nodes, num_feats))
labels = np.empty((num_nodes, 1), dtype=np.int64)
print(labels)
node_map = {}
label_map = {}

crp_loaded = pypuf.io.ChallengeResponseSet.load('crps.npz')
y = crp_loaded[0][0]
print()

'''for i in range(num_nodes):
    feat_data[i, :] = [float(x) for x in info[1:-1]]

    #node_map[crp_loaded[0]] = i
    if not info[-1] in label_map:
        label_map[info[-1]] = len(label_map)
    labels[i] = label_map[info[-1]]
    # print(info[-1])'''

adj_lists = defaultdict(set)
for i in range(0, 6-1, 2):
    node1 = i+1
    node2 = i+2
    adj_lists[node1].add(i+3)
    adj_lists[node1].add(i+4)
    adj_lists[node2].add(i+3)
    adj_lists[node2].add(i+4)

print(adj_lists)