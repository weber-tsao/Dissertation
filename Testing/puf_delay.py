# -*- coding: utf-8 -*-
"""
Created on Thu Nov  4 23:40:37 2021

@author: weber
"""
import pypuf.simulation
import pypuf.io
import numpy
from numpy import array, ones

puf = pypuf.simulation.ArbiterPUF(n=3, seed=1)
crp = pypuf.io.ChallengeResponseSet.from_simulation(puf, N=2, seed=2)
crp.save('crps.npz')
crp_loaded = pypuf.io.ChallengeResponseSet.load('crps.npz')
print(crp_loaded[0])
challenge = array([crp_loaded[0][0]])
print(len(challenge[0]))
#print(puf.val(challenge))

#----------------------------------------------------

# 0
puf_delay = pypuf.simulation.LTFArray(weight_array=puf.weight_array[:, :0], bias=None, transform=puf.transform)
x0 = puf_delay.val(challenge[:, :0])  # delay diff. after the 0th stage
print(x0)

# 1
puf_delay = pypuf.simulation.LTFArray(weight_array=puf.weight_array[:, :1], bias=None, transform=puf.transform)
x1 = puf_delay.val(challenge[:, :1])  # delay diff. after the 1th stage
print(x1)

# 2
puf_delay = pypuf.simulation.LTFArray(weight_array=puf.weight_array[:, :2], bias=None, transform=puf.transform)
x2 = puf_delay.val(challenge[:, :2])  # delay diff. after the 2th stage
print(x2)

# 3
puf_delay = pypuf.simulation.LTFArray(weight_array=puf.weight_array[:, :3], bias=None, transform=puf.transform)
x3 = puf_delay.val(challenge[:, :3])  # delay diff. after the 3th stage
print(x3)

print(abs(x3[0])-abs(x2[0]))
print(abs(x2[0])-abs(x1[0]))
print(abs(x1[0])-abs(x0[0]))