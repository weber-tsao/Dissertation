# -*- coding: utf-8 -*-
"""
Created on Thu Nov  4 23:40:37 2021

@author: weber
"""
import pypuf.simulation
import pypuf.io
#import numpy
#from numpy import array, ones

puf = pypuf.simulation.ArbiterPUF(n=3, seed=1)
crp = pypuf.io.ChallengeResponseSet.from_simulation(puf, N=2, seed=2)
crp.save('crps.npz')
crp_loaded = pypuf.io.ChallengeResponseSet.load('crps.npz')
print(crp_loaded[0])
challenge = array([crp_loaded[0][0]])
print(puf.val(challenge))

# 1
puf_delay = pypuf.simulation.LTFArray(weight_array=puf.weight_array[:, :1], bias=None, transform=puf.transform)
x1 = puf_delay.val(challenge[:, :1])  # delay diff. after the th stage
print(x1)

# 2
puf_delay = pypuf.simulation.LTFArray(weight_array=puf.weight_array[:, :2], bias=None, transform=puf.transform)
x2 = puf_delay.val(challenge[:, :2])  # delay diff. after the th stage
print(x2)

# 3
puf_delay = pypuf.simulation.LTFArray(weight_array=puf.weight_array[:, :3], bias=None, transform=puf.transform)
x3 = puf_delay.val(challenge[:, :3])  # delay diff. after the th stage
print(x3)

'''# Calculate delay difference
puf = pypuf.simulation.ArbiterPUF(n=5, seed=1)
challenges = pypuf.io.random_inputs(n=5, N=1, seed=2)
puf.val(challenges) # delay difference after the last stage
print(challenges)
print(puf.val(challenges))
#print(puf.weight_array[:, :2])
puf14 = pypuf.simulation.LTFArray(weight_array=puf.weight_array[:, :1], bias=None, transform=puf.transform)
x1 = puf14.val(challenges[:, :1])  # delay diff. after the 14th stage
print(x1)
puf14 = pypuf.simulation.LTFArray(weight_array=puf.weight_array[:, :2], bias=None, transform=puf.transform)
x = puf14.val(challenges[:, :2])  # delay diff. after the 14th stage
print(x)
puf13 = pypuf.simulation.LTFArray(weight_array=puf.weight_array[:, :3], bias=None, transform=puf.transform)
y = puf13.val(challenges[:, :3])
print(y)

print(abs(x[0])-abs(y[0]))'''