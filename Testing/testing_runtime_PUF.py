import pypuf.simulation
import pypuf.io
import timeit
import numpy
from numpy import array, ones
'''
puf = pypuf.simulation.ArbiterPUF(n=64, seed=1)

crp = pypuf.io.ChallengeResponseSet.from_simulation(puf, N=10, seed=2)
crp.save('crps.npz')
crp_loaded = pypuf.io.ChallengeResponseSet.load('crps.npz')
print(crp_loaded[0])
print(crp_loaded[1])

start = timeit.default_timer()
puf.eval(array([crp_loaded[0][0]]))
stop = timeit.default_timer()
print('Time: ', stop - start)
print(puf.eval(array([crp_loaded[0][0]])))

start = timeit.default_timer()
puf.eval(array([crp_loaded[1][0]]))
stop = timeit.default_timer()
print('Time: ', stop - start)
print(puf.eval(array([crp_loaded[1][0]])))'''

# Calculate delay difference
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

print(abs(x[0])-abs(y[0]))