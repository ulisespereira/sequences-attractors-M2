import numpy as np
import pickle
from classes.network_2areas import *
import matplotlib.pyplot as plt
'''define modelparameters and simulation functions'''

modelparams = dict(
    seed = 2, #random seed connectivity
    dt = .5, #integration time
    tau_m2 = 20.,#ms
    tau_th = 2.,#ms
    T = 100, # period simulations [ms] 
    c = 0.1, # dilution 
    N = 10000,  # number of neurons
    N_th = 1000,  # number of neurons
    p = 10, # number of patterns
    amp_ff = .65,#0.6,# strenght of the feedforward weights
    std_low_dim = .65,#standard deviation of the low rank noise
    std_priv = 0., # standard deviation private noise
    
    #learning rule
    amp_median = 3., # amplitude connectivity
    bf = 1e8, #  step function f and g
    bg = 1e8, #  step function f and g
    qf = .65,# offset f
    xf = 1.7, # threshold f
    xg = 1.7, # threshold f

    #amplitude ff and fb
    w_ctx_ctx = 1,
    w_th_ctx = 1,
    w_ctx_th = .1,
    
    indexes = 10000, # number of recorded neurons
    slope_tf = 1.,
    intercept_tf = 1.
)

def simulate_network(modelparams):

    #CREATING TRANSFER FUNCTION
    folder  = 'data/'
    param_tf = pickle.load(open(folder+'parametersHetero.p','rb'))

    #CREATING CONNECTIVITY
    conn = ConnectivityMatrices(param_tf, modelparams)

   # #CREATING DYNAMICS
   # indexes = range(modelparams['indexes'])#range(10000)
   # np.random.seed()
    dyn = NetworkDynamics(modelparams, conn)

   # #SIMULATING NETWORK DYNAMICS
   # dyn.neu_indexes = indexes
    u_ctx0 = conn.patterns['ctx'][0] 
    u_th0 = conn.patterns['th'][0]
    print(u_ctx0.shape, u_th0.shape)
    dynamics = dyn.dynamics(u_ctx0, u_th0)
    
    return dynamics

dyn = simulate_network(modelparams)
plt.plot(dyn['overlaps_ctx'])
plt.show()
