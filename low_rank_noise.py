import numpy as np
from classes.learning_rule import *
from classes.connectivity import *
from classes.transfer_function import *
from classes.network_dynamics import *
import pickle 
import matplotlib.pyplot as plt
''' 


'''

modelparams = dict(
    seed = 2, #random seed connectivity
    dt = 4., #integration time
    T = 20000, # period simulations [ms] 
    c = 0.1, # dilution 
    N = 10000,  # number of neurons
    p = 10, # number of patterns
    amp_ff = .65,#0.6,# strenght of the feedforward weights
    std_low_dim = .65,#standard deviation of the low rank noise
    std_priv = 0., # standard deviation private noise
    amp_median = 3., # amplitude connectivity
    bf = 1e8, #  step function f and g
    qf = .65,# offset f
    xf = 1.7 # threshold f

)

def simulate_network():

    #CREATING TRANSFER FUNCTION
    non_param_transfer_functions = pickle.load(open('data/nonparam_tf.p','rb'))
    param_transfer_functions = pickle.load(open('data/parametersHetero.p','rb'))

    tf_sig = TransferFunction(param_transfer_functions, non_param_transfer_functions, modelparams['N']) 


    #CREATING LEARNING RULE
    bf = modelparams['bf']
    qf = modelparams['qf']
    xf = modelparams['xf']
    paramLR = [xf, xf, bf, bf, qf, modelparams['amp_median']]  #learning rule
    lr = LearningRule(paramLR, tf_sig)

    #CREATING CONNECTIVITY0.85
    w_ff = (modelparams['amp_ff']/modelparams['amp_median']) * np.ones(modelparams['p']-1)#9)
    w_rec = np.ones(modelparams['p'])
    paramSim = [modelparams['N'], modelparams['c'], modelparams['p']] #N,c,p
    conn = ConnectivityMatrix(lr, tf_sig, paramSim, seed=modelparams['seed'])
    matrix = conn.connectivity_sequence(w_rec, w_ff)

    #CREATING DYNAMICS
    indexes =range(10000)
    np.random.seed()
    patterns_current = conn.patterns_current
    dyn = NetworkDynamics(lr, tf_sig, matrix, patterns_current)
    dyn.std_low_dim = modelparams['std_low_dim']
    dyn.std_priv = modelparams['std_priv']
    dyn.dt = modelparams['dt']

    #SIMULATING NETWORK DYNAMICS
    dyn.neu_indexes = indexes
    u_init = patterns_current[0] 

    sols = []
    noise = []
    overlaps = []
    ovs, ov_norm, sol, rn_all,  rn, noi = dyn.dynamics_low(modelparams['T'], u_init)
    return ov_norm, sol, rn

#plt.plot(time,ov_norm)
#plt.show()
#plt.plot(time,sol)
#plt.show()
#histogram plot
#plt.hist(rn, density=True, bins=50, histtype = 'step')
#plt.show()
