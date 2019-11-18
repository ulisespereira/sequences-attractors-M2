import numpy as np
import pickle 
import matplotlib.pyplot as plt
from low_rank_noise import *



modelparams['T']=4000
modelparams['std_low_dim']=0

the_w_ff = np.arange(0,1.6,.02)
the_w_rec = np.arange(0.05,4.05,.02)

#the_w_rec, the_w_ff = pickle.load(open('data/attracotrs_params.p','rb'))
#the_w_rec_seq, the_w_ff_seq = pickle.load(open('data/sequences_params.p','rb'))

def transition_attractors(seed = 2):
    modelparams['seed']=seed
    attractor_w_ff = []
    attractor_w_rec = []
    for w_ff in the_w_ff:
        for w_rec in the_w_rec:
            modelparams['amp_ff'] = w_ff
            modelparams['amp_median'] = w_rec
            ov_norm, sol, rn  = simulate_network()
            amax = np.max(ov_norm[-1,0:9])
            if .6<amax:
                attractor_w_ff.append(w_ff)
                attractor_w_rec.append(w_rec)
                break
            print('Attractors','|w_rec=',w_rec,'|w_ff=',w_ff)
    return attractor_w_ff, attractor_w_rec

def transition_sequences(seed=2):
    modelparams['seed']=seed
    sequences_w_ff = []
    sequences_w_rec = []
    for w_ff in the_w_ff:
        for w_rec in the_w_rec:
            modelparams['amp_ff'] = w_ff
            modelparams['amp_median'] = w_rec
            ov_norm, sol, rn  = simulate_network()
            the_max = np.max(ov_norm,axis=0)
            if np.argmax(the_max)==9:
                sequences_w_ff.append(w_ff)
                sequences_w_rec.append(w_rec)
                break
            print('Sequences','|w_rec=',w_rec,'|w_ff=',w_ff)
    return sequences_w_ff, sequences_w_rec
#pickle.dump((time,the_w_ff, the_w_rec,ovs),open('data/bif_diagram/overlaps_dynamics.p','wb'))

