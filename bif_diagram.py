import numpy as np
import pickle 
import matplotlib.pyplot as plt
from low_rank_noise import *
from analysis_functions import *
import pdb


modelparams['T']=4000
modelparams['std_low_dim']=0

the_w_ff = np.arange(0,1.6,.02)
the_w_rec = np.arange(0.05,4.05,.02)


def exploring_cv_skew():
    modelparams['T'] =  10.
    the_noise = np.arange(0.3,1.,.05)
    the_ff = np.arange(0.1,.9,.05)
    times = []
    for noi in the_noise:
        for a_ff in the_ff:
            modelparams['std_low_dim'] = noi
            modelparams['amp_ff'] = a_ff
            ov_norm, sol, rn  = simulate_network()
            d_times = dwell_time(ov_norm[:,1:9], modelparams['dt'])
            times.append(d_times)
            print('Noise=',noi,'|amp_ff=',a_ff)
    return times


def exploring_noise_sequences(seed=2):
    modelparams['std_low_d']=0
    modelparams['std_low_dim']=0
    modelparams['amp_median']=1.
    the_noise = np.arange(0,7,.02)
    seq =[ ]
    for noi in the_noise:
        modelparams['std_priv']=noi
        ov_norm, sol, rn  = simulate_network()
        max_last=np.max(ov_norm[:,9],axis=0)
        the_max_rest = np.max(ov_norm[:,0:9],axis=0)
        max_rest = np.max(the_max_rest)
        seq.append(max_rest<max_last)
        print('Noise=',noi,'|Seq=',max_rest<max_last)
    return the_noise, seq

        


def time_scale_sequences(seed=2):
    #the_w_rec, the_w_ff = pickle.load(open('data/attracotrs_params.p','rb'))
    the_w_rec_seq, the_w_ff_seq = pickle.load(open('data/sequences_params_mean.p','rb'))
    s_seq = 60
    ch_w_ff= [the_w_ff_seq[l] for l in range(1,s_seq,5)]
    ch_w_rec= [the_w_rec_seq[l] for l in range(1,s_seq,5)]
    the_time = []
    the_w_rec = []
    for  w_ff0,w_rec0 in zip(ch_w_ff,ch_w_rec): 
        if .3<w_ff0: # away triple point
            rec0 = max(w_rec0,.1)
            t_w_rec = np.arange(rec0+0.1,3.5,.01)
            t_time =  []
            el_w_rec = []
            is_seq_prev = False
            for w_rec in t_w_rec:
                modelparams['amp_ff'] = w_ff0
                modelparams['amp_median'] = w_rec
                ov_norm, sol, rn  = simulate_network()
                tim = dwell_time(ov_norm[:,1:9], modelparams['dt'])
                av_time = np.mean(tim)
                t_time.append(av_time)
                el_w_rec.append(w_rec)
                print('Time sequences','|w_rec=',w_rec,'|w_ff=',w_ff0)
            the_time.append(t_time)
            the_w_rec.append(el_w_rec)
    return the_w_rec,ch_w_rec,ch_w_ff, the_time


    

def transition_attractors(seed = 2):
    modelparams['seed']=seed
    attractor_w_ff = []
    attractor_w_rec = []
    ind_rec_start = 0
    ind_rec_end = the_w_rec.shape[0]
    for w_ff in the_w_ff:
        for w_rec in the_w_rec[ind_rec_start:ind_rec_end]:
            modelparams['amp_ff'] = w_ff
            modelparams['amp_median'] = w_rec
            ov_norm, sol, rn  = simulate_network()
            amax = np.max(ov_norm[-1,0:9])
            if .6<amax:
                attractor_w_ff.append(w_ff)
                attractor_w_rec.append(w_rec)
                ind_rec_start = np.max(np.where(the_w_rec==w_rec)[0][0]-10,0) 
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

