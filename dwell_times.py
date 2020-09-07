from low_rank_noise import *
from analysis_functions import *
import pickle
times = []
rates = []
unit_trials_states = np.loadtxt('UnitsTrialsStates.txt',delimiter=',')  
#n_rep = [int(n_trial) for n_trial in unit_trials_states[:,1]]
n_rep = 100
for l in range(n_rep):
    ov_norm, sol, r_f = simulate_network()
    ind_pat = ind_patterns(ov_norm)
    tim = dwell_time(ov_norm, modelparams['dt'])
    times.append(tim)
    #if np.sum(0<tim)==10: # all patterns happened
    pat_rates = mean_rate_at_overlap(sol, ind_pat)
    rates.append(pat_rates)
    print('Realization N=',l+1)


times=np.array(times)
rates=np.array(rates)

pickle.dump(rates,open('data/rates_2.p','wb'))
pickle.dump(times,open('data/times_2.p','wb'))
