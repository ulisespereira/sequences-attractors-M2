from low_rank_noise import *
from analysis_functions import *
import pickle
times = []
rates = []
n_rep = 400
for l in range(n_rep):
    ov_norm, sol, r_f = simulate_network()
    ind_pat = ind_patterns(ov_norm)
    tim = dwell_time(ov_norm, modelparams['dt'])
    times.append(tim)
    pat_rates = mean_rate_at_overlap(sol, ind_pat)
    rates.append(pat_rates)
    print('Realization N=',l+1)


times=np.array(times)
rates=np.array(rates)

pickle.dump(rates,open('rates_1.p','wr'))
pickle.dump(times,open('times_1.p','wr'))
