import numpy as np
import pickle 
from bif_diagram import *
import sys

seed = int(sys.argv[1])

attractor_w_ff, attractor_w_rec = transition_attractors(seed=seed)
pickle.dump((attractor_w_ff,attractor_w_rec),open('transition_attractors_seed_'+str(seed)+'.p','wb'))

