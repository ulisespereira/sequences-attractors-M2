import numpy as np
import pickle 
from bif_diagram import *
import sys

seed = int(sys.argv[1])
sequences_w_ff, sequences_w_rec = transition_sequences(seed=seed)
pickle.dump((sequences_w_ff,sequences_w_rec),open('transition_sequences_'+str(seed)+'.p','wb'))


