import numpy as np
import pickle 
from bif_diagram import *
import sys

seed = int(sys.argv[1])

the_w_ff, ch_rec, ch_ff, time_seq = time_scale_sequences(seed=seed)
pickle.dump((the_w_ff,ch_rec, ch_ff, time_seq),open('time_sequences_'+str(seed)+'.p','wb'))

