import numpy as np
import pickle 
from bif_diagram import *
import sys

seed = int(sys.argv[1])

noise, seq =  exploring_noise_sequences(seed=seed)
pickle.dump((noise, seq),open('noise_sequences_'+str(seed)+'.p','wb'))

