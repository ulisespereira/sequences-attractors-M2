import numpy as np
import pickle 
from bif_diagram import *
import sys

seed = int(sys.argv[1])

times =  exploring_cv_skew()
pickle.dump( times, open('cv_skew_'+str(seed)+'.p','wb'))

