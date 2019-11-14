import numpy as np
from scipy.optimize import brentq
from scipy.stats import lognorm
from random import choices

class TransferFunction:
    def __init__(self,param_tf, non_param_tf, num_neu):
        
        #median parameters
        self.rm_m = np.median(param_tf[:,0])
        self.b_m = np.median(param_tf[:,1])
        self.h0_m = np.median(param_tf[:,2])	
        
        
        #log-normal distribution of maximum firing rate
        self.mu = np.log(self.rm_m)
        self.rm_mean = np.mean(param_tf[:,0])
        self.sigma = np.sqrt(2 * (np.log(self.rm_mean) - self.mu))
        scale = np.exp(self.mu)
        r = lognorm.rvs(self.sigma,scale = scale, size=num_neu)
        self.rm= r

        # subsampling from distirbution
        indexes = np.random.choice(range(param_tf.shape[0]),num_neu)

        self.r_s = np.array([param_tf[l, 0] for l in indexes])
        self.b_s = np.array([param_tf[l, 1] for l in indexes])
        self.h0_s = np.array([param_tf[l, 2] for l in indexes])
        self.rmax = np.array([param_tf[l, 3] for l in indexes])

        #subsampling non-parametric transfer functions
        self.non_param_tf = choices(non_param_tf,k = num_neu)
        
        #self.param = True
    def TF(self,h):
        # sigmoidal TF
        phi = self.rm_m/(1.+np.exp(-self.b_m * (h-self.h0_m)))
        return phi
    def TF_dyn(self,h):
        # sigmoidal TF
        #phi = self.rm/(1.+np.exp(-self.b_m * (h-self.h0_m)))
        phi = self.r_s/(1.+np.exp(-self.b_s * (h-self.h0_s)))
        phi[self.rmax<phi]=self.rmax[self.rmax<phi]
        return phi
    
