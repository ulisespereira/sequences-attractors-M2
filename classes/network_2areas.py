import numpy as np
from scipy import sparse 
from random import choices
from scipy import sparse 
from scipy import integrate
from scipy.optimize import brentq
from two_area_network import *  

	
	
	

class LearningRule:
    ''' This class gives the learning rule'''
    def __init__(self,modelparams, TF):
        self.TF = TF
        #parameters function g and f
        self.xf = modelparams['xf']
        self.xg = modelparams['xg']
        self.betaf = modelparams['bf']
        self.betag = modelparams['bg']
        self.qf = modelparams['qf']
        self.Amp = modelparams['amp_median']
        self.scale = 1.
        # here it is very important do it in this order
        self.qg = self.Qg()
        self.intg2 = self.Eg2()
        self.intf2=self.Ef2()
        #print 'Amp',self.Amp,myint
        self.gamma = self.intf2 * self.intg2 * self.Amp**2

    def std_normal(self, x):
        sigma=1. # standard normal random variable passed thorugh transfer functiion
        mu=0
        pdf=(1./np.sqrt(2 * np.pi * sigma**2))*np.exp(-(1./2.)*((x-mu)/sigma)**2)
        return pdf
    
    def log_normal(self, x):
        sigma = self.TF.sigma
        mu = self.TF.mu
        pdf=(1./np.sqrt(2 * np.pi * (x * sigma)**2))*np.exp(-(1./2.)*((np.log(x)-mu)/sigma)**2)
        #pdf = (1./np.sqrt(2 * np.pi * sigma**2)) * np.exp(-(1./2.)*((x-mu)/sigma)**2)
        return pdf
    
    # separable functions learning process leanring rule f and g
    def f(self, x):
        return  self.scale * 0.5 * (2*self.qf-1.+np.tanh(self.betaf*(x-self.xf)))
    
    def g(self, x):
        return  self.scale * 0.5 * (2*self.qg-1.+np.tanh(self.betag*(x-self.xg)))
    
    def Qg(self):# mean of f**2
        return brentq(self.Eg,0.,1.)
    
    def Eg(self, q):# mean of g
        self.qg = q
        fun = lambda x:self.std_normal(x) * self.g(self.TF.TF(x))
        var, err = integrate.quad(fun,-10.,10.)
        return var
    
    def Ef2(self):# mean of f**2
        fun = lambda x:self.std_normal(x) * self.f(self.TF.TF(x))**2
        var, err = integrate.quad(fun,-10.,10.)
        return var
    
    def Ef(self):# mean of f
        fun = lambda x:self.std_normal(x) * self.f(self.TF.TF(x))
        var, err = integrate.quad(fun,-10.,10.)
        return var
    
    def Eg2(self):# mean of g**2
        fun = lambda x:self.std_normal(x) * self.g(self.TF.TF(x))**2
        var, err = integrate.quad(fun,-10,10.)
        return var
        
        
class ConnectivityMatrices:
    '''This class creates the connectivity matrix'''
    def __init__(self,param_tf, modelparams):
        # set-up random seed
        self.seed = modelparams['seed']
        
        #tranfer function and learning rule
        TF = TransferFunction(param_tf, modelparams) 
        LR = LearningRule(modelparams, TF)
        self.TF = TF
        self.LR = LR
        
        # parameters for the dynamics
        self.N = modelparams['N']
        self.N_th = modelparams['N_th']
        self.sparsity(modelparams['c'])
        self.K = self.c * self.N
        self.p = modelparams['p']
        self.make_patterns()
        self.w_c_c = modelparams['w_ctx_ctx'] 
        self.w_c_t = modelparams['w_ctx_th'] 
        self.w_t_c = modelparams['w_th_ctx']
        self._make_indexes_connectivity()
        self.dN = 300000 #sinze chunks connectivity
        self.n = int(self.N2bar/self.dN) #truncate number of chunks
        
        #generating connectivity
        self.connectivity_ctx_ctx()
        self.connectivity_ctx_th()
        self.connectivity_th_ctx()

    def sparsity(self, c):
        '''scaling sparsity network's connectivity'''
        if c == None:
            self.c = 1./np.sqrt(self.N)
        else:
            self.c = c

    def make_patterns(self):
        '''make patterns and fixed the random seed'''
        np.random.seed(self.seed)
        p_curr_ctx = np.random.normal(0.,1., size=(self.p,self.N))
        p_curr_th = np.random.normal(0.,1., size=(self.p-1,self.N_th))
        p_fr_ctx = self.TF.TF(p_curr_ctx)
        p_fr_th = self.TF.TF(p_curr_th)
        self.patterns = dict(
                ctx = p_fr_ctx, #ctx
                th = p_fr_th  #thal 
                )
    
    def _make_patterns_ctx_ctx(self):
        ''' make the pre and post synaptic patterns
        using a generalized Hebbian learning rule
        and the forgetting kernel'''    
        pat_pre = self.LR.g(self.patterns['ctx'])
        pat_post = self.LR.f(self.patterns['ctx'])
        print('Patterns ctx<->ctx constructed')
        return pat_pre, pat_post
    
    def _make_patterns_th_ctx(self):
        ''' make the pre and post synaptic patterns
        using a generalized Hebbian learning rule ctx->th'''    
        pat_pre = self.LR.g(self.patterns['ctx'])
        pat_post = self.LR.f(self.patterns['th'])
        print('Patterns ctx->th constructed')
        return pat_pre, pat_post

    def _make_patterns_ctx_th(self):
        ''' make the pre and post synaptic patterns
        using a generalized Hebbian learning rule th->ctx'''    
        pat_pre = self.LR.g(self.patterns['th'])
        pat_post = self.LR.f(self.patterns['ctx'])
        print('Patterns th->ctx constructed')
        return pat_pre, pat_post


    def _make_indexes_connectivity(self):
        #number of entries different than zero
        self.N2bar = np.random.binomial(self.N * self.N, self.c)
        self.row_ind = np.random.randint(0, high = self.N, size = self.N2bar)
        self.column_ind = np.random.randint(0, high = self.N, size = self.N2bar)
        print('Structural connectivity created')
    
    def _chunk_connectivity(self, l, pat_pre, pat_post):
        ''' make chunk connectivity'''
        post = pat_post[:, self.row_ind[l * self.dN:(l+1) * self.dN]]
        pre = pat_pre[:, self.column_ind[l * self.dN:(l+1) * self.dN]]
        con_chunk = np.einsum('ij,ij->j', post, pre)
        return con_chunk
        
    def connectivity_ctx_ctx(self):
        pat_pre, pat_post = self._make_patterns_ctx_ctx()
        connectivity=np.array([])
        for l in range(self.n):
            con_chunk = self._chunk_connectivity(l, pat_pre, pat_post)
            # smart way to write down the outer product learning
            connectivity = np.concatenate((connectivity, con_chunk), axis=0)
            print('Synaptic weights created:',np.round(100.*(l)/float(self.n),3),'%')
        post = pat_post[:, self.row_ind[self.n * self.dN:self.N2bar]]
        pre = pat_pre[:, self.column_ind[self.n * self.dN:self.N2bar]]
        con_chunk = np.einsum('ij,ij->j', post, pre)
        print('Synaptic weights created:',100.,'%')
        connectivity = np.concatenate((connectivity, con_chunk), axis = 0)	    
        connectivity = (self.w_c_c/self.K) * connectivity
        print('Synaptic weights created')
        connectivity = sparse.coo_matrix((connectivity,(self.row_ind, self.column_ind)), shape=(self.N,self.N))
        print('connectivity created')
        self.con_ctx_ctx =  connectivity.tocsr()
    
    def connectivity_ctx_th(self):
        '''th->ctx'''
        pat_pre, pat_post = self._make_patterns_ctx_th()
        con = np.zeros((self.N, self.N_th))
        for l in range(1, self.p):
            con += np.outer(pat_post[l, :], pat_pre[l-1, :])
        self.con_ctx_th = (self.w_c_t/self.K) * con

    def connectivity_th_ctx(self):
        '''ctx->th'''
        pat_pre, pat_post = self._make_patterns_th_ctx()
        con = np.zeros((self.N_th, self.N))
        for l in range(1, self.p):
            con += np.outer(pat_post[l - 1, :], pat_pre[l-1, :])
        self.con_th_ctx = (self.w_t_c/self.K) * con


class TransferFunction:
    ''' Transfer function class'''
    def __init__(self,param_tf, modelparams):
        
        #median parameters
        self.rm_m = np.median(param_tf[:,0])
        self.b_m = np.median(param_tf[:,1])
        self.h0_m = np.median(param_tf[:,2])
        
        #linear transfer function
        self.slope_tf = modelparams['slope_tf']
        self.intercept_tf = modelparams['intercept_tf']
        
        #subsampling from distirbution of parameters
        indexes = np.random.choice(range(param_tf.shape[0]), modelparams['N'])
        self.r_s = np.array([param_tf[l, 0] for l in indexes])
        self.b_s = np.array([param_tf[l, 1] for l in indexes])
        self.h0_s = np.array([param_tf[l, 2] for l in indexes])
        self.rmax = np.array([param_tf[l, 3] for l in indexes])
    
    def TF(self, h):
        # sigmoidal TF
        phi = self.rm_m/(1.+np.exp(-self.b_m * (h - self.h0_m)))
        return phi
    
    def TF_dyn(self, h):
        # sigmoidal TF
        phi = self.r_s/(1.+np.exp(-self.b_s * (h - self.h0_s)))
        phi[self.rmax<phi]=self.rmax[self.rmax<phi]
        return phi
    
    def TF_linear(self, h):
        # reLU
        phi = self.slope_tf * h + self.intercept_tf
        phi[phi<0] = 0
        return phi
    
    

class NetworkDynamics:
    '''This class creates the connectivity matrix'''
    def __init__(self, modelparams, connectivity):
        #tranfer function and learning rule
        self.N = modelparams['N']
        self.dt = modelparams['dt'] # dt integration
        self.period = modelparams['T']
        self.tau_m2 = modelparams['tau_m2'] # 20ms 
        self.tau_th = modelparams['tau_th'] # 20ms 
        self.neu_indexes_ctx = np.array([0])# # neurons to save cortex
        self.neu_indexes_th = np.array([0])# # neurons to save thalamus
        self.input_th = 0.
        self.input_ctx = 0.
        
        #noise
        self.std_priv = modelparams['std_priv']
        self.std_low_dim = modelparams['std_low_dim']#8.33
        self.tau_l = 20. # time-scale structured noise
        self.tau_p = 20. # time-scale private noise

        #connectivity
        self.LR = connectivity.LR
        self.TF = connectivity.TF
        self.patterns_fr_ctx = connectivity.patterns['ctx']
        self.patterns_fr_th = connectivity.patterns['th']
        self.con_ctx_ctx = connectivity.con_ctx_ctx
        self.con_th_ctx = connectivity.con_th_ctx
        self.con_ctx_th = connectivity.con_ctx_th
    
    def field_th(self, u_ctx, u_th, t):
        ''' field thalamus'''
        rate_ctx = self.TF.TF_dyn(u_ctx)
        curr_th = self.con_th_ctx.dot(rate_ctx)
        field = -u_th + self.input_th + curr_th
        return field/self.tau_th
    
    def field_M2(self, u_ctx, u_th, t):
        '''field M2'''
        rate_ctx = self.TF.TF_dyn(u_ctx)
        rate_th = self.TF.TF_linear(u_th)
        curr_ctx = self.con_ctx_ctx.dot(rate_ctx) + self.con_ctx_th.dot(rate_th)
        field = -u_ctx + self.input_ctx + curr_ctx
        return field/self.tau_m2
    
    def field_noise_low(self, noi):
        '''colorf low-dimensional noise'''
        gauss = np.random.normal(0,1)
        con = np.sqrt((2 * self.dt * self.std_low_dim**2)/self.tau_l)
        field = -(self.dt * noi)/self.tau_l +  con  * gauss
        return field
    
    def field_noise_priv(self, noi):
        ''' colored private  noise'''
        gauss = np.random.normal(0,1,noi.shape[0])
        con = np.sqrt((2 * self.dt * self.std_priv**2)/self.tau_p)
        field = -(self.dt * noi)/self.tau_p +  con  * gauss
        return field
    
    def _overlaps(self, rn):
        g = self.LR.g(self.patterns_fr_ctx)
        overlap = (1./self.N) * np.einsum('ij,j->i', g, rn)
        return overlap
    
    def _normed_overlaps(self, rn, overlap):
        return overlap/(np.sqrt(self.LR.intg2) * np.std(rn))

    # dynamics 
    def dynamics(self, u_ctx0, u_th0):
        ''' simulating the dynamics of neuronal network '''
        un_ctx = u_ctx0 #initial condition
        un_th = u_th0 #initial condition
        rates_ctx = [] #neurons dynammics
        rates_th = [] #neurons dynammics
        ovs_ctx = [] # overlap
        ov_norm_ctx = [] # normalized overlap
        def append_results():
            rn_ctx =  self.TF.TF_dyn(un_ctx)
            rn_th =  self.TF.TF_linear(un_th)
            overlap = self._overlaps(rn_ctx)
            normed_overlap = self._normed_overlaps(rn_ctx, overlap)
            rates_ctx.append(rn_ctx[self.neu_indexes_ctx])
            rates_th.append(rn_th[self.neu_indexes_th])
            ovs_ctx.append(overlap)
            ov_norm_ctx.append(normed_overlap)
        append_results()
        t=0
        while t<=self.period:
            un_ctx = un_ctx + self.dt * self.field_M2(un_ctx, un_th, t) 
            un_th = un_th + self.dt * self.field_th(un_ctx, un_th, t) 
            t += self.dt
            print('time=',t)
            append_results()
        rates_ctx = np.array(rates_ctx)    
        rates_th = np.array(rates_th)    
        ovs_ctx = np.array(ovs_ctx)
        ov_norm_ctx = np.array(ov_norm_ctx)
        dynamics = dict(
                rates_ctx = rates_ctx,
                rates_th = rates_th,
                overlaps_ctx = ovs_ctx,
                norm_overlaps_ctx = ov_norm_ctx
                )
        return dynamics


