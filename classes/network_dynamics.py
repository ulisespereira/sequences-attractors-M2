import numpy as np
from scipy import sparse 

class NetworkDynamics:
        '''This class creates the connectivity matrix'''
        def __init__(self,LR,TF,connectivity,patterns_current):
                #tranfer function and learning rule
            self.myLR=LR
            self.myTF=TF

                    #
            self.dt=4. # dt integration
            self.tau=20. # 20ms 

                    #input current
            self.Input=0.
            self.connectivity = connectivity
            self.patterns_fr = self.myTF.TF(patterns_current)
            self.patterns_current = patterns_current	
            self.std_priv = 0.
            self.std_low_dim = 5#8.33
            self.tau_l = 20. # time-scale structured noise
            self.tau_p = 20. # time-scale private noise
            self.neu_indexes = np.array([0])#n neurons to save
            

        def fieldDynamics(self,u,t):
            field = -u+self.Input+self.connectivity.dot(self.myTF.TF_dyn(u))
            return field/self.tau
    
        def field_noise_low(self,noi):
            gauss = np.random.normal(0,1)
            con = np.sqrt((2 * self.dt * self.std_low_dim**2)/self.tau_l)
            field = -(self.dt * noi)/self.tau_l +  con  * gauss
            return field

        def field_noise_priv(self, noi):
            gauss = np.random.normal(0,1,noi.shape[0])
            con = np.sqrt((2 * self.dt * self.std_priv**2)/self.tau_p)
            field = -(self.dt * noi)/self.tau_p +  con  * gauss
            return field
            
	# dynamics 
        def dynamics_low(self,period,u_init):
            T = period
            un = u_init #initial condition
            p, N = self.patterns_fr.shape
            noi_struc = np.random.normal(0, 1)
            noi_priv = np.random.normal(0, 1, u_init.shape[0])
            mysol=[] #neurons dynammics
            ovs=[] # overlap
            ov_norm = [] # normalized overlap
            noi = [noi_struc]
            rn =  self.myTF.TF_dyn(un)
            mysol.append(rn[self.neu_indexes])
            # overlaps
            overlap = (1./N) * np.einsum('ij,j->i',self.myLR.g(self.patterns_fr),rn)
            ov_norm.append(overlap/(np.sqrt(self.myLR.intg2)*np.std(rn)))
            ovs.append(overlap)

            #post_ff =np.concatenate((self.myLR.f(self.patterns_fr)[1:p,:],self.myLR.f(self.patterns_fr)[0:1,:]),axis = 0)	
            # no wrap
            rand_vec = np.random.normal(0,1,N)
            post_ff = self.myLR.f(self.patterns_fr)[1:p,:]
            t=0
            r_all = []
            while t<=T:
                    #NOISE = (self.bias_l + self.std_l * noi_struc * rand_vec) #* np.einsum('ij,i->j',post_ff,overlap[0:p-1]) + self.amp_noise * noi_priv
                    #NOISE_LOW_DIM =  np.sqrt((2 * self.dt)/self.tau) * noi_struc *  np.einsum('ij,i->j',post_ff,overlap[0:p-1])
                    #NOISE_PRIV =  np.sqrt((2 * self.dt)/self.tau) * noi_priv 
                    NOISE_LOW_DIM = noi_struc *  np.einsum('ij,i->j',post_ff,overlap[0:p-1])
                    NOISE_PRIV = noi_priv 
                    un = un + self.dt * self.fieldDynamics(un,t) + self.dt * (NOISE_LOW_DIM + NOISE_PRIV)/self.tau
                    noi_struc = noi_struc + self.field_noise_low(noi_struc) # colored structured noise 
                    noi_priv = noi_priv + self.field_noise_priv(noi_priv) # colored private noise (faster noise) 
                    t = t + self.dt
                    rn =  self.myTF.TF_dyn(un)
                    mysol.append(rn[self.neu_indexes])
                    if t%1000==0:
                        r_all.append(rn)
                    overlap = (1./N) * np.einsum('ij,j->i',self.myLR.g(self.patterns_fr),rn)
                    ov_norm.append(overlap/(np.sqrt(self.myLR.intg2)*np.std(rn)))
                    ovs.append(overlap)	
                    print('t=',t,' of T=',T,'overlaps')
            mysol = np.array(mysol)	
            ovs = np.array(ovs)
            ov_norm = np.array(ov_norm)
            noi = np.array(noi)
            return ovs, ov_norm, mysol, r_all, rn, noi






