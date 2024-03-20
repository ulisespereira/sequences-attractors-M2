from audioop import bias
import numpy as np
from numpy.linalg import svd 
import matplotlib.pyplot as plt



class stimulus:
    ''' External stimulation to the network'''
    def __init__(self, params):
        self.period = params['period']
        self.dt = params['dt']
        self.amp = params['amp']
        self.t_stim = params['t_stim']
        self.T_stim = params['period_stim']

    def stimulus(self, t):
        '''external stimulus'''
        if self.t_stim<t and t<(self.T_stim + self.t_stim):
            return self.amp
        else:
            return 0 * self.amp
    

class WongWang(object):
    ''' Wong-Wang decision area'''
    def __init__(self, params):
        self.dt = params['dt']
        #----------------------------------
        # Wong&Wang model
        self.A_t = params['A_t'] #Hz/nA
        self.B_t = params['B_t'] #Hz
        self.D_t = params['D_t'] #s
        self.tau_nmda = params['tau_nmda'] #s
        self.gamma = params['gamma']
        self.i_0 = params['i_0'] #nA
        self.tau_ampa = params['tau_ampa'] #ms
        self.sig_ampa = params['sig_ampa']#8#nA amplitude noise
        self.w_a_a = params['w_a_a']
        self.w_b_b = params['w_b_b']
        self.j_a_b = params['j_a_b']
        self.j_b_a = params['j_b_a']
        #-------------------------

        self.stim = stimulus(params)


    def t_fun_ww(self, current):
        '''transfer function, Wong&Wang, 2006'''
        num = self.A_t * current - self.B_t
        den = 1- np.exp(-self.D_t * num)
        return num/den


    def field_colored_noise(self, curr_a, curr_b):
        '''field of the noise to each population'''
        noise_a = np.random.normal(0, 1)
        noise_b = np.random.normal(0, 1)
        cons = np.sqrt(self.dt * (self.sig_ampa**2) * self.tau_ampa)/self.tau_ampa
        field_a = -curr_a * (self.dt/self.tau_ampa) + cons *  noise_a
        field_b = -curr_b * (self.dt/self.tau_ampa) + cons *  noise_b
        return field_a, field_b


    def field_one_area(self, s_a, s_b,  i_a, i_b, t):
        '''field large scale
        r_e_a: pop excitatory neurons a
        r_e_b: pop excitateory neurons b
        '''
        i_stim = self.stim.stimulus(t)
        # local and long-range inputs
        curr_a =  self.w_a_a * s_a - self.j_a_b * s_b + i_a + self.i_0 + i_stim[0]
        curr_b =  self.w_b_b * s_b - self.j_b_a * s_a + i_b + self.i_0 + i_stim[1]
        #rates
        r_a = self.t_fun_ww(curr_a)
        r_b = self.t_fun_ww(curr_b)
        #fields
        f_a = -s_a/self.tau_nmda + (1 - s_a) * self.gamma * r_a
        f_b = -s_b/self.tau_nmda + (1 - s_b) * self.gamma * r_b
        return f_a, f_b, r_a, r_b



class SynapticMechanism:
    def __init__(self, params, task):
        #parameter network
        self.wa = params['w1']
        self.wb = params['w2']
        self.s = params['sigma']
        self.std_priv = params['std_priv']
        self.connectivity()
        self.stim = stimulus(params)
        self.task = task
        
        self.dt = params['dt']
        self.period = params['period']
        self.i0 = params['i0']

        #parameters learning rule
        self.wmax = params['wmax']
        self.qp = params['qp']
        self.qm = params['qm']
        self.amp_learning = params['amp_learning']

        #feedback
        self.amp_feedback = params['amp_feedback']

    def connectivity(self):
        a = self.wa - 1 - self.s/2
        b = -self.s/2.
        c = -self.s/2.
        d = self.wb - 1 - self.s/2
        self.con = np.array([[a, b],[c , d]])

    def fixed_points(self):
        det = self.wa * self.wb - (1 + self.s/2.) * (self.wa + self.wb) + 1 + self.s
        I_a =  self.i0[0]
        I_b =  self.i0[1]
        num_a = (1. + self.s/2. - self.wb) * I_a - (self.s/2.) * I_b
        num_b = (1. + self.s/2. - self.wa) * I_b - (self.s/2.) * I_a
        r_a = num_a/det
        r_b = num_b/det
        return np.array([r_a, r_b])
    
    def eigenvalues(self):
        sum_w = self.wa + self.wb
        dif_w = self.wa - self.wb
        disc =  np.sqrt(dif_w**2 + self.s**2)
        lambda_p = 0.5 * (sum_w - 2 - self.s + disc)
        lambda_n = 0.5 * (sum_w - 2 - self.s - disc)
        return np.array([[lambda_p, 0], [0, lambda_n]])
    
    def eigenvectors(self):
        sum_w = self.wa + self.wb
        dif_w = self.wa - self.wb
        disc =  np.sqrt(dif_w**2 + self.s**2)
        v_p = np.array([-(dif_w + disc)/self.s, 1]) 
        v_n = np.array([-(dif_w - disc)/self.s, 1])
        v_p = v_p/np.sqrt(((dif_w + disc)/self.s)**2 + 1)
        v_n = v_n/np.sqrt(((dif_w - disc)/self.s)**2 + 1)
        return np.array([v_p, v_n]).transpose()


    def field(self, u, s_a, s_b, t):
        gauss = np.random.normal(0, 1, 2)
        const = np.sqrt((2 * self.std_priv**2 )/self.dt)
        i_noise = const * gauss
        i_stim = self.stim.stimulus(t)
        i_feedback = self.amp_feedback * np.array([s_a, s_b])
        return self.con.dot(u) + i_feedback + i_stim  + i_noise  + self.i0
    
    def synaptic_update_reward(self, reward, choice):
        '''Synaptic plasticity rule''' 
        if reward == 1:
            if choice == 1:
                self.wa = self.wa + self.qp * (self.wmax - self.wa)
            elif choice == -1:
                self.wb = self.wb + self.qp * (self.wmax - self.wb)
        elif reward == 0:
            if choice == 1:
                self.wa = self.wa - self.qm * self.wa
            elif choice == -1:
                self.wb = self.wb - self.qm * self.wb
        self.connectivity()
    
    def synaptic_update_rpe(self, choice):
        '''Synaptic plasticity rule'''  
        relu = lambda x: np.maximum(0, x) 
        if choice == 1:
            self.wa = self.wa * self.qp + self.amp_learning  * self.task.rpe_a #* relu(self.wmax - self.wa)
            self.wb = self.wb * self.qm
        elif choice == -1:
            self.wb = self.wb * self.qp + self.amp_learning * self.task.rpe_b #* relu(self.wmax - self.wb)
            self.wa = self.wa * self.qm
        self.connectivity()


class NetworkDynamicsSynaptic:
    '''This class creates the connectivity matrix'''
    def __init__(self, modelparams, params_ww, task):
        #tranfer function and learning rule
        self.task = task
        np.random.seed()#randomizing the seed 
        self.dt = modelparams['dt'] # dt integration
        self.period = modelparams['period']
        
        # go signal parameters
        self.i_iti = params_ww['i_iti']
        self.thres = modelparams['thres']
        self.choice_mechanism = 'recurrent'
        self.i_iti = params_ww['i_iti']
        self.w_iti = params_ww['w_iti']
        self.w_a_a = params_ww['w_a_a']
        self.w_b_b = params_ww['w_b_b']
        self.t_max = params_ww['t_max']# max time to respond after go signal

        #synaptic model
        self.syn = SynapticMechanism(modelparams, self.task)

        #wong&wang
        self.ww = WongWang(params_ww)

        self.amp_proj = modelparams['amp_ff']


    def _current_go_signal(self, l, t, r_a, r_b, curr_a, curr_b, noise_a, noise_b):
        '''current based go signal'''
        if t<=self.task.t_go:
            if r_a<self.thres and r_b<self.thres: 
                in_a = curr_a + noise_a + self.i_iti # this line might be problematic
                in_b = curr_b + noise_b + self.i_iti
        else:
            if (self.thres<=r_a or self.thres<=r_b): 
                l+=1
                if l==1:#firs time choice
                    choice = self.task.choice(r_a, r_b)
                    reward = self.task.foraging(choice)
                    self.task.average_reward(reward, choice)
                    self.syn.synaptic_update_rpe(choice)
                    self.task.t_reward = t
                    l+=1
                if 1<l:
                    in_a = curr_a + noise_a + self.i_iti # this line might be problematic
                    in_b = curr_b + noise_b + self.i_iti
            else:
                if l==0:
                    in_a = curr_a + noise_a 
                    in_b = curr_b + noise_b 
                else:
                    in_a = curr_a + noise_a + self.i_iti # this line might be problematic
                    in_b = curr_b + noise_b + self.i_iti
        return in_a, in_b, l
    
    def _recurrent_weights_go_signal(self, l, t, r_a, r_b):
        '''synaptic based go signal'''
        if t<=self.task.t_go:
            if r_a<self.thres and r_b<self.thres: 
                self.ww.w_a_a = self.w_iti
                self.ww.w_b_b = self.w_iti
        else:
            if (self.thres<=r_a or self.thres<=r_b): 
                l+=1
                if l==1:#firs time choice
                    choice = self.task.choice(r_a, r_b)
                    reward = self.task.foraging(choice)
                    self.task.average_reward(reward, choice)
                    self.syn.synaptic_update_rpe(choice)
                    self.task.t_reward = t
                    l+=1
                if 1<l:
                    self.ww.w_a_a = self.w_iti
                    self.ww.w_b_b = self.w_iti
            else:
                if t<self.task.t_go+self.t_max:
                    if l==0:
                        self.ww.w_a_a = self.w_a_a
                        self.ww.w_b_b = self.w_b_b
                    else:
                        self.ww.w_a_a = self.w_iti
                        self.ww.w_b_b = self.w_iti
                else:
                    self.ww.w_a_a = self.w_iti
                    self.ww.w_b_b = self.w_iti
        return l
    
    def dynamics(self, u_init):
        #synaptic
        un_pfc = u_init['pfc0'] 
        
        #Wong-wang
        s_a = u_init['s0_a']
        s_b = u_init['s0_b']
        noise_a = 0 #initial condition of the noise
        noise_b = 0
        curr_a = self.amp_proj * un_pfc[0]
        curr_b = self.amp_proj * un_pfc[1]
        in_a = curr_a + noise_a 
        in_b = curr_b + noise_b 
        
        #appending dynamics
        rates_pfc = [] #neurons dynammics LA
        rates_ww = [] #rates WW
    
        t=0
        l = 0
        while t<=self.period:
            #synaptic model
            un_pfc = un_pfc + self.dt * self.syn.field(un_pfc, s_a, s_b, t)
            rates_pfc.append(un_pfc)

            #Wong&Wang
            f_a, f_b, r_a, r_b = self.ww.field_one_area(s_a, s_b, in_a, in_b, t)
            s_a = s_a + self.ww.dt * f_a
            s_b = s_b + self.ww.dt * f_b
            f_noise_a, f_noise_b = self.ww.field_colored_noise(noise_a, noise_b)
            noise_a = noise_a + f_noise_a
            noise_b = noise_b + f_noise_b
            rates_ww.append([r_a, r_b])

            if self.choice_mechanism == 'current':
                # update in_a and in_b internally
                in_a, in_b, l = self._current_go_signal(l, t, r_a, r_b, curr_a, curr_b, noise_a, noise_b)
                curr_a = self.amp_proj * un_pfc[0]
                curr_b = self.amp_proj * un_pfc[1]
            elif self.choice_mechanism == 'recurrent':
                l = self._recurrent_weights_go_signal(l, t, r_a, r_b)
                in_a = curr_a + noise_a 
                in_b = curr_b + noise_b 
                curr_a = self.amp_proj * un_pfc[0]
                curr_b = self.amp_proj * un_pfc[1]
            t += self.dt
        self.task.rpe_a = 0 
        self.task.rpe_b = 0 
        dynamics = dict(
                #synaptic variables
                rates_pfc = rates_pfc,
                un_pfc = un_pfc,
                xn_pfc = np.zeros(un_pfc.shape),

                #wong-wang variables
                sn_a = s_a,
                sn_b = s_b,
                rates_ww = rates_ww,

                #task variables
                reaction_time = self.task.t_reward - self.task.t_go 
                )
        for key in dynamics:
            dynamics[key] = np.array(dynamics[key])
        return dynamics
