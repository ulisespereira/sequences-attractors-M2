import numpy as np
import matplotlib.pyplot as plt

class foraging_task(object):
    def __init__(self, modelparams):
        '''Foraging task
        '''
        self.N_pfc = modelparams['N_pfc'] # number of areas
        self.dt = modelparams['dt'] # integration time ms
        self.t_start = modelparams['t_start']#start stim
        self.t_stop = modelparams['t_stop'] #end stim
        self.t_go = modelparams['t_go'] # make the choice
        self.t_reward = np.inf #reward delivery
        self.period = modelparams['period'] #period strial
        self.T_rpe = modelparams['T_rpe'] #period strial
        self.tau_reward = modelparams['tau_reward'] #time-scale 10 trials
        self.beta = modelparams['beta']
        
        #response and dc
        self.inputs = dict( 
        # input to thalamus
        wait_pfc = np.zeros(self.N_pfc),   
        rpe_pfc = np.zeros(self.N_pfc)   
        )


        # probabilities
        self.p_a = .225 # probability reward forging task
        self.p_b = .225 # probability reward forging task
        self.bait_b = 0
        self.bait_a = 0
        self.ch = np.nan

        # rewards    
        self.reward_a = 0
        self.reward_b = 0
        self.m_reward_a = 0.
        self.m_reward_b = 0.
        self.rpe_a = 0
        self.rpe_b = 0
 
    def input_current(self,t):
        ''' square current to thalamus and cortex
        '''
        input_pfc = self.inputs['wait_pfc']
        # input
        if t<= self.t_start:
            input_pfc = self.inputs['wait_pfc'] 
        elif  self.t_start<t and  t<= self.t_stop:
            # during stimulation
            input_pfc = self.inputs['wait_pfc'] 
        elif self.t_stop<t and t<=self.t_reward:
            input_pfc = self.inputs['wait_pfc'] 
        elif self.t_reward<t and t<=(self.t_reward + self.T_rpe):
            input_pfc = self.inputs['rpe_pfc']
        else:
            input_pfc = self.inputs['wait_pfc']
        #input current
        input_current = dict(
                pfc = input_pfc)
        return input_current


    def choice(self, r_a, r_b):
        '''
        Choice defined as the WTA of the wong-wang model
        '''
        if r_a>r_b:
            self.ch = 1
        else:
            self.ch = -1
        return self.ch

    def choice_sigmoidal(self, overlap_a, overlap_b):
        '''
        choice is defined as 
        the larger average firing rate
        '''
        ov_1 = overlap_a
        ov_2 = overlap_b
        diff = ov_1 - ov_2
        p = 1./(1. + np.exp(-diff * self.beta))
        self.ch = 2 * np.random.binomial(1, p) - 1
        return self.ch
    

    def average_reward(self, reward, choice):
        ''' average reward'''
        if choice == 1:
            self.m_reward_a = self.m_reward_a + (reward - self.m_reward_a)/self.tau_reward
            if self.m_reward_a <=0:
                self.m_reward_a = 0
        elif choice == -1:
            self.m_reward_b = self.m_reward_b + (reward - self.m_reward_b)/self.tau_reward
            if self.m_reward_b<=0:
                self.m_reward_b =0
        print('Choice=', choice)
        print ('Mean reward A=', round(self.m_reward_a, 3) ,'| Mean reward B=', round(self.m_reward_b, 3))


    def foraging(self,choice, seed=None):

        '''forging task gtreward
        1: reward
        0: no reward
        '''
        if seed==None:
            pass
        else:
            np.random.seed(seed)

        if self.bait_b == 0:
            self.bait_b = np.random.binomial(1, self.p_b)

        if self.bait_a == 0:
            self.bait_a = np.random.binomial(1, self.p_a)
        print ('Bait A=',self.bait_a,'|Bait B=',self.bait_b)
        if choice == 1:
            ''' choice A'''
            reward = self.bait_a
            self.bait_a = 0 # the reward is collected
            self.reward_a = reward # instantaneous total reward a
            self.rpe_a = self.reward_a - self.m_reward_a
            print('reward a', reward)
            return reward
        elif choice == -1:
            '''choice b'''
            reward = self.bait_b
            self.bait_b = 0 #the reward is collected
            self.reward_b = reward # instantaneuos total reward b
            self.rpe_b = self.reward_b - self.m_reward_b
            print('reward b', reward)
            return reward
        else:
            return 0

class foraging_session(object):
    ''' One session of foraging'''

    def __init__(self, network, modelparams, foragingparams):
        self.network = network
        #self.learning_rule = learning_rule
        self.n_trials = foragingparams['n_trials']
        self.baiting_probs = foragingparams['baiting_probs']
        self.n_slicing = foragingparams['n_slicing']
        self.N_slicing = foragingparams['N_slicing']
        self.N_pfc = modelparams['N_pfc']

    def initial_condition_session(self):
        ''' intial condition first trial of the session'''
        u_init = dict(
            un_pfc = np.zeros(self.N_pfc),
            xn_pfc = np.zeros(self.N_pfc),
            sn_a = 0,
            sn_b = 0
            )
        return u_init

    def initial_condition_trial(self, dyn):
        '''initial conditions each trial'''
        u_init = dict(
            pfc0 = dyn['un_pfc'],
            pfc_noise0 = dyn['xn_pfc'],
            s0_a = dyn['sn_a'],
            s0_b = dyn['sn_b'],
            )
        return u_init

    def network_dynamics(self, dyn):
        '''  
        Run the dynamics for one trial
        from the dynamics in previous trial
        '''    
        u_init = self.initial_condition_trial(dyn)
        dynamics = self.network.dynamics(u_init)
        return dynamics
    
    def slicing_rates(self, dynamics):
        '''This scipit is to slice the dynamics for storing long sessions'''
        if type(self.network).__name__ == 'NetworkDynamics':
            keys = ['rates_pfc', 'rates_ww','overlaps_pfc']
            for key in keys:
                dynamics[key] = dynamics[key][::self.n_slicing,::self.N_slicing]
        else:
            keys = ['rates_pfc', 'rates_ww']
            for key in keys:
                dynamics[key] = dynamics[key][::self.n_slicing,:]
        return dynamics

    def choice_reward_avreward(self, dyn): 
        ''' Getting:
        -choice
        -reward
        -average reward
        '''
        choice = self.network.task.ch
        #reward
        if choice == 1:
            reward = self.network.task.reward_a
        else:
            reward = self.network.task.reward_b
        #mean reward
        av_reward_a  = self.network.task.m_reward_a
        av_reward_b  = self.network.task.m_reward_b
        av_reward = np.array([av_reward_a, av_reward_b])

        return choice, reward, av_reward

    def foraging_block(self, dyn, store_rates=False): 
        choices_vals = []
        rewards_vals = []

        rates_ww_vals = []
        rates_vals = []
        overlaps_vals = []
        d_weights = []

        reaction_times = []
        for l in range(self.n_trials):
            dyn = self.network_dynamics(dyn)
            reaction_times.append(dyn['reaction_time'])
            choice, reward, av_reward = self.choice_reward_avreward(dyn)
            if choice==1: # choice A
                ch = np.array([1, 0])
                rw = np.array([reward, 0])
                choices_vals.append(ch)
                rewards_vals.append(rw)
            else: # choice B
                ch = np.array([0, 1])
                rw = np.array([0, reward])
                choices_vals.append(ch)
                rewards_vals.append(rw)
            #learning
            #weight_update = self.learning_rule.reinforcement_update(rates, choice, reward, av_reward)
            #self.network.corticostriatal_update(weight_update)
            if store_rates==True:
                #self.slicing_rates(dyn)
                ind_go = int(self.network.task.t_go/self.network.task.dt)-1
                if type(self.network).__name__ == 'NetworkDynamics':
                    overlaps_vals.append(np.mean(dyn['overlaps_pfc'][0:ind_go,:], axis=0))
                rates_vals.append(np.mean(dyn['rates_pfc'][0:ind_go,:], axis=0))
                rates_ww_vals.append(np.mean(dyn['rates_ww'][0:ind_go,:], axis=0))
            #d_weights.append(weight_update)
            print('Trial=',l+1) 
        choices_vals = np.array(choices_vals).transpose()
        rewards_vals = np.array(rewards_vals).transpose()
        d_weights = np.array(d_weights)
        return choices_vals, rewards_vals, rates_vals, overlaps_vals, d_weights, rates_ww_vals, reaction_times, dyn
    
    def foraging_session(self, store_rates = False):
        ''' foraging one session
        rates_vals: session, trial, area
        '''
        dyn = self.initial_condition_session()
        choices_vals = []
        rewards_vals = []
        rates_vals = []
        rates_ww_vals = []
        overlaps_vals = []
        d_weights_vals = []
        reaction_times_vals = []
        for p_a, p_b in self.baiting_probs:
            self.network.task.p_a = p_a
            self.network.task.p_b = p_b
            choices, rewards, rates, overlaps, d_weights, rates_ww, reaction_times, dyn = self.foraging_block(dyn, store_rates = store_rates)
            choices_vals.append(choices)
            rewards_vals.append(rewards)
            d_weights_vals.append(d_weights)
            reaction_times_vals.append(reaction_times)
            if store_rates == True:
                rates_vals.append(rates)
                overlaps_vals.append(overlaps)
                rates_ww_vals.append(rates_ww)
        sim_results = dict(
                choices = choices_vals, 
                rewards = rewards_vals,
                rates= rates_vals,
                overlaps = overlaps_vals,
                rates_ww = rates_ww_vals,
                reaction_times = reaction_times_vals
                )

        return sim_results



    #
