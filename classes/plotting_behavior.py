import numpy as np
import matplotlib.pyplot as plt
import pickle
import matplotlib as mpl

def plot_instantaneous_fraction_choice_reward(p_ch, p_rw, f_ch, f_rw, foragingparams, modelparams):
    '''
    This plot gives the instantaneous  fraction of choice and reward 
    as in  Soltani et al, 06  7A 
    '''
    beta = modelparams['beta']
    amp_rpe = modelparams['amp_rpe']
    fs = 15
    ls = 15
    block_size = foragingparams['n_trials']
    n_blocks = len(foragingparams['baiting_probs'])
    baiting_ratios = [p[0]/(p[0]+p[1]) for p in foragingparams['baiting_probs']]
    fig = plt.figure(figsize=(18, 4))
    ax = fig.add_subplot(111)
    ax.plot(p_ch,lw = 2, label = 'Choice', color = 'k')
    ax.plot(p_rw,lw=2, label  = 'Reward', color = 'orange')
    ax.set_xlabel('Trials #', fontsize = fs)
    ax.set_ylabel('Instantaneous Fraction', fontsize = fs)
    BS = 0
    delta_trial =  40
    t_trials = float(block_size * n_blocks)+delta_trial
    for k in range(n_blocks):
        BS=BS+block_size#s[k]
        ax.axvline(x=BS, ymin=0, ymax=1,ls = '--', color='gray')
        ax.axhline(y = baiting_ratios[k], xmin = (BS-block_size)/t_trials, xmax = BS/t_trials,color='r')
        ax.axhline(y = f_ch[k], xmin = (BS-block_size)/t_trials, xmax = BS/t_trials,ls='--',color='k')
        ax.axhline(y = f_rw[k], xmin = (BS-block_size)/t_trials, xmax = BS/t_trials,ls='--',color='orange')
    ax.set_xlim([0, t_trials])
    ax.set_ylim([0, 1])
    ax.tick_params(axis='both', which='major', labelsize=ls)
    ax.tick_params(axis='both', which='minor', labelsize=ls)
    ax.legend(loc = 'upper right')
    ax.set_title(r'$\beta$='+str(round(beta, 2))+' amp_rpe='+str(round(amp_rpe,2 ))) 
    name = 'inst_choice_reward_alpha_'+str(beta)+'_sigma_'+str(amp_rpe)+'.pdf'
    plt.savefig(name, bbox_inches='tight')
    plt.close()
    #plt.show()

def plot_comulative_choice(c_choices, foragingparams, modelparams, x_max=400):
    ''' Plot comulative choices'''
    fs = 15
    ls = 15
    beta = modelparams['beta']
    amp_rpe = modelparams['amp_rpe']
    n_real = c_choices.shape[0]
    n_blocks = len(foragingparams['baiting_probs'])
    block_size = foragingparams['n_trials']
    bait_prob = foragingparams['baiting_probs']
    m_com_choices = np.mean(c_choices[:,:,:],axis=0)
    fig = plt.figure(figsize=(4, 3))
    ax = fig.add_subplot(111)
    #ploting different realizations
    for l in range(1,n_real):
        ax.plot(c_choices[l,:,0],c_choices[l,:,1],lw = 1, color = 'gray',alpha = 0.3)
    ax.plot(c_choices[0,:,0],c_choices[0,:,1],lw = 1, color = 'gray',alpha = 0.3,label = 'Sessions')
    ax.plot(m_com_choices[:,0],m_com_choices[:,1],lw = 2, color = 'k', label = 'Mean')
    #ploting slopes different blocks
    k=0 
    BS = 0
    m = bait_prob[k][1]/bait_prob[k][0]
    first_x = m_com_choices[BS,0]
    first_y = m_com_choices[BS,1]
    BS = BS + block_size#s[k]
    end_x = m_com_choices[BS, 0]
    x = np.linspace(first_x, end_x, 100)
    ax.plot(x, m * (x-first_x) + first_y,color = 'r',label = 'Bait Probs')
    for k in range(1, n_blocks):
        m = bait_prob[k][1]/bait_prob[k][0]
        first_x = m_com_choices[BS,0]
        first_y = m_com_choices[BS,1]
        BS = BS + block_size#s[k]
        end_x = m_com_choices[BS,0]
        x = np.linspace(first_x, end_x, 100)
        ax.plot(x, m * (x-first_x) + first_y,color = 'r')
    ax.set_xlim([0, x_max])
    ax.set_ylim([0, x_max])
    #ax.set_xticks([0,100,200,300,400,500])
    #ax.set_yticks([0,100,200,300,400,500])
    ax.set_xlabel('Choice A', fontsize = fs)
    ax.set_ylabel('Choice B', fontsize = fs)
    ax.tick_params(axis='both', which='major', labelsize=ls)
    ax.tick_params(axis='both', which='minor', labelsize=ls)
    ax.legend(loc = 'upper left')
    ax.set_title(r'$\beta$='+str(beta)+' amp_rpe='+str(amp_rpe)) 
    name = 'cumulative_choice_beta_'+str(beta)+'_sigma_'+str(amp_rpe)+'.pdf'
    plt.savefig(name, bbox_inches='tight')
    plt.close()

def plot_blockwise_choice_vs_reward(f_rw, f_ch):
    fs = 15
    ls = 15
    fig = plt.figure(figsize=(4, 3))
    ax = fig.add_subplot(111)
    x = np.linspace(0,1,100)
    ax.plot(f_rw,f_ch,'bo', ms = 4)
    ax.plot(x,x,'k')
    ax.set_xlim([0,1])
    ax.set_ylim([0,1])
    ax.set_xticks([0,.5,1])
    ax.set_yticks([0,.5,1])
    ax.set_xlabel('Reward fraction', fontsize = fs)
    ax.set_ylabel('Choice fraction', fontsize = fs)
    ax.tick_params(axis='both', which='major', labelsize=ls)
    ax.tick_params(axis='both', which='minor', labelsize=ls)
    plt.savefig('figures/behaviour/blockwise_choice_vs_reward.pdf', bbox_inches='tight')

