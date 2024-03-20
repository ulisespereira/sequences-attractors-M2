import numpy as np
''' This  scritps
contains a collection 
of functions for analyzing 
choice and value and 
its relation with behaviour '''


#parameters causal gaussian filter
SD_cgf = 4.
def causal_gaussian_filter(x_p, num_p):
    '''causal gaussian filter'''
    x_vals = np.arange(num_p)
    sd = SD_cgf
    gaussian = (1./np.sqrt(2 * np.pi * sd**2)) * np.exp(-(x_vals-x_p)**2/(2 * sd**2))
    gaussian[x_vals<x_p]=0
    gussian = gaussian/np.sum(gaussian)
    return gaussian

def causal_smoothing(the_choices, the_rewards):
    def c_smoothing(choices):
        num_p = choices.shape[1]
        ch_smooth=np.zeros((2, num_p))
        x_vals = np.arange(num_p) 
        for l in range(num_p):
            ch_smooth[0, l]= np.sum(causal_gaussian_filter(x_vals[l], num_p) * choices[0,:])
            ch_smooth[1, l]= np.sum(causal_gaussian_filter(x_vals[l], num_p) * choices[1,:])
        p_ch_smooth = ch_smooth[0,:]/(ch_smooth[0,:]+ch_smooth[1,:])
        return p_ch_smooth
    # choice
    p_choice = []
    p_reward = []
    for ch,rw in zip(the_choices, the_rewards):
        p_choice = p_choice + list(c_smoothing(ch))
        p_reward = p_reward + list(c_smoothing(rw))
    p_choice = np.array(p_choice)
    p_reward = np.array(p_reward)
    return p_choice, p_reward

def comulative_choices(the_choices):
    ''' This function calculates the 
    comulative choices and rewards'''

    the_com_choice = [[0,0]]
    for choices in the_choices:
        com_choice = []
        for l in range(choices.shape[1]):
            com_choice.append(the_com_choice[-1]+np.sum(choices[:,0:l], axis = 1))
        the_com_choice = the_com_choice + com_choice
    the_com_choice = np.array(the_com_choice)
    return the_com_choice

def blockwise_fraction_choices_rewards(the_choices,the_rewards):
    '''
    fraction of choices fraction of rewards
    plot 7B Soltani and Wang, 2006.  
    '''
    frac_choice = []
    frac_reward = []
    for choices,rewards in zip(the_choices,the_rewards):
        s_ch = np.sum(choices[:,0:-1], axis = 1)
        s_rw = np.sum(rewards[:,0:-1], axis = 1)
        frac_choice.append(s_ch[0]/(s_ch[0]+s_ch[1]))
        frac_reward.append(s_rw[0]/(s_rw[0]+s_rw[1]))
    frac_choice = np.array(frac_choice)
    frac_reward = np.array(frac_reward)
    return frac_choice, frac_reward


def slopes_comulative_choices(choices, block_sizes):
    com_ch = comulative_choices(choices)
    lr = []
    bs = 0
    for l in range(len(block_sizes)):
        ind_1 = bs
        bs=bs+block_sizes[l]
        ind_2 = bs
        m,b = np.polyfit(com_ch[ind_1:ind_2, 0],com_ch[ind_1:ind_2, 1],1)
        lr.append((m,b))
    return lr


