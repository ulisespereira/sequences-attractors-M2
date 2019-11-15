import numpy as np
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import CCA

def dwell_time(overlap, dt):
    ind_ov = ind_patterns(overlap)
    dwell = np.sum(ind_ov,axis = 0) * dt
    return dwell

def ind_patterns(overlap):
    '''indexes when each overlap is bigger than 0.5''' 
    ind_ov = overlap>0.4
    return ind_ov

def mean_rate_at_overlap(sol, ind_ov):
    '''firing rate patterns at each overlap'''
    rates = [] 
    for l in range(ind_ov.shape[1]):
        m_fr = np.mean(sol[ind_ov[:,l]], axis = 0)
        rates.append(m_fr)
    rates = np.array(rates)
    return rates

def covariance(rates,ind):
    '''covariances rates across attractors'''
    covar = []
    for l in range(rates.shape[1]):
        cov = np.cov(rates[:,l,ind],rowvar=False)
        covar.append(cov)
    covar = np.array(covar)
    return covar

def dimension(rates, ind):
    ''' dimension of the activity'''
    covs=covariance(rates, ind)
    dimen = []
    for l in range(rates.shape[1]):
        dim  = np.trace(covs[l])**2/np.trace(covs[l].dot(covs[l])) 
        dimen.append(dim)
    dimen = np.array(dimen)
    return dimen

def participation_ratio(rates, ind):
    ''' paritcipation ratio '''
    dimen = dimension(rates,ind)
    d = np.floor(np.mean(dimen))
    return int(d)

def pca(rates, ind):
    ''' dimension of the activity'''
    covs=covariance(rates, ind)
    dim = participation_ratio(rates, ind)
    pca = PCA(n_components=dim)  
    the_pca = []
    for l in range(rates.shape[1]):
        pca.fit(covs[l])
        the_pca.append(pca.components_)
    return the_pca

def cca(rates, ind):
    pcas = pca(rates, ind)
    alig = []
    for l in range(rates.shape[1]):
        the_al = []
        for i in range(rates.shape[1]):
            pca_i = pcas[i]
            pca_l = pcas[l]
            k = len(pcas[l])
            cca = CCA(n_components=k)
            cca.fit(pca_i.T, pca_l.T)
            U_c, V_c = cca.transform(pca_i.T, pca_l.T)
            al = 0
            for s in range(k):
                r = np.corrcoef(U_c[:,s],V_c[:,s])[0,1]
                al =al+r 
            the_al.append(al/float(k))
        alig.append(the_al)
    alig = np.array(alig)
    return alig
                    



