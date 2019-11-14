import numpy as np
from scipy.stats import bernoulli
from scipy import sparse 
import time

class ConnectivityMatrix:
	'''This class creates the connectivity matrix'''

	def __init__(self,LR,TF,param_net,seed = False):
		# set-up random seed
		if seed == False:
			pass
		else:
			np.random.seed(seed = seed)

		#tranfer function and learning rule
		self.myTF=TF
		self.myLR=LR

		# parameters for the dynamics
		self.N=int(param_net[0])
		self.c=param_net[1]
		self.p=int(param_net[2])			
		
		self.patterns_current = np.random.normal(0.,1., size=(self.p,self.N))
		self.patterns_fr = self.myTF.TF(self.patterns_current)
		self.sigma = 0.
		
	def connectivity_sequence(self,w_rec,w_ff):

		pre_rec = np.einsum('ij,i->ij',self.myLR.g(self.patterns_fr),w_rec)
		post_rec = self.myLR.f(self.patterns_fr)	
		
		
		pre_ff =np.einsum('ij,i->ij',self.myLR.g(self.patterns_fr)[0:self.p-1,:] , w_ff)
		post_ff =self.myLR.f(self.patterns_fr)[1:self.p,:]	
		
		# here we are not including the wrap
		pre_wrap =self.myLR.g(self.patterns_fr)[self.p-1:self.p,:] *  w_ff[0]
		post_wrap =self.myLR.f(self.patterns_fr)[0:1,:]	


		print('Patterns created. N patterns:', self.p)
		#number of entries different than zero
		N2bar=np.random.binomial(self.N*self.N,self.c)
		row_ind=np.random.randint(0,high=self.N,size=N2bar)
		column_ind=np.random.randint(0,high=self.N,size=N2bar)
		print('Structural connectivity created')
		
		dN=300000
		n = int(N2bar/dN)
		connectivity=np.array([])
		for l in range(n):
			# smart way to write down the outer product learning
			con_rec = np.einsum('ij,ij->j',post_rec[:,row_ind[l*dN:(l+1)*dN]],pre_rec[:,column_ind[l*dN:(l+1)*dN]])
			con_ff = np.einsum('ij,ij->j',post_ff[:,row_ind[l*dN:(l+1)*dN]],pre_ff[:,column_ind[l*dN:(l+1)*dN]])
			con_wrap = np.einsum('ij,ij->j',post_wrap[:,row_ind[l*dN:(l+1)*dN]],pre_wrap[:,column_ind[l*dN:(l+1)*dN]])
			gaussian = self.sigma * np.random.normal(0,1,con_ff.shape[0]) #gaussian
			connectivity=np.concatenate((connectivity,con_rec + con_ff +  gaussian),axis=0)
			#connectivity=np.concatenate((connectivity,con_rec + con_ff +  gaussian),axis=0)
			print('Synaptic weights created:',100.*(l)/float(n),'%')
		con_rec = np.einsum('ij,ij->j',post_rec[:,row_ind[n*dN:N2bar]],pre_rec[:,column_ind[n*dN:N2bar]])
		con_ff = np.einsum('ij,ij->j',post_ff[:,row_ind[n*dN:N2bar]],pre_ff[:,column_ind[n*dN:N2bar]])
		con_wrap = np.einsum('ij,ij->j',post_wrap[:,row_ind[n*dN:N2bar]],pre_wrap[:,column_ind[n*dN:N2bar]])
		gaussian = self.sigma * np.random.normal(0,1,con_ff.shape[0]) #gaussian
		print('Synaptic weights created:',100.,'%')
		connectivity=np.concatenate((connectivity,con_rec + con_ff + gaussian),axis=0)		
		#connectivity=np.concatenate((connectivity,con_rec + con_ff + gaussian),axis=0)		
		connectivity=(self.myLR.Amp/(self.c*self.N))*connectivity
		print('Synaptic weights created')

		connectivity=sparse.csr_matrix((connectivity,(row_ind,column_ind)),shape=(self.N,self.N))
		print('connectivity created')

		return connectivity
	
		
		
		
	def connectivity_sequence_gaussian(self,w_rec,w_ff):

		pre_rec = np.einsum('ij,i->ij',self.patterns_current,w_rec)
		post_rec = self.patterns_current	
		
		
		pre_ff =np.einsum('ij,i->ij',self.patterns_current[0:self.p-1,:] , w_ff)
		post_ff =self.patterns_current[1:self.p,:]	
		
		# here we are not including the wrap
		pre_wrap = self.patterns_current[self.p-1:self.p,:] *  w_ff[0]
		post_wrap = self.patterns_current[0:1,:]	


		print('Patterns created. N patterns:', self.p)
		#number of entries different than zero
		N2bar=np.random.binomial(self.N*self.N,self.c)
		row_ind=np.random.randint(0,high=self.N,size=N2bar)
		column_ind=np.random.randint(0,high=self.N,size=N2bar)
		print('Structural connectivity created')
		
		dN=300000
		n = int(N2bar/dN)
		connectivity=np.array([])
		for l in range(n):
			# smart way to write down the outer product learning
			con_rec = np.einsum('ij,ij->j',post_rec[:,row_ind[l*dN:(l+1)*dN]],pre_rec[:,column_ind[l*dN:(l+1)*dN]])
			con_ff = np.einsum('ij,ij->j',post_ff[:,row_ind[l*dN:(l+1)*dN]],pre_ff[:,column_ind[l*dN:(l+1)*dN]])
			con_wrap = np.einsum('ij,ij->j',post_wrap[:,row_ind[l*dN:(l+1)*dN]],pre_wrap[:,column_ind[l*dN:(l+1)*dN]])
			gaussian = self.sigma * np.random.normal(0,1,con_ff.shape[0]) #gaussian
			connectivity=np.concatenate((connectivity,con_rec + con_ff +  gaussian),axis=0)
			#connectivity=np.concatenate((connectivity,con_rec + con_ff +  gaussian),axis=0)
			print('Synaptic weights created:',100.*(l)/float(n),'%')
		con_rec = np.einsum('ij,ij->j',post_rec[:,row_ind[n*dN:N2bar]],pre_rec[:,column_ind[n*dN:N2bar]])
		con_ff = np.einsum('ij,ij->j',post_ff[:,row_ind[n*dN:N2bar]],pre_ff[:,column_ind[n*dN:N2bar]])
		con_wrap = np.einsum('ij,ij->j',post_wrap[:,row_ind[n*dN:N2bar]],pre_wrap[:,column_ind[n*dN:N2bar]])
		gaussian = self.sigma * np.random.normal(0,1,con_ff.shape[0]) #gaussian
		print('Synaptic weights created:',100.,'%')
		connectivity=np.concatenate((connectivity,con_rec + con_ff + gaussian),axis=0)		
		#connectivity=np.concatenate((connectivity,con_rec + con_ff + gaussian),axis=0)		
		connectivity=(self.myLR.Amp/(self.c*self.N))*connectivity
		print('Synaptic weights created')

		connectivity=sparse.csr_matrix((connectivity,(row_ind,column_ind)),shape=(self.N,self.N))
		print('connectivity created')

		return connectivity
	
		

		pre_ff = np.einsum('ij,i->ij',self.patterns_current[0:self.p-1,:] , w_ff)
		post_ff = self.patterns_current[1:self.p,:]	
		
		# here we are not including the wrap
		pre_wrap = self.patterns_current[self.p-1:self.p,:] *  w_ff[0]
		post_wrap = self.patterns_current[0:1,:]	


		print('Patterns created. N patterns:', self.p)
		#number of entries different than zero
		N2bar=np.random.binomial(self.N*self.N,self.c)
		row_ind=np.random.randint(0,high=self.N,size=N2bar)
		column_ind=np.random.randint(0,high=self.N,size=N2bar)
		print('Structural connectivity created')
		
		dN=300000
		n = int(N2bar/dN)
		connectivity=np.array([])
		for l in range(n):
			# smart way to write down the outer product learning
			con_rec = np.einsum('ij,ij->j',post_rec[:,row_ind[l*dN:(l+1)*dN]],pre_rec[:,column_ind[l*dN:(l+1)*dN]])
			con_ff = np.einsum('ij,ij->j',post_ff[:,row_ind[l*dN:(l+1)*dN]],pre_ff[:,column_ind[l*dN:(l+1)*dN]])
			con_wrap = np.einsum('ij,ij->j',post_wrap[:,row_ind[l*dN:(l+1)*dN]],pre_wrap[:,column_ind[l*dN:(l+1)*dN]])
			gaussian = self.sigma * np.random.normal(0,1,con_ff.shape[0]) #gaussian
			connectivity=np.concatenate((connectivity,con_rec + con_ff +  gaussian),axis=0)
			#connectivity=np.concatenate((connectivity,con_rec + con_ff +  gaussian),axis=0)
			print('Synaptic weights created:',100.*(l)/float(n),'%')
		con_rec = np.einsum('ij,ij->j',post_rec[:,row_ind[n*dN:N2bar]],pre_rec[:,column_ind[n*dN:N2bar]])
		con_ff = np.einsum('ij,ij->j',post_ff[:,row_ind[n*dN:N2bar]],pre_ff[:,column_ind[n*dN:N2bar]])
		con_wrap = np.einsum('ij,ij->j',post_wrap[:,row_ind[n*dN:N2bar]],pre_wrap[:,column_ind[n*dN:N2bar]])
		gaussian = self.sigma * np.random.normal(0,1,con_ff.shape[0]) #gaussian
		print('Synaptic weights created:',100.,'%')
		connectivity=np.concatenate((connectivity,con_rec + con_ff + gaussian),axis=0)		
		#connectivity=np.concatenate((connectivity,con_rec + con_ff + gaussian),axis=0)		
		connectivity=(self.myLR.Amp/(self.c*self.N))*connectivity
		print('Synaptic weights created')

		connectivity=sparse.csr_matrix((connectivity,(row_ind,column_ind)),shape=(self.N,self.N))
		print('connectivity created')

		return connectivity
	
		



		
	


