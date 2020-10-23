from scipy.optimize import minimize
from sklearn.mixture import GaussianMixture
import numpy as np
import pandas as pd


normal_pdf = lambda x, mean, std: np.exp(-0.5*((x-mean)/std)**2)/(std*np.sqrt(2*np.pi))

class MHH:

    def __init__(self, n_hidden_states = 3, covariance_matriz = 'spherical'):
        self._n_states = n_hidden_states
        self.gmm = GaussianMixture(n_components = n_hidden_states, covariance_type = covariance_matriz, max_iter = 1000, random_state = 42)

    def fit_HMM(self,Prices,nstarts=10):

        R = np.log(Prices[1:]/Prices[:-1])
        R = pd.DataFrame(R)
        n=R.shape[0]
        self.gmm.fit(R)
        self.means_ = self.gmm.means_.ravel()
        self.std = np.sqrt(self.gmm.covariances_.ravel())
    
        # bnds = ((None,None),(None,None),(None,None),(None,None),(None,0),(None, 0),(None, 0), (0, 1),(0, 1), (0, 1),(None,None))
        bnds = ((None,0),(None, 0),(None, 0), (0, 1),(0, 1), (0, 1),(None,None))

        def HMM_NLL(x):
            # sig=np.exp(x[0])
            # MU=x[1:4]
            # r0,r1,r2=np.exp(x[4:7])
            r0,r1,r2 = np.exp(x[0:3])
            p0,p1,p2=x[3:-1]
            beta=x[-1]
            TP=np.array([[1-r0,r0*p0,r0*(1-p0)],[r1*p1,1-r1,r1*(1-p1)],[r2*p2,r2*(1-p2),1-r2]]).T
            TP *= 1/np.sum(TP, axis = 1)[:, np.newaxis]
            P=np.zeros((n+1,self._n_states))
            P[0,:]=np.ones(self._n_states)/self._n_states
            S=np.zeros(n+1)
            rold=0
            for t in range(n):
                P[t+1]=np.matmul(TP,P[t])
                difference_gain = [R.values[t] - rold*beta]*self._n_states
                for j, params in enumerate(tuple(zip(difference_gain, self.means_, self.std))):
                    P[t+1,j]=P[t+1,j]*normal_pdf(*params)
                rold=R.values[t]
                S[t+1]=max(P[t+1])
                P[t+1]=P[t+1]/S[t+1]
            nll = -np.sum(np.log(S[1:]))
            return nll

        best=np.inf
        for i in range(nstarts):
            # mu0=np.random.rand()*0.001
            # mu1=np.random.rand()*0.001
            # mu2=-np.random.rand()*0.001
            r0=np.random.rand()
            r1=np.random.rand()
            r2=np.random.rand()
            p0=np.random.rand()
            p1=np.random.rand()
            p2=np.random.rand()
            # sig=np.random.rand()*0.1
            beta=np.random.rand()*0.1
            # x0=np.array([np.log(sig),mu0,mu1,mu2,np.log(r0),np.log(r1),np.log(r2),p0,p1,p2,beta])
            x0=np.array([np.log(r0),np.log(r1),np.log(r2),p0,p1,p2,beta])

            OPT = minimize(HMM_NLL, x0,bounds=bnds)

            if i==0:
                x=OPT.x    
                OPTbest=OPT

            if OPT.fun<best:
                best=OPT.fun
                x=OPT.x
                OPTbest=OPT

        # self.sig=np.exp(x[0])
        self.sig = self.std
        self.MU=self.means_
        r0,r1,r2=np.exp(x[0:3])
        p0,p1,p2=x[3:-1]
        self.TP=np.array([[1-r0,r0*p0,r0*(1-p0)],[r1*p1,1-r1,r1*(1-p1)],[r2*p2,r2*(1-p2),1-r2]]).T
        self.TP *= 1/np.sum(self.TP, axis = 1)[:, np.newaxis]
        self.beta=x[-1]
        self.x=x
        self.OPT=OPT
        
        # reorder so MU is increasing 
        ix=np.argsort(-self.MU)
        self.MU=self.MU[ix]
        self.TP=self.TP[np.ix_(ix,ix)]
        
    def get_hidden_state_probabilities(self,Prices):
        R=np.log(Prices[1:]/Prices[:-1])
        n=R.shape[0]
        P=np.zeros((n+1,self._n_states))
        P[0,:]=np.ones(self._n_states)/self._n_states
        rold=0
        for t in range(n):
            P[t+1]=np.matmul(self.TP,P[t])
            difference_gain = [R[t] - rold*self.beta]*self._n_states
            for j, params in enumerate(tuple(zip(difference_gain, self.means_, self.std))):
                P[t+1,j]=P[t+1,j]*normal_pdf(*params)
            rold=R[t]
            # P[t+1]=P[t+1]/np.sum(P[t+1])
            P[t+1] *= 1/max(P[t+1])
        return P
        
    def get_expected_abnormal_rates(self,Prices):
        P=self.get_hidden_state_probabilities(Prices)
        
        R=np.zeros(Prices.shape[0])
        R[1:]=np.log(Prices[1:]/Prices[:-1])
        
        lam,V=np.linalg.eig(self.TP)
        ix=np.argsort(lam)
        lam=lam[ix]
        V=V[:,ix]
        V[:,2]=V[:,2]/np.sum(V[:,2])
        VMU=np.matmul(V.T,self.MU)
        D=(1/(1-hmm.beta))*(lam[:2]/(1-lam[:2]))*VMU[:2]

        EAR=np.matmul(D,np.linalg.solve(V,P.T)[:2,:])+(1/(1-self.beta))*R
        
        return EAR