import pandas as pd
from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt
from bitcoin import getBTCData

def BW_NdataNew(y, A, p, mean, sigma, T, K, N):
    maxiter = 500
    tol = 1e-6
    oldLL = -100
    diff = 100
    i = 0
    while((i < maxiter) and (diff > tol)):
        Var = ESTEP_N_DATA(y,A,p,mean,sigma,T,K,N)
        diff = Var.get('LL1') - oldLL
        A = Var.get('S')
        p = Var.get('p')
        mean = Var.get('mean')
        sigma = Var.get('sigma')
        oldLL = Var.get('LL1')
        u = Var.get('u1')
        i += 1
    metadata = dict(u = u, A = A, p = p, mean = mean, sigma = sigma, LL = Var.get('LL1'), diff = diff, i = i)
    return metadata

def ESTEP_N_DATA(y: np.ndarray, A: np.ndarray, p, mean: np.ndarray, sigma: np.ndarray, T: int, K: int, N: int):
    m = A.shape[0]
    S1 = np.zeros((m,m))
    u1 = np.zeros((T,m))
    LL1 = 0
    mean_new = np.zeros((m))
    sigma_new = np.zeros((m))
    S2 = np.zeros((m,m))
    for k in range(K):
        x = y[k,:]
        UV = Estep_Ndata(x, A, p, mean, sigma, N)
        u = UV.get('u')
        u1 = u1 + u
        S1 = S1 + UV.get('S')
        LL1 = LL1 + UV.get('LL')
        v = UV.get('v')
        S2 = S2 + np.identity(m)/np.sum(np.sum(v, axis = 0), axis = 0)
        mean_new += np.dot(x,u)
        x_ = np.repeat(x[:,np.newaxis], m, axis=1)
        mean_ = np.array(mean.tolist()*T).reshape(T,-1)
        sigma_new += np.sum(((x_ - mean_)**2)*u, axis = 0)
    S = S2@S1
    S *= 1/np.sum(S, axis = 1)[:,np.newaxis]
    mean = mean_new/np.sum(u1, axis = 0)[::np.newaxis]
    sigma = np.sqrt(sigma_new/np.sum(u1, axis = 0)[::np.newaxis])
    u1 = u1/K
    p = u1[0,:]
    p *= 1/np.sum(p)
    metadata = dict(mean=mean,sigma=sigma,LL1=LL1,S=S,p=p,u1=u1)
    return metadata

def Estep_Ndata(x, A: np.ndarray, p, mean, sigma, N):
    m = A.shape[0]
    S = np.zeros((m,m))
    n = len(x)
    h = forwardback(x, A, p, mean, sigma, N)
    logbeta = h.get('logbeta')
    logalpha = h.get('logalpha')
    LL = h.get('LL')
    # forward = np.exp(logalpha)
    # backward = np.exp(logbeta)
    u = np.exp(logalpha + logbeta - np.ones((n,m))*LL)
    u = np.where(u == np.inf, 1, u)
    v = np.empty((m, n-1 ,m))
    v[:] = np.nan
    for k in range(m):
        logprob = norm.logpdf(x[1:], mean[k], sigma[k])
        logA = np.log(A[:,k]).flatten()*np.ones((n-1, m))
        logPbeta = logprob + logbeta[1:,k]
        logPbeta = np.repeat((logprob + logbeta[1:,k])[:,np.newaxis], m, axis = 1)
        v[k,:,:] = logA + logalpha[:-1,:] + logPbeta - LL*np.ones((n-1,m))
    v = np.exp(v)
    v = np.where(v == np.inf, 1, v)
    S = np.sum(v, axis = 1).T
    metadata = dict(S=S, u=u, v=v, LL=LL)
    return metadata

def forwardback(x, A: np.ndarray, p, mu, sigma, N):
    m = A.shape[0]
    n = len(x)
    prob = np.zeros((n, m))
    for k in range(m):
        prob[:,k] = norm.pdf(x, mu[k], sigma[k])
    phi = p
    logalpha = np.zeros((n,m))
    lscale = 0
    for i in range(n):
        if i>0:
            phi = phi@A
        phi = phi*prob[i,:]
        # if all(phi == 0):
        #     print('Problem')
        sumphi = np.sum(phi)
        phi = phi/sumphi
        lscale += np.log(sumphi)
        log_phi = np.where(phi == 0, np.log(0.0001), np.log(phi))
        logalpha[i,:] = log_phi + lscale*np.ones(phi.shape)
    LL = lscale
    logbeta = np.zeros((n,m))
    phi = np.ones((m,))*1/m
    lscale = np.log(m)
    for i in range(n-2, -1, -1):
        phi = A@(prob[i+1,:]*phi)
        log_phi = np.where(phi == 0, np.log(0.0001), np.log(phi))
        logbeta[i,:] = log_phi + lscale*np.ones(phi.shape)
        sumphi = np.sum(phi)
        phi *= 1/sumphi
        lscale += np.log(sumphi)
    metadata = dict(logalpha=logalpha, logbeta=logbeta, LL=LL)
    return metadata

def forward(x,A,p,mu,sigma):
    m = A.shape[0]
    n = len(x)
    prob = np.zeros((n, m))
    for k in range(m):
        prob[:,k] = norm.pdf(x, mu[k], sigma[k])
    #calculate log of forward probabilities
    phi = p.copy()
    lscale = 0
    for i in range(n):
        if i>0:
            phi = phi@A
        phi = phi*prob[i,:]
        sumphi = np.sum(phi)
        phi = phi/sumphi
        lscale += np.log(sumphi)
    LL = lscale
    return LL

def hmm_iniNdata(y, N):
    m = np.mean(y)
    s = np.std(y, ddof = 1)
    b = N - 1
    mean = np.random.normal(loc = m, scale = s, size=(N,))
    p = [1] + [0]*b
    sigma = np.ones((N,))*s
    A = np.ones((N,N))/N
    IV = dict(A=A,p=p,mean=mean,sigma=sigma)
    return IV

def main():
    Sp = getBTCData()
    T1 = Sp.shape[0]
    K = 4 #number of observation data                  
    y = Sp.values
    #Set up training window	
    WD = 100
    T2 = T1-WD
    AIC = np.empty((4,WD))
    AIC[:] = np.nan
    for N in range(2, 6):
        for l in range(WD):
            x = np.empty((K,WD))
            x[:] = np.nan
            # Training data of length WD
            x = y[(T2-WD+l):(T2+l),0:K].T
            #first step using initial parameters		
            if l == 0:
                Par = hmm_iniNdata(x,N)
                mean = Par.get('mean')
                sigma = Par.get('sigma')
                A = Par.get('A')
                p = Par.get('p')
                k = BW_NdataNew(x, A, p, mean, sigma, WD, K, N)
            # after first step using HMM parameters from previous step
            else:
                A = k.get('A')
                k = BW_NdataNew(x, A, k.get('p'), k.get('mean'), k.get('sigma'), WD, K, N)
                # calculate AIC
            b = N*N + 2*N-1
            AIC[(N-2), l] =-2*k.get('LL') + 2*b
    for N in range(4):
        y = AIC[N, :]
        x = list(range(1, 1+len(y)))
        plt.plot(x, y, label = f'AIC FOR N STATES: {N+2}')
    plt.legend()
    plt.show()

def main_2():
    Sp = getBTCData()
    WD = 252
    D = Sp.shape[0] - WD
    K = 4
    N = 4
    Pstate2 = np.zeros((WD,))
    for i in range(WD):
        y = Sp.values[(D-WD+i):(D+i),0:K].T
        T = WD
        if i == 0:
            Par = hmm_iniNdata(y,N)
            A = Par.get('A')
            p = Par.get('p')
            mean = Par.get('mean')
            sigma = Par.get('sigma')
            k = BW_NdataNew(y, A, p, mean, sigma, T,K,N)
        else:
            A = k.get('A')
            p = k.get('p')
            mean = k.get('mean')
            sigma = k.get('sigma')
            k = BW_NdataNew(y, A, p, mean, sigma, T,K,N)
        mean = k.get('mean')
        sigma = k.get('sigma')
        confiability = mean/sigma
        most_conf = np.argmin(confiability)
        u = k.get('u')
        Pstate2[i] = u[T-1,most_conf]
    print("Start date")
    print(Sp.index[D+2])
    print("End date")
    print(Sp.index[-1])
    
    y11 = Sp['close'][(D+2):(D+WD)].values.ravel()
    time = Sp.index[(D+2):(D+WD)]
    fig, ax = plt.subplots(2,1, figsize = (20, 10))
    ax[0].plot(time, y11, color="royalblue", label = "Prices")
    ax[1].plot(time,Pstate2[1:(WD-1)], color = "indianred", label="Prob. regime 4")
    ax[0].legend()
    ax[1].legend()
    plt.show()

def main_3():
    Sp = getBTCData()
    T1 = Sp.shape[0]
    K=4   #number of observation data
    N=4 #number of states
    y = Sp.values
    #Set up window
    WD = 52
    T2 = T1-WD
    T = WD
    price = np.zeros((WD,))
    for l in range(WD):
        # Training data
        x = y[(T2-WD+l):(T2+l),0:K].T
        #first step using initial parameters
        if l== 0:
            Par = hmm_iniNdata(x,N)
            mean = Par.get('mean')
            sigma = Par.get('sigma')
            A = Par.get('A')
            p = Par.get('p')
            k = BW_NdataNew(x, A, p, mean, sigma, WD, K, N)
        # after first step using HMM parameters from previous step
        else:
            A = k.get('A')
            k = BW_NdataNew(x, A, k.get('p'), k.get('mean'), k.get('sigma'), WD, K, N)
        # finding the most similar (log likelihood) date in the past
        ll = np.zeros((WD,))
        for i in range(WD):
            for j in range(K):
                z = y[(T2-WD-i+l):(T2-i+l)][j]
                A = k.get('A')
                p = k.get('p')
                mean = k.get('mean')
                sigma = k.get('sigma')
                f = forward(z, A, p, mean, sigma)
                ll[i] = ll[i] + f
        Log_likelihood = k.get('LL')
        Log_likelihood = Log_likelihood*np.ones(ll.shape)
        point = np.argmin(np.abs(ll-Log_likelihood))
        price[l] = y[(T2+l)][3] + (y[(T2-point+l)][3] - y[(T2-point+l-1)][3])
    #Print results
    print("Start date")
    print(Sp.index[(T2+1)])
    print("End date")
    print(Sp.index[-1])
    #calculate error
    real_price = Sp['close'][T2:T1].values.ravel()
    time = Sp.index[T2:T1]
    MAPE = np.mean(np.abs(price-real_price)/real_price)
    print(f"Error is: {MAPE: .2%}")
    #Plot results
    fig, ax = plt.subplots(figsize = (20, 10))
    ax.plot(time, real_price, color = 'royalblue', label = 'True Price')
    ax.plot(time, price, color = 'indianred', ls = '-.', label = 'Preditic Price')
    ax.legend()
    plt.show()

if __name__ == "__main__":
    # main() Criterio AIC para determinação da quantidade de estados ocultos
    # main_2() Verifica o comportamento de regime para os preços do BTC
    main_3()