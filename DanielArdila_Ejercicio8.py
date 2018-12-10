# -*- coding: utf-8 -*-
"""

@author: danie
"""

import numpy as np
import matplotlib.pyplot as plt

alcance=np.array([880, 795, 782, 976, 178])
y_obs=alcance
sigma_y_obs=np.ones(len(alcance))*5

def model(param):
    """
    Modelo de alcance maximo dado param[0] (angulo) y param[1] (velocidad)
    """
    y = (np.sin(2.0*param[0])*param[1]**2)/9.8
    return y 

def loglikelihood(y_obs, sigma_y_obs, param):
    
    d = y_obs -  model(param)
    d = d/sigma_y_obs
    d = -0.5 * np.sum(d**2)
    return d

def logprior(param):
    p = -np.inf
    if param[0] > 0 and param[0]<np.pi/2.0 and param[1]>0 and param[1]<10000:
        p = 0.0
    return p

N = 100000
l_param = [np.array([np.pi/4.0, 10.0])]
sigma_param = np.array([0.1, 1.0])
n_param = len(sigma_param)
for i in range(1,N):
    propuesta  = l_param[i-1] + np.random.normal(size=n_param)*sigma_param
    logposterior_viejo = loglikelihood(y_obs, sigma_y_obs, l_param[i-1]) + logprior(l_param[i-1])
    logposterior_nuevo = loglikelihood(y_obs, sigma_y_obs, propuesta) + logprior(propuesta)

    r = min(1,np.exp(logposterior_nuevo-logposterior_viejo))
    alpha = np.random.random()
    if(alpha<r):
        l_param.append(propuesta)
    else:
        l_param.append(l_param[i-1])
        

l_param=np.array(l_param)
l_param = l_param[N//10:,:]
velocidad=l_param[:,1]
v_medio  = np.mean(l_param[:,1])
plt.hist(velocidad,bins=20)
