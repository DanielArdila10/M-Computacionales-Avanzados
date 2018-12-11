# -*- coding: utf-8 -*-
"""

@author: danie
"""

import matplotlib.pyplot as plt
import numpy as np


data=np.loadtxt('fitting.txt')

x_obs = data[:,0]
y_obs = data[:,1]
y_sigma_obs = data[:,2]
plt.errorbar(x_obs, y_obs, yerr=sigma_y, fmt='o')

def model(x,coef):
    polinomial_oder=len(coef)
    y = np.zeros(polinomial_oder)
    for i in range(0,len(y)):
        y+=coef[i]*x**i
        
    return y



def likelihood(x_obs, y_obs, y_sigma_obs, params):
    y_model = model(x_obs, params)
    d = -0.5 * ((y_model - y_obs)/y_sigma_obs)**2
    return np.sum(d)



def evidence(x_obs, y_obs, y_sigma_obs, n_dim=1, N = 100000):
    params = np.random.random(N * n_dim) * 2.0 - 1.0
    params = np.reshape(params, [N, n_dim])
    like_params = np.zeros(N)
    for i in range(N):
        like_params[i] = likelihood(x_obs, y_obs, y_sigma_obs, params[i,:])
    
    return np.mean(like_params)

n_dims = 21
e = np.zeros(n_dims)
for i in range(n_dims):
    e[i] = evidence(x_obs, y_obs, y_sigma_obs, n_dim=i+1)
    print(e[i])

prueba=model(1,[20,10,5,15,15])    
