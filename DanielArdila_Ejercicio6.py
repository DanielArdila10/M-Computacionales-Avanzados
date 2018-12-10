# -*- coding: utf-8 -*-
"""
Created on Fri Aug 24 07:49:05 2018

@author: danie
"""

import numpy as np
import matplotlib.pyplot as plt


x_obs = np.array([-2.0,1.3,0.4,5.0,0.1, -4.7, 3.0, -3.5,-1.1])
y_obs = np.array([ -1.931,   2.38,   1.88,  -24.22,   3.31, -21.9,  -5.18, -12.23,   0.822])
sigma_y_obs = ([ 2.63,  6.23, -1.461, 1.376, -4.72,  1.313, -4.886, -1.091,  0.8054])
plt.errorbar(x_obs, y_obs, yerr=sigma_y_obs, fmt='o')

def model(x,a,b,c):
    return a*x*x + b*x+c


def modela(l,g,r):
    return 2*np.pi*np.sqrt((l+r)/g)*np.sqrt(1+(2*r*r/(5*(l+r)**2)))

def loglikelihood(x_obs, y_obs, sigma_y_obs, a, b,c):
    d = y_obs -  model(x_obs, a, b,c)
    d = d/sigma_y_obs
    d = -0.5 * np.sum(d**2)
    return d

def logprior(a, b,c):
    p = -np.inf
    if a < 10 and a >-10 and b >-20 and b<20 and c<30 and c>-30:
        p = 0.0
    return p

N = 50000
lista_a = [np.random.random()]
lista_b = [np.random.random()]
lista_c = [np.random.random()]
logposterior = [loglikelihood(x_obs, y_obs, sigma_y_obs, lista_a[0], lista_b[0], lista_c[0]) + logprior(lista_a[0], lista_b[0],lista_c[0])]

sigma_delta_a = 0.2
sigma_delta_b = 1.0
sigma_delta_c = 1.0
for i in range(1,N):
    propuesta_a  = lista_a[i-1] + np.random.normal(loc=0.0, scale=sigma_delta_a)
    propuesta_b  = lista_b[i-1] + np.random.normal(loc=0.0, scale=sigma_delta_b)
    propuesta_c  = lista_c[i-1] + np.random.normal(loc=0.0, scale=sigma_delta_c)

    logposterior_viejo = loglikelihood(x_obs, y_obs, sigma_y_obs, lista_a[i-1], lista_b[i-1],lista_c[i-1]) + logprior(lista_a[i-1], lista_b[i-1],lista_c[i-1])
    logposterior_nuevo = loglikelihood(x_obs, y_obs, sigma_y_obs, propuesta_a, propuesta_b,propuesta_c) + logprior(propuesta_a, propuesta_b,propuesta_c)

    r = min(1,np.exp(logposterior_nuevo-logposterior_viejo))
    alpha = np.random.random()
    if(alpha<r):
        lista_a.append(propuesta_a)
        lista_b.append(propuesta_b)
        lista_c.append(propuesta_c)
        logposterior.append(logposterior_nuevo)
    else:
        lista_a.append(lista_a[i-1])
        lista_b.append(lista_b[i-1])
        lista_c.append(lista_c[i-1])
        logposterior.append(logposterior_viejo)
lista_a = np.array(lista_a)
lista_b = np.array(lista_b)
lista_c = np.array(lista_c)
logposterior = np.array(logposterior)
"""
plt.plot(lista_a[100:], label='a')
plt.plot(lista_b[100:], label='b')
plt.plot(lista_c[100:], label='intercepto')
"""


new_x=np.linspace(x_obs.min(),x_obs.max(),100)
y_final= model(new_x,np.mean(lista_a),np.mean(lista_b),np.mean(lista_c))
plt.plot(new_x,y_final)
plt.errorbar(x_obs,y_obs, yerr=sigma_y_obs, fmt='o')

#plt.plot(logposterior[100:], label='loglikelihood')
