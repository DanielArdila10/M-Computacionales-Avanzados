# -*- coding: utf-8 -*-
"""
Created on Wed Sep  5 06:49:44 2018

@author: danie
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data=pd.read_csv('gdp.csv')
data=data.dropna()

index=data[data['Year']==2011]

expenditure=index["Health_expenditure_per_capita_PPP"]
years_lived=index['Years_Lived_With_Disability']

expen=np.array(expenditure)
years=np.array(years_lived)

#plt.scatter(expen,years)

x_obs=expen
y_obs=years
sigama_y_obs=[0.1]*len(y_obs)
sigma_y_obs=np.array(sigama_y_obs)

def model1(x,m,b):
    return np.log(x*m) + b

def loglikelihood(x_obs, y_obs, sigma_y_obs, m, b):
    d = y_obs -  model1(x_obs, m, b)
    d = d/sigma_y_obs
    d = -0.5 * np.sum(d**2)
    return d

def logprior(m,b):
    if  np.abs(m < 1E-2) and b >0 and b<20:
        area = 2.0*1E-2*20.0
        p = np.log(1.0/area)
    else:
        p = -np.inf
    return p

N = 50000
lista_m = [np.random.random()/10]
lista_b = [np.random.random()/10]
logposterior = [loglikelihood(x_obs, y_obs, sigma_y_obs, lista_m[0], lista_b[0]) + logprior(lista_m[0], lista_b[0])]

sigma_delta_m = 0.02
sigma_delta_b = 1

for i in range(1,N):
    propuesta_m  = lista_m[i-1] + np.abs(np.random.normal(loc=0.0, scale=sigma_delta_m)/10)
    propuesta_b  = lista_b[i-1] + np.random.normal(loc=0.0, scale=sigma_delta_b)/10

    logposterior_viejo = loglikelihood(x_obs, y_obs, sigma_y_obs, lista_m[i-1], lista_b[i-1]) + logprior(lista_m[i-1], lista_b[i-1])
    logposterior_nuevo = loglikelihood(x_obs, y_obs, sigma_y_obs, propuesta_m, propuesta_b) + logprior(propuesta_m, propuesta_b)

    r = min(1,np.exp(logposterior_nuevo-logposterior_viejo))
    alpha = np.random.random()
    if(alpha<r):
        lista_m.append(propuesta_m)
        lista_b.append(propuesta_b)
        logposterior.append(logposterior_nuevo)
    else:
        lista_m.append(lista_m[i-1])
        lista_b.append(lista_b[i-1])
        logposterior.append(logposterior_viejo)
lista_m = np.array(lista_m)
lista_b = np.array(lista_b)
logposterior = np.array(logposterior)

plt.figure()
y_model = model1(x_obs,np.mean(lista_m)/10,np.mean(lista_b)/10)
plt.errorbar(x_obs,y_obs, yerr=sigma_y_obs, fmt='o')
x=sorted(x_obs)
y=sorted(y_model)
plt.plot(np.sort(x),np.sort(y),c='r')


