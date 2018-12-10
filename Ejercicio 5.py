# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 18:43:00 2018

@author: danie
"""

import numpy as np
import matplotlib.pyplot as plt

n_puntos=1000
def f(x, l=10):
    if(np.isscalar(x)):# esto va a funcionar si entra un numero (escalar)
        if x>1E-6:
            y = l*(1/x**2)*np.exp(-l/x)
        else:
            y = 0.0
    else: # esto funciona si es un array
        y = l*(1/x**2)*np.exp(-l/x)
      
    return y

 



#METROPOLIS HASTINGS

sigma=4
N=1000000   
x1=np.random.random()
lista=[x1]
for i in range(1,N):
    x1=lista[i-1]
    x2=x1+np.random.normal(loc=0.0, scale=sigma)
    r=min(1,f(x2)/f(x1))
    alpha=np.random.random()
    if alpha<r:
        lista.append(x2)
    else:
        lista.append(x1)

x = np.linspace(min(lista),max(lista),1000)

plt.hist(lista,density=True,bins=200)
plt.plot(x,f(x))


