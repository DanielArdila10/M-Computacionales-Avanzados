# -*- coding: utf-8 -*-
"""

@author: danie
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


data=pd.read_csv('years-lived-with-disability-vs-health-expenditure-per-capita.csv')
data=data.dropna()

index=data[data['Year']==2011]

expenditure=index["Health_expenditure_per_capita_PPP"]
years_lived=index['Years_Lived_With_Disability']

expen=np.array(expenditure)
years=np.array(years_lived)



def model1(x,param):    
    return x*param[0] + param[1]

def model2(x,param):    
    return param[0]*np.log(x*param[1])

def model3(x, param):
    return param[0]*np.log(param[1]*x)+param[2]


def likelihood(x,y,param,model):
    y_model = model(x, param)
    p = y_model * np.exp(-(y_model/y))
    p = p/(y**2)
    return np.sum(p)

def montecarlo(x,y,N,param,model):
    likelihoods = np.ones(N)
    parametro0=param[0]
    parametro1=param[1]
    if len(param)<3:
    
        for i in range(N):
            likelihoods[i] = likelihood(x,y,[parametro0[i],parametro1[i]],model)
    else:
        parametro2=param[2]
        for i in range(N):
            likelihoods[i] = likelihood(x,y,[parametro0[i],parametro1[i],parametro2[i]],model)
        
    return np.sum(likelihoods)/N



N = 1000
param_model1=np.array([np.random.uniform(0,1E-2,N),np.random.uniform(0,20,N)])
param_model2=np.array([np.random.uniform(0,2,N),np.random.uniform(0,3,N)])
param_model3=np.array([ np.random.uniform(0,10,N),np.random.uniform(-10,10,N), np.random.uniform(0,0.5,N)])
result_model1 = montecarlo(expenditure,years_lived,N,param_model1,model1)
result_model2 = montecarlo(expenditure,years_lived,N,param_model2,model2)
result_model3= montecarlo(expenditure,years_lived,N,param_model3,model3)


print('Modelo 1:',str(result_model1))
print('Modelo 2:',str(result_model2))
print('Modelo 3:',str(result_model3))
