# -*- coding: utf-8 -*-
"""
Created on Sun Aug 13 19:46:05 2023

@author: Luca
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from neurodiffeq.solvers import BundleSolution1D
from neurodiffeq.conditions import BundleIVP
import torch

# Load the networks:
nets = torch.load('nets_LCDM.ph',
                  map_location=torch.device('cpu')  # Needed if trained on GPU but this sciprt is executed on CPU
                  )

# Define the reparametrizations that enforce the initial conditions:

Om_m=0.272
Om_r=9.24*10**(-5)
Om_L=1-Om_r-Om_m
sigma8=0.8
a_eq=Om_r/Om_m #Equality between matter and radiation
B=a_eq**3*Om_L/Om_m
#B=0
a_0 = 10**(-3)
a_f = 1.0
y_0=a_0/a_eq
y_f=1/a_eq

condition = [BundleIVP(a_0, a_0),
             BundleIVP(a_0, 1)]

# Incorporate the nets and the reparametrizations into a solver:
    
x = BundleSolution1D(nets, condition)


# The Hubble parameter as a function of the dependent variables of the system:

def delta(a):
    deltas = x(a, to_numpy=True)[0]
    return deltas


# Plot the Hubble parameter for different values of the independent variable an its parameters:

a = np.linspace(a_0, a_f,200)
delta_ann=delta(a)
plt.plot(a, delta_ann, label='ANN')
plt.legend()


#Defino la función del sistema de ecuaciones

def F(t,X):
    f1=X[1] 
    #f2=(3/(2*t*(1+t)))*X[0]-((2+3*t)/(2*t*(1+t)))*X[1]  #Kamionkowski
    f2=(3/(2*t*(1+t+B*t**4)))*X[0]-((2+3*t+6*B*t**4)/(2*t*(1+t+B*t**4)))*X[1]
    return np.array([f1,f2])

y=a/a_eq
#y=np.linspace(y_0,y_f,1000000)
atol, rtol = 1e-15, 1e-12
#Perform the backwards-in-time integration
out2 = solve_ivp(fun = F, t_span = [y_0,y_f], y0 = np.array([a_0,a_eq]),
                t_eval = y, method = 'RK45')

delta_num=out2.y[0]
delta_p=out2.y[1]

plt.plot(y*a_eq,delta_num,label='Numerical')
#plt.plot(y,delta)
plt.xlabel('a')
plt.ylabel(r'$\delta_m$')
#plt.yscale('log')
plt.legend()
#plt.plot(N,delta_p)


dif_rel=[]
for i in range(len(a)):
    dif_rel.append(100*np.abs(delta_ann[i]-delta_num[i])/np.abs(delta_num[i]))
plt.figure()
plt.plot(a,dif_rel)
plt.xlabel('a')
plt.ylabel('err%')