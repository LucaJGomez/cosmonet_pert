# -*- coding: utf-8 -*-
"""
Created on Sun Aug 13 16:18:21 2023

@author: Luca
"""

# Import libraries:
import matplotlib.pyplot as plt
from neurodiffeq.solvers import BundleSolver1D
from neurodiffeq.conditions import BundleIVP
from neurodiffeq import diff  # the differentiation operation
import torch
from neurodiffeq.generators import Generator1D
import numpy as np
import torch.nn as nn
from neurodiffeq.networks import FCNN
# Set the parameters of the problem

Om_r_0=9.24*10**(-5)
Om_m_0=0.272
a_eq=Om_r_0/Om_m_0
Om_L_0=1-Om_m_0-Om_r_0
B=a_eq**3*Om_L_0/Om_m_0
#B=0


# Set the range of the independent variable:

a_0 = 10**(-3)
a_f = 1.0

# Define the differential equation:
    
def ODE_LCDM(delta, delta_prime, a):
    r"""Function that defines the differential equation of the system, by defining the residual of it. In this case:
    :math:`\displaystyle \mathcal{R}\left(\tilde{x},z\right)=\dfrac{d\tilde{x}}{dz} - \dfrac{3\tilde{x}}{1+z}.`
    :param x: The reparametrized output of the network corresponding to the dependent variable.
    :type x: `torch.Tensor`.
    :param z: The independent variable.
    :type z: `torch.Tensor`.
    :return: The residual of the differential equation.
    :rtype: list[`torch.Tensor`].
    """
    res1 = diff(delta, a) - delta_prime
    #res2 = a_eq**2*diff(delta_prime, a) - (3/(2*(a/a_eq)*(1+(a/a_eq)+B*(a/a_eq)**4)))*delta + ((2+3*(a/a_eq)+6*B*(a/a_eq)**4)/(2*(a/a_eq)*(1+(a/a_eq)+B*(a/a_eq)**4)))*a_eq*delta_prime
    res2 = a_eq*diff(delta_prime, a) - (3/(2*a*(1+(a/a_eq)+B*(a/a_eq)**4)))*delta + ((2+3*(a/a_eq)+6*B*(a/a_eq)**4)/(2*(a)*(1+(a/a_eq)+B*(a/a_eq)**4)))*a_eq*delta_prime

    
    #print(delta_prime)
    
    return [res1 , res2]

# Define the initial condition:

condition = [BundleIVP(a_0, a_0),
             BundleIVP(a_0, 1)]

# Define a custom loss function:

def weighted_loss_LCDM(res, x, t):
    r"""A custom loss function. While the default loss is the square of the residual,
    here a weighting function is added:
    :math:`\displaystyle L\left(\tilde{x},z\right)=\mathcal{R}\left(\tilde{x},z\right)^2e^{-2\left(z-z_0\right)}.`
    :param res: The residuals of the differential equation.
    :type res: `torch.Tensor`.
    :param x: The reparametrized output of the network corresponding to the dependent variable.
    :type x: `torch.Tensor`.
    :type t: The inputs of the neural network: i.e, the independent variable and the parameter of the bundle.
    :param t: list[`torch.Tensor`, `torch.Tensor`].
    :return: The mean value of the loss across the training points.
    :rtype: `torch.Tensor`.
    """
    a = t[0]
    w = 2

    #loss = (res ** 2) * torch.exp(-w * (a - a_0))
    loss = (res ** 2)
    return loss.mean()

nets = [FCNN(n_input_units=1,  hidden_units=(32,32,)) for _ in range(2)]


adam = torch.optim.Adam(set([p for net in nets for p in net.parameters()]), \
                        lr=5e-3,  betas=(0.9, 0.999))
    
# Define the ANN based solver:
    
solver = BundleSolver1D(ode_system=ODE_LCDM,
                        conditions=condition,
                        t_min=a_0, t_max=a_f,
#                        optimizer=adam,
                        loss_fn=weighted_loss_LCDM,
                        )

# Set the amount of interations to train the solver:
iterations = 700000

# Start training:
solver.fit(iterations)

# Plot the loss during training, and save it:
loss = solver.metrics_history['train_loss']
plt.plot(loss, label='training loss')
plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
plt.xlabel('iterations')
plt.ylabel('Loss')
plt.yscale('log')
plt.legend()
plt.suptitle('Total loss during training')
plt.savefig('loss_LCDM.png')

# Save the neural network:
torch.save(solver._get_internal_variables()['best_nets'], 'nets_LCDM.ph')