# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 15:33:41 2015

@author: Usamahk
"""

# Testing the PyMC function. Learning about stochastic and deterministic
# variables. From Cam Pilon's Book

import pymc as pm

# Determining a stochastic value - random with no influences from 
# parent variables

lambda_1 = pm.Exponential("lambda_1", 1)  # prior on first behaviour
lambda_2 = pm.Exponential("lambda_2", 1)  # prior on second behaviour
tau = pm.DiscreteUniform("tau", lower=0, upper=10)  # prior on behaviour change

print ("lambda_1.value = %.3f" % lambda_1.value)
print ("lambda_2.value = %.3f" % lambda_2.value)
print ("tau.value = %.3f" % tau.value)

lambda_1.random(), lambda_2.random(), tau.random()

print ("After calling random() on the variables...")
print ("lambda_1.value = %.3f" % lambda_1.value)
print ("lambda_2.value = %.3f" % lambda_2.value)
print ("tau.value = %.3f" % tau.value)

# Note: - Don't change values in-place. It messes with PyMCs caching

# Defining a deterministic value - Values dependent on lambda_1 and lambda_2
# If we have that information then we can use it to determine lambda

import numpy as np
n_data_points = 5  # in CH1 we had ~70 data points


@pm.deterministic
def lambda_(tau=tau, lambda_1=lambda_1, lambda_2=lambda_2):
    out = np.zeros(n_data_points)
    out[:tau] = lambda_1  # lambda before tau is lambda1
    out[tau:] = lambda_2  # lambda after tau is lambda2
    return out

# Here, what does prior distribution look like?
  
%matplotlib inline
from IPython.core.pylabtools import figsize
from matplotlib import pyplot as plt
figsize(12.5, 4)


samples = [lambda_1.random() for i in range(20000)]
plt.hist(samples, bins=70, normed=True, histtype="stepfilled")
plt.title("Prior distribution for $\lambda_1$")
plt.xlim(0, 8);