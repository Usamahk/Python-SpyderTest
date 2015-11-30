# -*- coding: utf-8 -*-
"""
Created on Sat Nov 28 13:59:47 2015

@author: Usamahk
"""

# All the code below is taken from Probabilistic Programming and Bayesian
# Methods for Hackers by Cameron Davidson Pilon. I own no part of it, just
# isolating the examples so I can follow better

# These scripts are taken from his code chunks. Ths example deals with
# text message data over the course of a 2 months or so. The hypothesis
# is that at some point the users behaviour changed. Can we infer this?

# Import all libraries

from IPython.core.pylabtools import figsize
import numpy as np
from matplotlib import pyplot as plt
import pymc as pm

# Set figsize 

figsize(12.5, 3.5)

# Load data

count_data = np.loadtxt("txtdata.csv")
n_count_data = len(count_data)

# Making a plot of daily messaging

plt.bar(np.arange(n_count_data), count_data, color="#348ABD")
plt.xlabel("Time (days)")
plt.ylabel("count of text-msgs received")
plt.title("Did the user's texting habits change over time?")
plt.xlim(0, n_count_data);

# Defining all the variables, alpha, lambda_1 and lambda_2

alpha = 1.0 / count_data.mean()  

lambda_1 = pm.Exponential("lambda_1", alpha)
lambda_2 = pm.Exponential("lambda_2", alpha)

tau = pm.DiscreteUniform("tau", lower=0, upper=n_count_data)

# Define a function for lambda_, another random variable

@pm.deterministic
def lambda_(tau=tau, lambda_1=lambda_1, lambda_2=lambda_2):
    out = np.zeros(n_count_data)
    out[:tau] = lambda_1  # lambda before tau is lambda1
    out[tau:] = lambda_2  # lambda after (and including) tau is lambda2
    return out
    
# Observation - combines count data with data generation scheme
    
observation = pm.Poisson("obs", lambda_, value=count_data, observed=True)

# Creates a model

model = pm.Model([observation, lambda_1, lambda_2, tau])

# Mysterious code

mcmc = pm.MCMC(model)
mcmc.sample(40000, 10000, 1)

# lambda_1 and lambda_2 samples

lambda_1_samples = mcmc.trace('lambda_1')[:]
lambda_2_samples = mcmc.trace('lambda_2')[:]
tau_samples = mcmc.trace('tau')[:]

# Find mean

lambda_1_samples.mean()
lambda_2_samples.mean()

# Find Percentage increase

lambda_1_samples/lambda_2_samples

# At this point plot the density functions

# Set the figsize

figsize(12.5, 10)

# histogram of the samples:

ax = plt.subplot(311)
ax.set_autoscaley_on(False)

plt.hist(lambda_1_samples, histtype='stepfilled', bins=30, alpha=0.85,
         label="posterior of $\lambda_1$", color="#A60628", normed=True)
plt.legend(loc="upper left")
plt.title(r"""Posterior distributions of the variables
    $\lambda_1,\;\lambda_2,\;\tau$""")
plt.xlim([15, 30])
plt.xlabel("$\lambda_1$ value")

ax = plt.subplot(312)
ax.set_autoscaley_on(False)
plt.hist(lambda_2_samples, histtype='stepfilled', bins=30, alpha=0.85,
         label="posterior of $\lambda_2$", color="#7A68A6", normed=True)
plt.legend(loc="upper left")
plt.xlim([15, 30])
plt.xlabel("$\lambda_2$ value")

plt.subplot(313)
w = 1.0 / tau_samples.shape[0] * np.ones_like(tau_samples)
plt.hist(tau_samples, bins=n_count_data, alpha=1,
         label=r"posterior of $\tau$",
         color="#467821", weights=w, rwidth=2.)
plt.xticks(np.arange(n_count_data))

plt.legend(loc="upper left")
plt.ylim([0, .75])
plt.xlim([35, len(count_data) - 20])
plt.xlabel(r"$\tau$ (in days)")
plt.ylabel("probability");

# Plot the Density and the changes

figsize(12.5, 5)

N = tau_samples.shape[0]
expected_texts_per_day = np.zeros(n_count_data)
for day in range(0, n_count_data):
  
    ix = day < tau_samples
  
    expected_texts_per_day[day] = (lambda_1_samples[ix].sum()
                                   + lambda_2_samples[~ix].sum()) / N


plt.plot(range(n_count_data), expected_texts_per_day, lw=4, color="#E24A33",
         label="expected number of text-messages received")
plt.xlim(0, n_count_data)
plt.xlabel("Day")
plt.ylabel("Expected # text-messages")
plt.title("Expected number of text-messages received")
plt.ylim(0, 60)
plt.bar(np.arange(len(count_data)), count_data, color="#348ABD", alpha=0.65,
        label="observed texts per day")

plt.legend(loc="upper left");