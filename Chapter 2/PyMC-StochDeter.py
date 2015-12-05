# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 15:33:41 2015

@author: Usamahk
"""

# Testing the PyMC function. Learning about stochastic and deterministic
# variables. From Cam Pilon's Book

import pymc as pm
import numpy as np

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

# Take the case of the sms data in the previous chapter, knowing what we do
# about parent and child variables and taking an omniscient view on the data
# and determining a modeling procedure we can work backwards to create the 
# data mimicing the expected creation of the data. i.e.

tau = pm.rdiscrete_uniform(0, 80)
print( tau )

alpha = 1. / 20.
lambda_1, lambda_2 = pm.rexponential(alpha, 2)
print( lambda_1, lambda_2)

data = np.r_[pm.rpoisson(lambda_1, tau), pm.rpoisson(lambda_2, 80 - tau)]

# Plot the distribution

plt.bar(np.arange(80), data, color="#348ABD")
plt.bar(tau - 1, data[tau - 1], color="r", label="user behaviour changed")
plt.xlabel("Time (days)")
plt.ylabel("count of text-msgs received")
plt.title("Artificial dataset")
plt.xlim(0, 80)
plt.legend();

# This becomes important, I assume when we start checking to see if our
# inference was indeed correct. If we were to create a function then this
# would be the case:

def plot_artificial_sms_dataset():
    tau = pm.rdiscrete_uniform(0, 80)
    alpha = 1. / 20.
    lambda_1, lambda_2 = pm.rexponential(alpha, 2)
    data = np.r_[pm.rpoisson(lambda_1, tau), pm.rpoisson(lambda_2, 80 - tau)]
    plt.bar(np.arange(80), data, color="#348ABD")
    plt.bar(tau - 1, data[tau - 1], color="r", label="user behaviour changed")
    plt.xlim(0, 80)
    
# Set up for Bayesian A/B testing. In this case, 2 website designs, A and B
# which one is better? we have N number of users coming and p_A/p_B buying
# from the site. We do not know prior probabilities in practice but assume
# here only for p_A

p = pm.Uniform('p', lower=0, upper=1)
   
# set constants
p_true = 0.05  # remember, this is unknown.
N = 1500

# sample N Bernoulli random variables from Ber(0.05).
# each random variable has a 0.05 chance of being a 1.
# this is the data-generation step
occurrences = pm.rbernoulli(p_true, N)

print (occurrences)  # Remember: Python treats True == 1, and False == 0
print (occurrences.sum())

# Occurrences.mean is equal to n/N.
print ("What is the observed frequency in Group A? %.4f" % occurrences.mean())
print ("Does this equal the true frequency? %s" % (occurrences.mean()==p_true))

# Combine 2 into inference algorithm

# include the observations, which are Bernoulli
obs = pm.Bernoulli("obs", p, value=occurrences, observed=True)

# To be explained in chapter 3
mcmc = pm.MCMC([p, obs])
mcmc.sample(18000, 1000)

figsize(12.5, 4)
plt.title("Posterior distribution of $p_A$, the true effectiveness of site A")
plt.vlines(p_true, 0, 90, linestyle="--", label="true $p_A$ (unknown)")
plt.hist(mcmc.trace("p")[:], bins=25, histtype="stepfilled", normed=True)
plt.legend()

# If we combine the 2 we can model it and compare as such. Assume p_A, p_B and
# Delta = p_A - p_B

figsize(12, 4)

# these two quantities are unknown to us.
true_p_A = 0.05
true_p_B = 0.04

# notice the unequal sample sizes -- no problem in Bayesian analysis.
N_A = 1500
N_B = 750

# generate some observations
observations_A = pm.rbernoulli(true_p_A, N_A)
observations_B = pm.rbernoulli(true_p_B, N_B)
print( "Obs from Site A: ", observations_A[:30].astype(int), "...")
print( "Obs from Site B: ", observations_B[:30].astype(int), "...")

print (observations_A.mean())
print (observations_B.mean())

# Set up the pymc model. Again assume Uniform priors for p_A and p_B.
p_A = pm.Uniform("p_A", 0, 1)
p_B = pm.Uniform("p_B", 0, 1)


# Define the deterministic delta function. This is our unknown of interest.
@pm.deterministic
def delta(p_A=p_A, p_B=p_B):
    return p_A - p_B

# Set of observations, in this case we have two observation datasets.
obs_A = pm.Bernoulli("obs_A", p_A, value=observations_A, observed=True)
obs_B = pm.Bernoulli("obs_B", p_B, value=observations_B, observed=True)

# To be explained in chapter 3.
mcmc = pm.MCMC([p_A, p_B, delta, obs_A, obs_B])
mcmc.sample(20000, 1000)

p_A_samples = mcmc.trace("p_A")[:]
p_B_samples = mcmc.trace("p_B")[:]
delta_samples = mcmc.trace("delta")[:]

figsize(12.5, 10)

# histogram of posteriors

ax = plt.subplot(311)

plt.xlim(0, .1)
plt.hist(p_A_samples, histtype='stepfilled', bins=25, alpha=0.85,
         label="posterior of $p_A$", color="#A60628", normed=True)
plt.vlines(true_p_A, 0, 80, linestyle="--", label="true $p_A$ (unknown)")
plt.legend(loc="upper right")
plt.title("Posterior distributions of $p_A$, $p_B$, and delta unknowns")

ax = plt.subplot(312)

plt.xlim(0, .1)
plt.hist(p_B_samples, histtype='stepfilled', bins=25, alpha=0.85,
         label="posterior of $p_B$", color="#467821", normed=True)
plt.vlines(true_p_B, 0, 80, linestyle="--", label="true $p_B$ (unknown)")
plt.legend(loc="upper right")

ax = plt.subplot(313)
plt.hist(delta_samples, histtype='stepfilled', bins=30, alpha=0.85,
         label="posterior of delta", color="#7A68A6", normed=True)
plt.vlines(true_p_A - true_p_B, 0, 60, linestyle="--",
           label="true delta (unknown)")
plt.vlines(0, 0, 60, color="black", alpha=0.2)
plt.legend(loc="upper right");

# Count the number of samples less than 0, i.e. the area under the curve
# before 0, represent the probability that site A is worse than site B.
print( "Probability site A is WORSE than site B: %.3f" % \
    (delta_samples < 0).mean())

print( "Probability site A is BETTER than site B: %.3f" % \
    (delta_samples > 0).mean())