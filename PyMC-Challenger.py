# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 19:29:51 2015

@author: Usamahk
"""

# This file examines the challenger data to determine and model where failure
# of an O ring will occur due to temperature.

# Libraries we will need

from IPython.core.pylabtools import figsize
from matplotlib import pyplot as plt
import pymc as pm
import numpy as np

# Loading the data and setting figsize as a standard
figsize(12.5, 3.5)
np.set_printoptions(precision = 3, suppress = True)
challenger_data = np.genfromtxt("data/challenger_data.csv", skip_header = 1,
                                usecols = [1,2], missing_values = "NA",
                                delimiter = ",")

# Print to see data
print("Temp (F), O-Ring Failure")
print(challenger_data)

# Now that we've read in the data, we need to drop all the NA values, in this
# case only one at the end. Should do this before in practice but getting a 
# hang of things here.

challenger_data = challenger_data[~np.isnan(challenger_data[:,1])]

# Print to see cleaned data
print("Temp (F), O-Ring Failure")
print(challenger_data)

# Plot the data

plt.scatter(challenger_data[:,0], challenger_data[:,1], s = 75, color = "k",
            alpha = 0.5)
plt.yticks([0,1])
plt.ylabel("Damage Incident?")
plt.xlabel("Outside Temperature (Fahrenheit)")
plt.title("Defects of the Space Shuttle O-Rings vs Temperature")

# We can see no ready cutoff point as to where the temperature starts affecting
# the stability of the O-Ring. The best we can do is offere a probability. Use
# a function, in this case, a logistic function

################## Refresher on Logistic Functions ############################

figsize(12, 3)

def logistic(x, beta):
    return 1.0 / (1.0 + np.exp(beta * x))
    
x = np.linspace(-4, 4, 100)
plt.plot(x, logistic(x, 1), label=r"$\beta = 1$")
plt.plot(x, logistic(x, 3), label=r"$\beta = 3$")
plt.plot(x, logistic(x, -5), label=r"$\beta = -5$")
plt.legend();

# Here we see that the probability basically only changes around 0. We want
# to move it up so we can add a parameter to the logistic function, alpha.

def logistic(x, beta, alpha=0):
    return 1.0 / (1.0 + np.exp(np.dot(beta, x) + alpha))

x = np.linspace(-4, 4, 100)

plt.plot(x, logistic(x, 1), label=r"$\beta = 1$", ls="--", lw=1)
plt.plot(x, logistic(x, 3), label=r"$\beta = 3$", ls="--", lw=1)
plt.plot(x, logistic(x, -5), label=r"$\beta = -5$", ls="--", lw=1)

plt.plot(x, logistic(x, 1, 1), label=r"$\beta = 1, \alpha = 1$",
         color="#348ABD")
plt.plot(x, logistic(x, 3, -2), label=r"$\beta = 3, \alpha = -2$",
         color="#A60628")
plt.plot(x, logistic(x, -5, 7), label=r"$\beta = -5, \alpha = 7$",
         color="#7A68A6")

plt.legend(loc="lower left");

# The alpha is known as the Bias. What we know is for our problem for alpha,
# it is not positive, and not bounded but not large so model with a Normal
# distribution. We can see here a couple of random distributions

import scipy.stats as stats

nor = stats.norm
x = np.linspace(-8, 7, 150)
mu = (-2, 0, 3)
tau = (.7, 1, 2.8)
colors = ["#348ABD", "#A60628", "#7A68A6"]
parameters = zip(mu, tau, colors)

for _mu, _tau, _color in parameters:
    plt.plot(x, nor.pdf(x, _mu, scale=1. / _tau),
             label="$\mu = %d,\;\\tau = %.1f$" % (_mu, _tau), color=_color)
    plt.fill_between(x, nor.pdf(x, _mu, scale=1. / _tau), color=_color,
                     alpha=.33)

plt.legend(loc="upper right")
plt.xlabel("$x$")
plt.ylabel("density function at $x$")
plt.title("Probability distribution of three different Normal random \
variables");

# Look at the graphs. A smaller tau leads to a larger spread, i.e more 
# uncertain and larger means more precise, which is why is referred to as 
# the precision variable. mu is simply the mean.
# A normal variable can take on any number, but it will tend to the mean. So
# the expected value can be thought of as the mean and variance the inverse 
# of tau

################## Refresher on Logistic Functions ############################

temperature = challenger_data[:, 0]
D = challenger_data[:, 1]  # defect or not?

# notice the`value` here. We explain why below.
beta = pm.Normal("beta", 0, 0.001, value=0)
alpha = pm.Normal("alpha", 0, 0.001, value=0)

@pm.deterministic
def p(t=temperature, alpha=alpha, beta=beta):
    return 1.0 / (1. + np.exp(beta * t + alpha))

# We set the values for alpha and beta as 0 due to computation. We want an
# initial guess where alpha and beta are not too large or too small to affect
# a p value of 0 or 1 which pm.bernoulli does not like

observed = pm.Bernoulli("bernoulli_obs", p, value=D, observed=True)

model = pm.Model([observed, beta, alpha])

# Mysterious code to be explained in Chapter 3
map_ = pm.MAP(model)
map_.fit()
mcmc = pm.MCMC(model)
mcmc.sample(120000, 100000, 2)

# At this point we have trained the code on the set. Now lets take a look at
# the posterior distributions we calculated

alpha_samples = mcmc.trace('alpha')[:, None]  # best to make them 1d
beta_samples = mcmc.trace('beta')[:, None]

figsize(12.5, 6)

# histogram of the samples:
plt.subplot(211)
plt.title(r"Posterior distributions of the variables $\alpha, \beta$")
plt.hist(beta_samples, histtype='stepfilled', bins=35, alpha=0.85,
         label=r"posterior of $\beta$", color="#7A68A6", normed=True)
plt.legend()

plt.subplot(212)
plt.hist(alpha_samples, histtype='stepfilled', bins=35, alpha=0.85,
         label=r"posterior of $\alpha$", color="#A60628", normed=True)
plt.legend();