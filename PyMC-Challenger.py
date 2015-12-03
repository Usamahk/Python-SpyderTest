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