#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 21 18:33:48 2026

@author: Onerva Korhonen

A script for calculating the probability for a nanobubble to expand and fill
the entire conduit that it occupies (i.e., the bubble expansion probability). The
probability is calculated for a set of bubble surface pressures (i.e., pressure
differences between the sap outside the bubble and air inside it) following
the formalism of Ingram et al., 2024.

"""
import numpy as np
import pickle

import params
import bubble_expansion

T = 300.0 # K
apl = 0.6 # 'equilibrium' area per lipid, nm^-2
mean_r = 190 # mean bubble radius, nm
mu = np.log(mean_r)
sigma = 0.6 # standard deviation in natural log units
n_bubble = 5000 # number of bubbles in sample
 
pressure_range = np.arange(0, 8, step=0.01)*1E6

n_iterations = 1

save_path = params.bubble_propagation_pressure_data_path

# lets not go crazy here
assert mu - 3*sigma > 1

# radii values to evaluate the Gibbs free energy at
r_range = np.logspace(
    start=mu - 3*sigma,
    stop=mu + 6*sigma,
    base=np.e,
    num=500
    )
 
expansion_probabilities = {}

for pressure in pressure_range:
    expansion_probabilities_per_pressure = []
    for i in range(n_iterations):
        expansion_probability = bubble_expansion.probability(-pressure, T, mu, sigma, apl, r_range/1e9, n_bubble)
        expansion_probabilities_per_pressure.append(expansion_probability)
    expansion_probabilities[pressure] = np.mean(expansion_probabilities_per_pressure)

with open(save_path, 'wb') as f:
    pickle.dump(expansion_probabilities, f)
f.close()
     

