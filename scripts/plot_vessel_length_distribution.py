"""
A script for calculating and plotting the distribution of vessel (conduit) lengths in networks used for simulating embolism spreading.

Written by Onerva Korhonen
"""
import numpy as np
import matplotlib.pylab as plt
import pickle
from scipy.stats import binned_statistic

import params
import mrad_model
import mrad_params

n_networks = 100
conduit_lengths = []
n_bins = 50

cec_indicator = mrad_params.cec_indicator
Lce = params.Lce # conduit element length in m

figure_save_path = params.conduit_length_distribution_save_path

for i in range(n_networks):
    network_save_path = params.spreading_probability_optimization_network_save_path_base + '_' + str(i) + '.pkl'
    with open(network_save_path, 'rb') as f:
        network_data = pickle.load(f)
        f.close()
    net = network_data['network']
    conns = net['throat.conns']
    conn_types = net['throat.type']
    cec_mask = conn_types == cec_indicator
    cec = conns[cec_mask]
    conduits = mrad_model.get_conduits(cec)
    conduit_lengths.extend(conduits[:,-1] * Lce)

length_distribution, bin_edges, _ = binned_statistic(conduit_lengths, conduit_lengths, statistic='count', bins=n_bins)
length_distribution /= float(np.sum(length_distribution) * np.abs(bin_edges[0] - bin_edges[1]))

bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:]) 

fig = plt.figure()
ax = fig.add_subplot(111)

plt.plot(bin_centers, length_distribution)
ax.set_ylabel('PDF')
ax.set_xlabel('conduit length (m)')
plt.savefig(figure_save_path, format='pdf', bbox_inches='tight')








