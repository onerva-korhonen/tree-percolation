"""
A script for calculating some properties of the simulated xylem networks.

Written by Onerva Korhonen
"""
import params
import mrad_model
import mrad_params

import numpy as np
import pickle
import networkx as nx
import matplotlib.pylab as plt
from scipy.stats import binned_statistic

conduit_element_length = params.Lce
use_cylindrical_coords = False
heartwood_d = mrad_params.heartwood_d
cec_indicator = mrad_params.cec_indicator

network_save_path_base = params.spreading_probability_optimization_network_save_path_base
iteration_indices = np.arange(0, 100)
n_bins = 10

average_degrees = []
densities = []

fig = plt.figure()
ax = fig.add_subplot(111)

#import pdb; pdb.set_trace()

for iteration_index in iteration_indices:
    network_save_path = network_save_path_base + '_' + str(iteration_index) + '.pkl'
    with open(network_save_path, 'rb') as f:
        network_data = pickle.load(f)
        f.close()
    net = network_data['network']
    conduit_elements = mrad_model.get_conduit_elements(net=net, use_cylindrical_coords=use_cylindrical_coords, 
                                                       conduit_element_length=conduit_element_length, 
                                                       heartwood_d=heartwood_d, cec_indicator=cec_indicator)
    throats = net.get('throat.conns', [])
    n_pores = net['pore.coords'].shape[0]

    throat_conduits = []
    conduit_indices = []
    if len(throats) > 0:
        for i, throat in enumerate(throats):
            start_conduit = conduit_elements[throat[0], 3]
            end_conduit = conduit_elements[throat[1], 3]
            if not start_conduit in conduit_indices:
                conduit_indices.append(start_conduit)
            if not end_conduit in conduit_indices:
                conduit_indices.append(end_conduit)
            if not start_conduit == end_conduit:
                throat_conduits.append((start_conduit, end_conduit))
        n_conduits = len(conduit_indices)
        G = nx.Graph()
        G.add_nodes_from(conduit_indices)
        G.add_edges_from(throat_conduits)
        degrees = [element[1] for element in list(G.degree())]
        average_degrees.append(np.mean(degrees))
        densities.append(nx.density(G))

        degree_distribution, bin_edges, _ = binned_statistic(degrees, degrees, statistic='count', bins=n_bins)
        degree_distribution /= float(np.sum(degree_distribution * np.abs(bin_edges[0] - bin_edges[1])))

        bin_centers = 0.5*(bin_edges[:-1] + bin_edges[1:])

        plt.plot(bin_centers, degree_distribution)

average_average_degree = np.mean(average_degrees)
std_average_degree = np.std(average_degrees)

average_density = np.mean(densities)
std_density = np.std(densities)

print(f'Average degree: {average_average_degree} +/- {std_average_degree}')
print(f'Average density: {average_density} +/- {std_density}')

ax.set_xlabel('Degree')
ax.set_ylabel('PDF')
figure_save_path = params.degree_distribution_fig_save_path
plt.savefig(figure_save_path, format='pdf', bbox_inches='tight')
