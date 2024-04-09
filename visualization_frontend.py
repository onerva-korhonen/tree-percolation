# a script for visualizing percolation outcomes based on the percolation data calculated with slurm

import params
import visualization

import pickle
import numpy as np
import matplotlib.pylab as plt

f = open(params.percolation_data_save_path, 'rb')
data = pickle.load(f)
f.close()

effective_conductances = data['effective_conductances']
lcc_sizes = data['lcc_sizes']
functional_lcc_sizes = data['functional_lcc_sizes']
nonfunctional_component_size = data['nonfunctional_component_size']
susceptibilities = data['susceptibilities']
functional_susceptibilities = data['functional_susceptibilities']
n_inlets = data['n_inlets']
n_outlets = data['n_outlets']
nonfunctional_component_volume = data['nonfunctional_component_volume']
total_n_nodes = data['total_n_nodes']
x = data['x']

percolation_outcome_values = np.concatenate((np.expand_dims(effective_conductances, axis=0), 
                                             np.expand_dims(lcc_sizes, axis=0), np.expand_dims(functional_lcc_sizes, axis=0)),
                                             axis=0)

visualization.plot_percolation_curve(total_n_nodes, percolation_outcome_values,
                                     colors=params.percolation_outcome_colors, labels=params.percolation_outcome_labels, 
                                     alphas=params.percolation_outcome_alphas, y_labels=params.percolation_outcome_ylabels,
                                     axindex=params.percolation_outcome_axindex, save_path=params.percolation_plot_save_path, x=x)
visualization.plot_percolation_curve(total_n_nodes, np.expand_dims(nonfunctional_component_volume, axis=0),
                                     colors=[params.percolation_nonfunctional_component_size_color], labels=[params.percolation_nonfunctional_component_size_label], 
                                     alphas=[params.percolation_nonfunctional_component_size_alpha], save_path=params.nonfunctional_componen_size_save_path, x=x)
visualization.plot_percolation_curve(total_n_nodes, 
                                     np.concatenate((np.expand_dims(n_inlets, axis=0), np.expand_dims(n_outlets, axis=0)), axis=0),
                                     colors=[params.percolation_ninlet_color, params.percolation_noutlet_color],
                                     labels=[params.percolation_ninlet_label, params.percolation_noutlet_label],
                                     alphas=[params.percolation_ninlet_alpha, params.percolation_noutlet_alpha],
                                     save_path=params.ninlet_save_path, x=x)
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(x)
ax.set_xlabel('time')
ax.set_ylabel('prevalence (fraction of embolized conduits)')
plt.savefig(params.prevalence_save_path, format='pdf',bbox_inches='tight')

fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(lcc_sizes)
ax.set_xlabel('time')
ax.set_ylabel('LCC')
plt.savefig(params.lcc_in_time_save_path, format='pdf', bbox_inches='tight')
