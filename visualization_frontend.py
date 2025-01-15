# a script for visualizing percolation outcomes based on the percolation data calculated with slurm

import params
import visualization

import pickle
import numpy as np
import matplotlib.pylab as plt

visualize_single_param_spreading = True
visualize_vc = False
visualize_optimized_vc = False

if visualize_single_param_spreading:
    
    prevalence_fig = plt.figure()
    prevalence_ax = prevalence_fig.add_subplot(111)
    
    lcc_fig = plt.figure()
    lcc_ax = lcc_fig.add_subplot(111)
    
    percolation_outcome_values_all = []
    nonfunctional_component_volumes_all = []
    n_inlets_all = []
    total_n_nodes_all = []
    x_all = []
    
    for path, line_style, label in zip(params.single_param_visualization_data_paths, params.percolation_linestyles, params.percolation_labels):

        f = open(path, 'rb')
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
        x = data['x']
        
        prevalence_ax.plot(x, ls=line_style, label=label)
        lcc_ax.plot(lcc_sizes, ls=line_style, label=label)
        
        
    
        percolation_outcome_values = np.concatenate((np.expand_dims(effective_conductances, axis=0), 
                                                     np.expand_dims(lcc_sizes, axis=0), np.expand_dims(functional_lcc_sizes, axis=0)),
                                                     axis=0)
        
        percolation_outcome_values_all.append(percolation_outcome_values)
        nonfunctional_component_volumes_all.append(np.expand_dims(nonfunctional_component_volume, axis=0))
        n_inlets_all.append(np.concatenate((np.expand_dims(n_inlets, axis=0), np.expand_dims(n_outlets, axis=0)), axis=0))
        total_n_nodes_all.append(data['total_n_nodes'])
        x_all.append(x)
        
    visualization.plot_percolation_curve(total_n_nodes_all, percolation_outcome_values_all,
                                         colors=params.percolation_outcome_colors, labels=params.percolation_outcome_labels, 
                                         alphas=params.percolation_outcome_alphas, y_labels=params.percolation_outcome_ylabels,
                                         axindex=params.percolation_outcome_axindex, save_path=params.percolation_plot_save_path, xs=x_all, 
                                         param_linestyles=params.percolation_linestyles, param_labels=params.percolation_labels)
    visualization.plot_percolation_curve(total_n_nodes_all, nonfunctional_component_volumes_all,
                                         colors=[params.percolation_nonfunctional_component_size_color], labels=[params.percolation_nonfunctional_component_size_label], 
                                         alphas=[params.percolation_nonfunctional_component_size_alpha], save_path=params.nonfunctional_component_size_save_path, xs=x_all,
                                         param_linestyles=params.percolation_linestyles, param_labels=params.percolation_labels)
    visualization.plot_percolation_curve(total_n_nodes_all, n_inlets_all,
                                         colors=[params.percolation_ninlet_color, params.percolation_noutlet_color],
                                         labels=[params.percolation_ninlet_label, params.percolation_noutlet_label],
                                         alphas=[params.percolation_ninlet_alpha, params.percolation_noutlet_alpha],
                                         save_path=params.ninlet_save_path, xs=x_all,
                                         param_linestyles=params.percolation_linestyles, param_labels=params.percolation_labels)

    prevalence_ax.set_xlabel('time')
    prevalence_ax.set_ylabel('prevalence (fraction of embolized conduits)')
    plt.savefig(params.prevalence_save_path, format='pdf',bbox_inches='tight')
    
    lcc_ax.set_xlabel('time')
    lcc_ax.set_ylabel('LCC')
    plt.savefig(params.lcc_in_time_save_path, format='pdf', bbox_inches='tight')
    
if visualize_vc:
    f = open(params.vc_data_save_path, 'rb')
    vc = pickle.load(f)
    f.close()
    
    visualization.plot_vulnerability_curve(vc, params.physiological_vc_color, params.physiological_vc_alpha, vc_type=params.si_type, save_path=params.vc_plot_save_path)
    
if visualize_optimized_vc:
    data_save_folders = [base.rsplit('/', 1)[0] for base in params.optimized_vc_plot_data_save_path_bases]
    visualization.plot_optimized_vulnerability_curve(data_save_folders, params.physiological_vc_color, params.stochastic_vc_color, 
                                                     params.physiological_vc_alpha, params.stochastic_vc_alpha, params.optimized_vc_linestyles, params.optimized_vc_labels,
                                                     params.optimized_vc_plot_save_path, pooled_data=True, pooled_data_save_name=params.pooled_optimized_spreading_probability_save_name,
                                                     std_alpha=params.std_alpha, prevalence_linestyles=params.prevalence_linestyles, 
                                                     prevalence_plot_save_path_base=params.optimized_prevalence_plot_save_path_base)
