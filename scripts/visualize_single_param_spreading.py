"""
A script for visualizing the evolution of effective conductance and various network properties for a set of pressure differences 
and corresponding SI spreading probabilities.
This script produces the figures used for Figs. 3 and 5 of the manuscript.

Written by Onerva Korhonen
"""
import params
import visualization

import pickle
import numpy as np
import matplotlib.pylab as plt

fixed_variable = 'pressure' # TODO: select either 'pressure' or 'probability'; use 'probability' to visualize the spreading fitted to empirical data
assert fixed_variable in ['pressure', 'probability'], 'unknown fixed variable, select pressure or probability'
if fixed_variable == 'pressure':
    pooled_data_save_name = params.pooled_optimized_spreading_probability_save_name
elif fixed_variable == 'probability':
    pooled_data_save_name = params.pooled_optimized_spreading_probability_vs_empirical_save_name
prevalence_save_path_base = params.prevalence_save_path.split('.')[0]
percolation_plot_save_path_base = params.percolation_plot_save_path.split('.')[0]
nonfunc_volume_save_path_base = params.nonfunctional_component_size_save_path.split('.')[0]
ninlet_save_path_base = params.ninlet_save_path.split('.')[0]

prevalence_colors = params.prevalence_colors
prevalence_labels = params.prevalence_labels
linestyles = params.percolation_linestyles
labels = params.percolation_labels
std_alpha = params.std_alpha
percolation_outcome_colors = params.percolation_outcome_colors
percolation_outcome_alphas = params.percolation_outcome_alphas
percolation_outcome_labels = params.percolation_outcome_labels
percolation_outcome_ylabels = params.percolation_outcome_ylabels
percolation_outcome_axindex = params.percolation_outcome_axindex
nonfunc_volume_color = params.percolation_nonfunctional_component_size_color
nonfunc_volume_alpha = params.percolation_nonfunctional_component_size_alpha
nonfunc_volume_label = params.percolation_nonfunctional_component_size_label
ninlet_colors = [params.percolation_ninlet_color, params.percolation_noutlet_color]
ninlet_labels = [params.percolation_ninlet_label, params.percolation_noutlet_label]
ninlet_alphas = [params.percolation_ninlet_alpha, params.percolation_noutlet_alpha]

pressure_differences = [1.26E6, 1.28E6, 1.29E6] # TODO: add the desired pressure differences
spreading_probabilities = [0.015, 0.02, 0.05] # TODO: alternatively, add the desired spreading probabiltiies

def rmse(x1, x2): # helper function for calculating RMSE
    max_dim = max(len(x1), len(x2))
    if len(x1) < max_dim:
        x1_padded = np.concatenate((x1, np.ones(max_dim - len(x1)) * x1[-1]), axis=0)
        x2_padded = x2
        x1_cut = x1
        x2_cut = x2[:len(x1)]
        x1_resampled = x1
        x2_resampled = np.interp(np.arange(len(x1)), np.arange(max_dim), x2)
    elif len(x2) < max_dim:
        x2_padded = np.concatenate((x2, np.ones(max_dim - len(x2)) * x2[-1]), axis=0)
        x1_padded = x1
        x1_cut = x1[:len(x2)]
        x2_cut = x2
        x1_resampled = np.interp(np.arange(len(x2)), np.arange(max_dim), x1)
        x2_resampled = x2
    rmse_cut = np.sqrt(np.sum((x1_cut - x2_cut)**2) / len(x1_cut))
    rmse_padded = np.sqrt(np.sum((x1_padded - x2_padded)**2) / len(x2_padded))
    rmse_resampled = np.sqrt(np.sum((x1_resampled - x2_resampled)**2) / len(x2_resampled))
    rmse_resampled_norm = rmse_resampled / np.mean(x1)
    return rmse_cut, rmse_padded, rmse_resampled, rmse_resampled_norm

if __name__=='__main__':

    data_save_folders = [base.rsplit('/', 1)[0] for base in params.optimized_vc_plot_data_save_path_bases]
    for data_save_folder, ls, label in zip(data_save_folders, linestyles, labels):
        file = data_save_folder + '/' + pooled_data_save_name
        with open(file, 'rb') as f:
            data = pickle.load(f)
            f.close()

        saved_pressure_differences = data['pressure_differences']
        saved_spreading_probabilities = data['optimized_spreading_probabilities']

        mins = [] # TODO: remove
        maxs = []

        if fixed_variable == 'pressure':
            x_values = pressure_differences
            search_data = saved_pressure_differences
        elif fixed_variable == 'probability':
            x_values = spreading_probabilities
            search_data = saved_spreading_probabilities

        for x_value in x_values:
            index = np.where(np.isclose(search_data, x_value))[0][0]
            spreading_probability = saved_spreading_probabilities[index]
            pressure_difference = saved_pressure_differences[index]
             
            # 1) Physiological spreading

            # i) prevalence

            if fixed_variable == 'pressure':
                av_phys_prevalence = data['average_physiological_prevalences'][index]
                std_phys_prevalence = data['std_physiological_prevalences'][index]
                av_phys_prevalence_spontaneous = data['average_physiological_prevalences_due_to_spontaneous_embolism'][index]
                std_phys_prevalence_spontaneous = data['std_physiological_prevalences_due_to_spontaneous_embolism'][index]
                av_phys_prevalence_spreading = data['average_physiological_prevalences_due_to_spreading'][index]
                std_phys_prevalence_spreading = data['std_physiological_prevalences_due_to_spreading'][index]
                av_prevalences = [av_phys_prevalence, av_phys_prevalence_spontaneous, av_phys_prevalence_spreading]
                std_prevalences = [std_phys_prevalence, std_phys_prevalence_spontaneous, std_phys_prevalence_spreading]

                prevalence_fig = plt.figure()
                prevalence_ax = prevalence_fig.add_subplot(111)

                for av_prevalence, std_prevalence, prevalence_color, prevalence_label in zip(av_prevalences, std_prevalences, prevalence_colors, prevalence_labels):
                    prevalence_ax.plot(av_prevalence, ls=ls, color=prevalence_color, label=label + ', ' + prevalence_label)
                    prevalence_ax.fill_between(np.arange(len(av_prevalence)), av_prevalence - std_prevalence, av_prevalence + std_prevalence, color=prevalence_color, alpha=std_alpha)

                plt.ylim((params.prevalence_ylims[0], params.prevalence_ylims[1]))
                prevalence_ax.set_xlabel('Time')
                prevalence_ax.set_ylabel('Prevalence (fraction of embolised conduits)')
                prevalence_ax.legend()
                prevalence_save_path = prevalence_save_path_base + '_phys_' + str(pressure_difference).replace('.','_') + '.pdf'
                plt.savefig(prevalence_save_path, format='pdf', bbox_inches='tight')

                # ii) effective conductance, lcc size, functional lcc size
                av_phys_effective_conductances = data['average_physiological_full_effective_conductances'][index]
                std_phys_effective_conductances = data['std_physiological_full_effective_conductances'][index]
                av_phys_lcc_size = data['average_physiological_lcc_size'][index]
                std_phys_lcc_size = data['std_physiological_lcc_size'][index]
                av_phys_func_lcc_size = data['average_physiological_functional_lcc_size'][index]
                std_phys_func_lcc_size = data['std_physiological_functional_lcc_size'][index]

                percolation_fig = plt.figure()
                eff_conductance_ax = percolation_fig.add_subplot(111)
                lcc_size_ax = eff_conductance_ax.twinx()
                axes = [eff_conductance_ax, lcc_size_ax]

                av_percolation_outcomes = np.concatenate((np.expand_dims(av_phys_effective_conductances, axis=0),
                                                          np.expand_dims(av_phys_lcc_size, axis=0), 
                                                          np.expand_dims(av_phys_func_lcc_size, axis=0)), axis=0)
                std_percolation_outcomes = np.concatenate((np.expand_dims(std_phys_effective_conductances, axis=0),
                                                           np.expand_dims(std_phys_lcc_size, axis=0),
                                                           np.expand_dims(std_phys_func_lcc_size, axis=0)), axis=0)

                for av_percolation_outcome, std_percolation_outcome, color, percolation_label, percolation_alpha, axindex in zip(av_percolation_outcomes, std_percolation_outcomes, percolation_outcome_colors, percolation_outcome_labels, percolation_outcome_alphas, percolation_outcome_axindex):
                    axes[axindex].plot(av_percolation_outcome, ls=ls, color=color, label=label + ', ' + percolation_label, alpha=percolation_alpha)
                    axes[axindex].fill_between(np.arange(len(av_percolation_outcome)), av_percolation_outcome - std_percolation_outcome, av_percolation_outcome + std_percolation_outcome, color=color, alpha=std_alpha * percolation_alpha)
                    mins.append(np.amin(av_percolation_outcome - std_percolation_outcome)) # TODO: remove
                    maxs.append(np.amax(av_percolation_outcome + std_percolation_outcome))
            
                eff_conductance_ax.set_ylim((params.keff_ylims[0], params.keff_ylims[1]))
                eff_conductance_ax.set_xlabel('Time')
                eff_conductance_ax.set_ylabel(percolation_outcome_ylabels[0])
                lcc_size_ax.set_ylabel(percolation_outcome_ylabels[1])
                lcc_size_ax.set_ylim((params.lcc_ylims[0], params.lcc_ylims[1]))
                eff_conductance_ax.legend()

                percolation_plot_save_path = percolation_plot_save_path_base + '_phys_' + str(pressure_difference).replace('.','_') + '.pdf'
                plt.savefig(percolation_plot_save_path, format='pdf', bbox_inches='tight')

                # iii) nonfunctional component volume (dead water)
                av_phys_nonfunc_volume = data['average_physiological_nonfunctional_component_volume'][index]
                std_phys_nonfunc_volume = data['std_physiological_nonfunctional_component_volume'][index]

                nonfunc_volume_fig = plt.figure()
                nonfunc_volume_ax = nonfunc_volume_fig.add_subplot(111)

                nonfunc_volume_ax.plot(av_phys_nonfunc_volume, ls=ls, color=nonfunc_volume_color, alpha=nonfunc_volume_alpha, label=label + ', ' + nonfunc_volume_label)
                nonfunc_volume_ax.fill_between(np.arange(len(av_phys_nonfunc_volume)), av_phys_nonfunc_volume - std_phys_nonfunc_volume, av_phys_nonfunc_volume + std_phys_nonfunc_volume, color=nonfunc_volume_color, alpha=std_alpha * nonfunc_volume_alpha)

                plt.ylim((params.nonfunc_volume_ylims[0], params.nonfunc_volume_ylims[1]))
                nonfunc_volume_ax.set_xlabel('Time')
                nonfunc_volume_ax.set_ylabel('Nonfunctional component volume (m^3)')
                nonfunc_volume_ax.legend()

                nonfunc_volume_save_path = nonfunc_volume_save_path_base + '_phys_' + str(pressure_difference).replace('.','_') + '.pdf'
                plt.savefig(nonfunc_volume_save_path, format='pdf', bbox_inches='tight')

                # iv) number of inlets and outlets
                av_phys_ninlets = data['average_physiological_n_inlets'][index]
                std_phys_ninlets = data['std_physiological_n_inlets'][index]
                av_phys_noutlets = data['average_physiological_n_outlets'][index]
                std_phys_noutlets = data['std_physiological_n_outlets'][index]

                ninlet_fig = plt.figure()
                ninlet_ax = ninlet_fig.add_subplot(111)

                av_ns = [av_phys_ninlets, av_phys_noutlets]
                std_ns = [std_phys_ninlets, std_phys_noutlets]

                for av_n, std_n, ncolor, nlabel, nalpha in zip(av_ns, std_ns, ninlet_colors, ninlet_labels, ninlet_alphas):
                    ninlet_ax.plot(av_n, ls=ls, color=ncolor, label=label + ', ' + nlabel, alpha=nalpha)
                    ninlet_ax.fill_between(np.arange(len(av_n)), av_n - std_n, av_n + std_n, color=ncolor, alpha=std_alpha * nalpha)

                plt.ylim((params.ninlet_ylims[0], params.ninlet_ylims[1]))
                ninlet_ax.set_xlabel('Time')
                ninlet_ax.legend()

                ninlet_save_path = ninlet_save_path_base + '_phys_' + str(pressure_difference).replace('.','_') + '.pdf'
                plt.savefig(ninlet_save_path, format='pdf', bbox_inches='tight')

            # 2) SI spreading

            # i) prevalence
            av_stoch_prevalence = data['average_stochastic_prevalences'][index]
            std_stoch_prevalence = data['std_stochastic_prevalences'][index]
            av_stoch_prevalence_spontaneous = data['average_stochastic_prevalences_due_to_spontaneous_embolism'][index]
            std_stoch_prevalence_spontaneous = data['std_stochastic_prevalences_due_to_spontaneous_embolism'][index]
            av_stoch_prevalence_spreading = data['average_stochastic_prevalences_due_to_spreading'][index]
            std_stoch_prevalence_spreading = data['std_stochastic_prevalences_due_to_spreading'][index]
            av_prevalences = [av_stoch_prevalence, av_stoch_prevalence_spontaneous, av_stoch_prevalence_spreading]
            std_prevalences = [std_stoch_prevalence, std_stoch_prevalence_spontaneous, std_stoch_prevalence_spreading]
 
            prevalence_fig = plt.figure()
            prevalence_ax = prevalence_fig.add_subplot(111)
 
            for av_prevalence, std_prevalence, prevalence_color, prevalence_label in zip(av_prevalences, std_prevalences, prevalence_colors, prevalence_labels):
                prevalence_ax.plot(av_prevalence, ls=ls, color=prevalence_color, label=label + ', ' + prevalence_label)
                prevalence_ax.fill_between(np.arange(len(av_prevalence)), av_prevalence - std_prevalence, av_prevalence + std_prevalence, color=prevalence_color, alpha=std_alpha)
 
            plt.ylim((params.prevalence_ylims[0], params.prevalence_ylims[1]))
            prevalence_ax.set_xlabel('Time')
            prevalence_ax.set_ylabel('Prevalence (fraction of embolised conduits)')
            prevalence_ax.legend()
            if fixed_variable == 'pressure':
                prevalence_save_path = prevalence_save_path_base + '_stoch_' + str(pressure_difference).replace('.','_') + '.pdf'
            elif fixed_variable == 'probability':
                prevalence_save_path = prevalence_save_path_base + '_vs_empirical_' + str(x_value).replace('.','_') + '.pdf'
            plt.savefig(prevalence_save_path, format='pdf', bbox_inches='tight')
 
            # ii) effective conductance, lcc size, functional lcc size
            av_stoch_effective_conductances = data['average_stochastic_full_effective_conductances'][index]
            std_stoch_effective_conductances = data['std_stochastic_full_effective_conductances'][index]
            av_stoch_lcc_size = data['average_stochastic_lcc_size'][index]
            std_stoch_lcc_size = data['std_stochastic_lcc_size'][index]
            av_stoch_func_lcc_size = data['average_stochastic_functional_lcc_size'][index]
            std_stoch_func_lcc_size = data['std_stochastic_functional_lcc_size'][index]
 
            percolation_fig = plt.figure()
            eff_conductance_ax = percolation_fig.add_subplot(111)
            lcc_size_ax = eff_conductance_ax.twinx()  
            axes = [eff_conductance_ax, lcc_size_ax]   

            av_percolation_outcomes = np.concatenate((np.expand_dims(av_stoch_effective_conductances, axis=0),
                                                      np.expand_dims(av_stoch_lcc_size, axis=0),
                                                      np.expand_dims(av_stoch_func_lcc_size, axis=0)), axis=0)
            std_percolation_outcomes = np.concatenate((np.expand_dims(std_stoch_effective_conductances, axis=0),
                                                       np.expand_dims(std_stoch_lcc_size, axis=0),
                                                       np.expand_dims(std_stoch_func_lcc_size, axis=0)), axis=0)
 
            for av_percolation_outcome, std_percolation_outcome, color, percolation_label, percolation_alpha, axindex in zip(av_percolation_outcomes, std_percolation_outcomes, percolation_outcome_colors, percolation_outcome_labels, percolation_outcome_alphas, percolation_outcome_axindex):
                axes[axindex].plot(av_percolation_outcome, ls=ls, color=color, label=label + ', ' + percolation_label, alpha=percolation_alpha)
                axes[axindex].fill_between(np.arange(len(av_percolation_outcome)), av_percolation_outcome - std_percolation_outcome, av_percolation_outcome + std_percolation_outcome, color=color, alpha=std_alpha * percolation_alpha)
                mins.append(np.amin(av_percolation_outcome - std_percolation_outcome)) # TODO: remove
                maxs.append(np.amax(av_percolation_outcome + std_percolation_outcome))

            eff_conductance_ax.set_ylim((params.keff_ylims[0], params.keff_ylims[1]))
            eff_conductance_ax.set_xlabel('Time')
            eff_conductance_ax.set_ylabel(percolation_outcome_ylabels[0])
            lcc_size_ax.set_ylim((params.lcc_ylims[0], params.lcc_ylims[1]))
            lcc_size_ax.set_ylabel(percolation_outcome_ylabels[1])
            eff_conductance_ax.legend()
 
            if fixed_variable == 'pressure':
                percolation_plot_save_path = percolation_plot_save_path_base + '_stoch_' + str(pressure_difference).replace('.','_') + '.pdf'
            elif fixed_variable == 'probability':
                percolation_plot_save_path = percolation_plot_save_path_base + '_vs_empirical_' + str(x_value).replace('.','_') + '.pdf'
            plt.savefig(percolation_plot_save_path, format='pdf', bbox_inches='tight')
 
            # iii) nonfunctional component volume (dead water)
            av_stoch_nonfunc_volume = data['average_stochastic_nonfunctional_component_volume'][index]
            std_stoch_nonfunc_volume = data['std_stochastic_nonfunctional_component_volume'][index]
 
            nonfunc_volume_fig = plt.figure()
            nonfunc_volume_ax = nonfunc_volume_fig.add_subplot(111)
 
            nonfunc_volume_ax.plot(av_stoch_nonfunc_volume, ls=ls, color=nonfunc_volume_color, alpha=nonfunc_volume_alpha, label=label + ', ' + nonfunc_volume_label)
            nonfunc_volume_ax.fill_between(np.arange(len(av_stoch_nonfunc_volume)), av_stoch_nonfunc_volume - std_stoch_nonfunc_volume, av_stoch_nonfunc_volume + std_stoch_nonfunc_volume, color=nonfunc_volume_color, alpha=std_alpha * nonfunc_volume_alpha)

            plt.ylim((params.nonfunc_volume_ylims[0], params.nonfunc_volume_ylims[1]))
            nonfunc_volume_ax.set_xlabel('Time')
            nonfunc_volume_ax.set_ylabel('Nonfunctional component volume (m^3)')
            nonfunc_volume_ax.legend()
 
            if fixed_variable == 'pressure':
                nonfunc_volume_save_path = nonfunc_volume_save_path_base + '_stoch_' + str(pressure_difference).replace('.','_') + '.pdf'
            elif fixed_variable == 'probability':
                nonfunc_volume_save_path = nonfunc_volume_save_path_base + '_vs_empirical_' + str(x_value).replace('.','_') + '.pdf'
            plt.savefig(nonfunc_volume_save_path, format='pdf', bbox_inches='tight')

            # iv) number of inlets and outlets
            av_stoch_ninlets = data['average_stochastic_n_inlets'][index]
            std_stoch_ninlets = data['std_stochastic_n_inlets'][index]
            av_stoch_noutlets = data['average_stochastic_n_outlets'][index]
            std_stoch_noutlets = data['std_stochastic_n_outlets'][index]
 
            ninlet_fig = plt.figure()
            ninlet_ax = ninlet_fig.add_subplot(111)
 
            av_ns = [av_stoch_ninlets, av_stoch_noutlets]
            std_ns = [std_stoch_ninlets, std_stoch_noutlets]
 
            for av_n, std_n, ncolor, nlabel, nalpha in zip(av_ns, std_ns, ninlet_colors, ninlet_labels, ninlet_alphas):
                ninlet_ax.plot(av_n, ls=ls, color=ncolor, label=label + ', ' + nlabel, alpha=nalpha)
                ninlet_ax.fill_between(np.arange(len(av_n)), av_n - std_n, av_n + std_n, color=ncolor, alpha=std_alpha * nalpha)
  
            plt.ylim((params.ninlet_ylims[0], params.ninlet_ylims[1]))
            ninlet_ax.set_xlabel('Time')
            ninlet_ax.legend()
 
            if fixed_variable == 'pressure':
                ninlet_save_path = ninlet_save_path_base + '_stoch_' + str(pressure_difference).replace('.','_') + '.pdf'
            elif fixed_variable == 'probability':
                ninlet_save_path = ninlet_save_path_base + '_vs_empirical_' + str(x_value).replace('.','_') + '.pdf'
            plt.savefig(ninlet_save_path, format='pdf', bbox_inches='tight')
 
            # 3) Calculating RMSE
            if fixed_variable == 'pressure':
                rmse_cut_prevalence, rmse_padded_prevalence, rmse_resampled_prevalence, rmse_resampled_norm_prevalence = rmse(av_phys_prevalence, av_stoch_prevalence)
                rmse_cut_eff_conductance, rmse_padded_eff_conductance, rmse_resampled_eff_conductance, rmse_resampled_norm_eff_conductance = rmse(av_phys_effective_conductances, av_stoch_effective_conductances)
                rmse_cut_lcc_size, rmse_padded_lcc_size, rmse_resampled_lcc_size, rmse_resampled_norm_lcc_size = rmse(av_phys_lcc_size, av_stoch_lcc_size)
                rmse_cut_func_lcc_size, rmse_padded_func_lcc_size, rmse_resampled_func_lcc_size, rmse_resampled_norm_func_lcc_size = rmse(av_phys_func_lcc_size, av_stoch_func_lcc_size)
                rmse_cut_nonfunc_volume, rmse_padded_nonfunc_volume, rmse_resampled_nonfunc_volume, rmse_resampled_norm_nonfunc_volume  = rmse(av_phys_nonfunc_volume, av_stoch_nonfunc_volume)
                rmse_cut_ninlets, rmse_padded_ninlets, rmse_resampled_ninlets, rmse_resampled_norm_ninlets = rmse(av_phys_ninlets, av_stoch_ninlets)
                rmse_cut_noutlets, rmse_padded_noutlets, rmse_resampled_noutlets, rmse_resampled_norm_noutlets = rmse(av_phys_noutlets, av_stoch_noutlets)
 
                print(f'At pressure difference {pressure_difference}, optimal SI spreading probability is {spreading_probability}')
                print(f'Prevalence: cut RMSE: {rmse_cut_prevalence}, padded RMSE: {rmse_padded_prevalence}, resampled RMSE: {rmse_resampled_prevalence}, {rmse_resampled_norm_prevalence}')
                print(f'Effective conductance: cut RMSE: {rmse_cut_eff_conductance}, padded RMSE: {rmse_padded_eff_conductance}, resampled RMSE: {rmse_resampled_eff_conductance}, {rmse_resampled_norm_eff_conductance}')
                print(f'LCC size: cut RMSE: {rmse_cut_lcc_size}, padded RMSE: {rmse_padded_lcc_size}, resampled RMSE: {rmse_resampled_lcc_size}, {rmse_resampled_norm_lcc_size}')
                print(f'Functional LCC size: cut RMSE: {rmse_cut_func_lcc_size}, padded RMSE: {rmse_padded_func_lcc_size}, resampled RMSE: {rmse_resampled_func_lcc_size}, {rmse_resampled_norm_func_lcc_size}')
                print(f'Nonfunctional component volume: cut RMSE: {rmse_cut_nonfunc_volume}, padded RMSE: {rmse_padded_nonfunc_volume}, resampled RMSE: {rmse_resampled_nonfunc_volume}, {rmse_resampled_norm_nonfunc_volume}')
                print(f'N inlets: cut RMSE: {rmse_cut_ninlets}, padded RMSE: {rmse_padded_ninlets}, resampled RMSE: {rmse_resampled_ninlets}, {rmse_resampled_norm_ninlets}')
                print(f'N outlets: cut RMSE: {rmse_cut_noutlets}, padded RMSE: {rmse_padded_noutlets}, resampled RMSE: {rmse_resampled_noutlets}, {rmse_resampled_norm_noutlets}')





            

