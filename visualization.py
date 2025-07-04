#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 15:31:29 2023

@author: onerva

Functions for visualization by Petri Kiuru & Onerva Korhonen
"""
import matplotlib.pylab as plt
import numpy as np
import openpnm as op
import pickle
import os

import mrad_params as params

def visualize_network_with_openpnm(net, cylinder=False, Lce=params.Lce, pore_coordinate_key='pore.coords'):
    """
    Visualizes a conduit network using the OpenPNM visualization tools.
    
    Parameters:
    -----------
    net : openpnm.Network() object
        pores correspond to conduit elements, throats to connections between elements
    cylinder : bln, optional
        visualize the network in cylinder coordinate system instead of the Cartesian one
    Lce : float, optional
        lenght of the conduit element
    pore_coordinate_key : str, optional
        key, under which the pore coordinate information is stored in net

    Returns
    -------
    None.

    """
    ts = net.find_neighbor_throats(pores=net.pores('all'))
    
    ax = op.visualization.plot_coordinates(network=net, pores=net.pores('all'), c='r')
    ax = op.visualization.plot_connections(network=net, throats=ts, ax=ax, c='b')
    if cylinder:
        ax._axes.set_zlim([-1*Lce, 1.1*np.max(net[pore_coordinate_key][:, 2])])
    
def visualize_pores(net, pore_coordinate_key='pore.coords'):
    """
    Visualizes the location of the nodes (pores) of an OpenPNM network as a scatter plot.

    Parameters
    ----------
    net : openpnm.Network()
        pores correspond to conduit elements, throats to connections between them
    pore_coordinate_key : str, optional
        key, under which the pore coordinate information is saved
        
    Returns
    -------
    None.
    """
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(net[pore_coordinate_key][:, 0], net[pore_coordinate_key][:, 1], net[pore_coordinate_key][:, 2],
                      c='r', s=5e5*net['pore.diameter'])
    
def make_colored_pore_scatter(net, pore_values, title='', cmap=plt.cm.jet):
    """
    Plots a scatter where points correspond to network pores, their location is determined by the
    network geometry, and they are colored by some pore property (e.g. pressure or concentration at
    pores).
 
    
    Parameters
    ----------
    net : openpnm.Network()
        pores correspond to conduit elements, throats to connections between them
    pore_values : iterable of floats
        values that define pore colors: pore colors are set so that the minimum of the color map corresponds
        to the minimum of pore_values and maximum of the color map corresponds to the maximum of pore_values
    title : str, optional
        figure title
    cmap : matplotlib colormap, optional (default: jet)
    
    Returns
    -------
    None.
    """
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    p = ax.scatter(1000 * net['pore.coords'][:, 0],
                   1000 * net['pore.coords'][:, 1],
                   1000 * net['pore.coords'][:, 2],
                   c=pore_values, s = 1e11 * net['pore.diameter']**2,
                   cmap=plt.cm.jet)
    fig.colorbar(p, ax=ax)
    plt.title(title)
    
def plot_percolation_curve(total_n_nodes, values, colors, labels, alphas, axindex=[], y_labels=[], save_path='', xs=[], param_linestyles=[], param_labels=[]):
    """
    Plots the percolation analysis outcomes (e.g. largest connected component size and effective conductance) as a function
    of the fraction of nodes removed; outcomes from several sets of analysis parameters can be plotted in a single image

    Parameters
    ----------
    total_n_nodes : list of int
        list number of nodes in the network subject to percolation analysis (used for constructing x axis); one element per parameter set
    values : list of np.arrays
        one array per parameter set;
        each row corresponds to a set of percolation outcome values to be plotted against the fraction of
        removed pores. if values.shape[1] < total_n_nodes + 1, the assumption is that only total_n_nodes + 1 - len(effective_conductance)
        nodes have been removed
    colors : list of strs
        colors used for plotting. len(colors) should equal to values.shape[0]
    labels : list of strs
        labels of the curves plotted. len(labels) should equal to values.shape[0]
    alphas : iterable of floats
        transparency values of the plotted lines. len(alphas) should equal to values.shape[0]
    axindex : iterable of ints, optional
        index of the y axis (first or secondary), on which the corresponding row of values should be drawn. len(axindex) should equal to values.shape[0]
    y_labels : list of strs, optional
        labels of the first and secondary y axis. only used if the secondary y axis is used, in which case len(y_labels) should be 2
    save_path : str, optional
        if len(save_path) > 0, the figure will be saved as .pdf to the given path
    xs : list of np.array, optional
        one array per parameter set;
        the x axis, against which to plot the percolation outcomes; if x is not given, the x axis is constructed as linspace
        assuming that one node/edge is removed at each step
    param_linestyles : list of strs, optional
        linestyles corresponding to the outcomes from different parameter sets to be visualized    
    param_labels : list of strs, optional
        labels corresponding to the outcomes from different parameter sets to be visualized
        
    Returns
    -------
    None.

    """
    fig = plt.figure()
    ax = fig.add_subplot(111)
    secondary_ax = len(np.unique(axindex)) > 1
    if secondary_ax: # a secondary y axis will be used 
        ax2 = ax.twinx()
        axis = [ax, ax2]
    
    for total_n_nodes, value_array, x, param_ls, param_label in zip(total_n_nodes, values, xs, param_linestyles, param_labels):
        n_percolation_steps = value_array.shape[1]
        assert n_percolation_steps > 1, 'only the values calculated for the full network given for plotting percolation curves'
        if len(x) > 0:
            assert len(x) == n_percolation_steps, "length of the given x axis does not match the number of percolation outcomes"
        else:
            if n_percolation_steps < total_n_nodes - 1:
                print('Warning: number of effective conductance values from percolation analysis does not match the number of nodes')
                x = np.linspace(0, 100 * (n_percolation_steps / total_n_nodes), n_percolation_steps)
            else:
                x = np.linspace(0, 100, total_n_nodes)
    
        if secondary_ax: #secondary y axis will be used 
            ax2 = ax.twinx()
            axis = [ax, ax2]
            for value, color, label, alpha, axind in zip(value_array, colors, labels, alphas, axindex):
                axis[axind].plot(x, value, color=color, label=label + ', ' + param_label, alpha=alpha, ls=param_ls)
        else:    
            for value, color, label, alpha in zip(value_array, colors, labels, alphas):
                ax.plot(x, value, color=color, label=label + ', ' + param_label, alpha=alpha, ls=param_ls)
            
    
    if secondary_ax:
        ax.set_ylabel(y_labels[0])
        ax2.set_ylabel(y_labels[1])
        
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2, loc=0)
    else:
        ax.set_ylabel('Percolation outcome')
        ax.legend()
    ax.set_xlabel('Fraction of edges/nodes removed')

    if len(save_path) > 0:
        plt.savefig(save_path, format='pdf',bbox_inches='tight')
        
def plot_vulnerability_curve(vc, color, alpha, vc_type='physiological', save_path=''):
    """
    Plots the vulnerability curve (percentage of effective conductance lost as a function of 
    pressure difference / SI spreading parameter).
    
    Parameters
    ----------
    vc : tuple
        vulnerability curve: vc[0] contains the x values and vc[1] the related conductance loss values. output of 
        percolation.construct_vulnerability_curve().
    color : str
        color of the curve
    alpha : float
        transparency of the curve
    vc_type : str
        'physiological' or 'stochastic' (default 'phsyiological'). used to define x axis label
    save_path : str
        path to which save the plt. The default is '', in which case the plot is not saved but only shown.

    Returns
    -------
    None.
    """
    assert vc_type in ['physiological', 'stochastic'], 'unknown vc type, select physiological or stochastic'
    if vc_type == 'physiological':
        x = -1 * vc[0][::-1] # inverting the x axis following the convention of presenting negative pressure differences
        y = vc[1][::-1]
    else:
        x = vc[0]
        y = vc[1]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(x, y, color=color, alpha=alpha)
    if vc_type == 'physiological':
        x_label = 'Pressure difference'
    else:
        x_label = 'Spreading probability'
    ax.set_xlabel(x_label)
    ax.set_ylabel('PLC (%)')
    if len(save_path) > 0:
        plt.savefig(save_path, format='pdf', bbox_inches='tight')
        
def plot_optimized_vulnerability_curve(data_save_folders, physiological_color, stochastic_color, physiological_alpha, stochastic_alpha, line_styles, labels, p_50_color, p_50_alpha, p_50_line_style, save_path, pooled_data=False, pooled_data_save_name='', std_alpha=0.5, prevalence_plot_save_path_base='', prevalence_linestyles=[], plc_in_file=False):
    """
    Reads from files the optimized SI spreading probabilities and related effective conductance values for a set of pressure difference values, calculates the percentage of
    conductance lost (PLC) values (out of effective conductance at pressure difference 0) and plots the vulnerability curves and prevalence plots for all pressure differences.

    Parameters
    ----------
    data_save_folders : list of str
        folders in which the data is saved; the folders MUST NOT contain other files
    physiological_color : str
        color of the vc curve corresponding to physiological spreading
    stochastic_color : str
        color of the vc curve corresponding to stochastic spreading
    physiological_ls : str
        line style of the vc curve corresponding to physiological spreading
    stochastic_ls : str
        line style of the vc curve corresponding to stochastic spreading
    physiological_alpha : float
        transparency of the vc curve corresponding to physiological spreading
    stochastic_alpha : float
        transparency of the vc curve corresponidng to stochastic spreading
    line_styles : list of strs
        line styles of the vc curve; one style per data_save_folder
    labels : list of strs
        labels of the vc curbes; one label per data_save_folder
    p_50_color : str
        color of the lines marking the P_12, P_50, and P_88 values
    p_50_alpha : float
        transparency of the lines marking the P_12, P_50, and P_88 values
    p_50_line_style : str
        line style of the lines marking the P_12, P_50, and P_88 values
    save_path : str
        path to which to save the plot
    pooled_data : bln, optional
        has the data already been pooled, i.e., can all data be read from a single file per data_save_folder? (default: False)
    pooled_data_save_name : str, optional
        the name (NOTE: not path) of the pooled data file (default: '')
    std_alpha : float, optional
        alpha used to color the mean +/- std in the prevalence plots (default: 0.5)
    prevalence_linestyels : list of strs, optional
        the linestyles used for plotting three types of prevalence: total, due to spontaneous embolism, and due to spreading
    prevalence_plot_save_path_base : str, optional
        the base path to which to save the prevalence plot
    plc_in_file : bln, optional
        if True, the pooled data file already contains PLC values instead of effective conductances and PLC should not be precalculated (this is the case if spreading probability has been
        optimized against an empirical vulnerability curve) (default: False)
    Returns
    -------
    None
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax2 = ax.twinx()
    
    if pooled_data:
        tot_prevalence_ls = prevalence_linestyles[0]
        spontaneous_prevalence_ls = prevalence_linestyles[1]
        spreading_prevalence_ls = prevalence_linestyles[2]
    
    for data_save_folder, line_style, label in zip(data_save_folders, line_styles, labels): 
        if pooled_data:
            file = data_save_folder + '/' + pooled_data_save_name
            with open(file, 'rb') as f:
                data = pickle.load(f)
                f.close()
            pressure_diffs = data['pressure_differences']
            optimized_spreading_probabilities = data['optimized_spreading_probabilities']
            if plc_in_file:
                physiological_plc = data['physiological_PLCs']
                stochastic_plc = data['stochastic_PLCs']
            else:
                physiological_effective_conductances = data['physiological_effective_conductances']
                stochastic_effective_conductances = data['stochastic_effective_conductances']
            
            if not plc_in_file: # if physiological data is an empirical VC, prevalence information is not available
                average_phys_prevalences = data['average_physiological_prevalences']
                std_phys_prevalences = data['std_physiological_prevalences']
                average_phys_prevalences_spontaneous = data['average_physiological_prevalences_due_to_spontaneous_embolism']
                std_phys_prevalences_spontaneous = data['std_physiological_prevalences_due_to_spontaneous_embolism']
                average_phys_prevalences_spreading = data['average_physiological_prevalences_due_to_spreading']
                std_phys_prevalences_spreading = data['std_physiological_prevalences_due_to_spreading']
                average_stoch_prevalences = data['average_stochastic_prevalences']
                std_stoch_prevalences = data['std_stochastic_prevalences']
                average_stoch_prevalences_spontaneous = data['average_stochastic_prevalences_due_to_spontaneous_embolism']
                std_stoch_prevalences_spontaneous = data['std_stochastic_prevalences_due_to_spontaneous_embolism']
                average_stoch_prevalences_spreading = data['average_stochastic_prevalences_due_to_spreading']
                std_stoch_prevalences_spreading = data['std_stochastic_prevalences_due_to_spreading']
        else:
            data_files = [os.path.join(data_save_folder, file) for file in os.listdir(data_save_folder) if os.path.isfile(os.path.join(data_save_folder, file))]
            pressure_diffs = np.zeros(len(data_files))
            physiological_effective_conductances = np.zeros(len(data_files))
            stochastic_effective_conductances = np.zeros(len(data_files))
        
            optimized_spreading_probabilities = np.zeros(len(data_files))
        
            for i, file in enumerate(data_files):
                with open(file, 'rb') as f:
                    data = pickle.load(f)
                    f.close()
                pressure_diffs[i] = data['pressure_difference']
                physiological_effective_conductances[i] = data['physiological_effective_conductance']
                stochastic_effective_conductances[i] = data['stochastic_effective_conductance']
                if not 'optimized_spreading_probability' in data.keys():
                    optimized_spreading_probabilities[i] = data['optimal_spreading_probability'] # backward compatibility case, should be removed
                else:
                    optimized_spreading_probabilities[i] = data['optimized_spreading_probability']
                    
        if not plc_in_file:
            indices = np.argsort(pressure_diffs)
            pressure_diffs = pressure_diffs[indices]
            physiological_effective_conductances = physiological_effective_conductances[indices]
            stochastic_effective_conductances = stochastic_effective_conductances[indices]
            optimized_spreading_probabilities = optimized_spreading_probabilities[indices]
            assert pressure_diffs[0] == 0, 'effective conductance at pressure difference 0 required for calculating percentage of conductance lost; values < 0 are not allowed!'
            physiological_plc = 100 * (1 - physiological_effective_conductances / physiological_effective_conductances[0])
            stochastic_plc = 100 * (1 - stochastic_effective_conductances / stochastic_effective_conductances[0]) # normalized by the effective conductance at the spreading probability optimized for pressure difference 0
            
        pressure_diffs = -1 * pressure_diffs[::-1] # inverting the x axis to follow the convention of presenting negative pressure differences
        physiological_plc = physiological_plc[::-1]
        stochastic_plc = stochastic_plc[::-1]
        optimized_spreading_probabilities = optimized_spreading_probabilities[::-1]
        
        p_25 = pressure_diffs[np.argmin(np.abs(physiological_plc - 25))]
        p_50 = pressure_diffs[np.argmin(np.abs(physiological_plc - 50))]
        p_75 = pressure_diffs[np.argmin(np.abs(physiological_plc - 75))]
        p_12 = pressure_diffs[np.argmin(np.abs(physiological_plc - 12))]
        p_88 = pressure_diffs[np.argmin(np.abs(physiological_plc - 88))]
        
        print(f'P_25: {p_25}')
        print(f'P_50: {p_50}')
        print(f'P_75: {p_75}')
        print(f'P_12: {p_12}')
        print(f'P_88: {p_88}')

        ax.plot(pressure_diffs, physiological_plc, color=physiological_color, alpha=physiological_alpha, ls=line_style, label='physiological ' + label)
        ax.plot(pressure_diffs, stochastic_plc, color=stochastic_color, alpha=stochastic_alpha, ls=line_style, label='stochastic ' + label)
        ax2.plot(pressure_diffs, optimized_spreading_probabilities, label='optimized spreading probability ' + label)
        
        ax.plot([p_12, p_12], [0, 1], color=p_50_color, alpha=p_50_alpha, ls=p_50_line_style)
        ax.plot([p_50, p_50], [0, 1], color=p_50_color, alpha=p_50_alpha, ls=p_50_line_style)
        ax.plot([p_88, p_88], [0, 1], color=p_50_color, alpha=p_50_alpha, ls=p_50_line_style)
        
        if pooled_data and not plc_in_file: # prevalence information is included only in the pooled data
            for i, (av_phys_prevalence, std_phys_prevalence, av_phys_prevalence_spontaneous, std_phys_prevalence_spontaneous, av_phys_prevalence_spreading, std_phys_prevalence_spreading, av_stoch_prevalence, std_stoch_prevalence, av_stoch_prevalence_spontaneous, std_stoch_prevalence_spontaneous, av_stoch_prevalence_spreading, std_stoch_prevalence_spreading) in enumerate(zip(average_phys_prevalences, std_phys_prevalences, average_phys_prevalences_spontaneous, std_phys_prevalences_spontaneous, average_phys_prevalences_spreading, std_phys_prevalences_spreading, average_stoch_prevalences, std_stoch_prevalences, average_stoch_prevalences_spontaneous, std_stoch_prevalences_spontaneous, average_stoch_prevalences_spreading, std_stoch_prevalences_spreading)):
                prevalence_fig = plt.figure()
                prevalence_ax = prevalence_fig.add_subplot(111)
                prevalence_ax.plot(av_phys_prevalence, color=physiological_color, alpha=physiological_alpha, ls=tot_prevalence_ls, label='phys total ' + label)
                prevalence_ax.fill_between(np.arange(len(av_phys_prevalence)), av_phys_prevalence - std_phys_prevalence, av_phys_prevalence + std_phys_prevalence, color=physiological_color, alpha=std_alpha)
                prevalence_ax.plot(av_phys_prevalence_spontaneous, color=physiological_color, alpha=physiological_alpha, ls=spontaneous_prevalence_ls, label='phys spontaneous ' + label)
                prevalence_ax.fill_between(np.arange(len(av_phys_prevalence_spontaneous)), av_phys_prevalence_spontaneous - std_phys_prevalence_spontaneous, av_phys_prevalence_spontaneous + std_phys_prevalence_spontaneous, color=physiological_color, alpha=std_alpha)
                prevalence_ax.plot(av_phys_prevalence_spreading, color=physiological_color, alpha=physiological_alpha, ls=spreading_prevalence_ls, label='phys spreading ' + label)
                prevalence_ax.fill_between(np.arange(len(av_phys_prevalence_spreading)), av_phys_prevalence_spreading - std_phys_prevalence_spreading, av_phys_prevalence_spreading + std_phys_prevalence_spreading, color=physiological_color, alpha=std_alpha)
                
                prevalence_ax.plot(av_stoch_prevalence, color=stochastic_color, alpha=stochastic_alpha, ls=tot_prevalence_ls, label='stoch tot ' + label)
                prevalence_ax.fill_between(np.arange(len(av_stoch_prevalence)), av_stoch_prevalence - std_stoch_prevalence, av_stoch_prevalence + std_stoch_prevalence, color=stochastic_color, alpha=std_alpha)
                prevalence_ax.plot(av_stoch_prevalence_spontaneous, color=stochastic_color, alpha=stochastic_alpha, ls=spontaneous_prevalence_ls, label='stoch spontaneous ' + label)
                prevalence_ax.fill_between(np.arange(len(av_stoch_prevalence_spontaneous)), av_stoch_prevalence_spontaneous - std_stoch_prevalence_spontaneous, av_stoch_prevalence_spontaneous + std_stoch_prevalence_spontaneous, color=stochastic_color, alpha=std_alpha)
                prevalence_ax.plot(av_stoch_prevalence_spreading, color=stochastic_color, alpha=stochastic_alpha, ls=spreading_prevalence_ls, label='stoch spreading ' + label)
                prevalence_ax.fill_between(np.arange(len(av_stoch_prevalence_spreading)), av_stoch_prevalence_spreading - std_stoch_prevalence_spreading, av_stoch_prevalence_spreading + std_stoch_prevalence_spreading, color=stochastic_color, alpha=std_alpha)
                
                prevalence_ax.set_xlabel('Time')
                prevalence_ax.set_ylabel('Prevalence (fraction of embolized conduits)')
                prevalence_ax.legend()
                prevalence_plot_save_path = prevalence_plot_save_path_base + '_' + str(pressure_diffs[i]).replace('.', '_') + '.pdf'
                plt.savefig(prevalence_plot_save_path, format='pdf', bbox_inches='tight')
    
    plt.figure(fig)
    ax.set_xlabel('Pressure difference')
    ax.set_ylabel('PLC (%)')
    ax.legend()
    ax2.set_ylabel('Optimized spreading probability')
    plt.savefig(save_path, format='pdf', bbox_inches='tight')

