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
    
def plot_percolation_curve(total_n_nodes, values, colors, labels, alphas, axindex=[], y_labels=[], save_path='', x=[]):
    """
    Plots the percolation analysis outcomes (e.g. largest connected component size and effective conductance) as a function
    of the fraction of nodes removed.

    Parameters
    ----------
    total_n_nodes : int
        number of nodes in the network subject to percolation analysis (used for constructing x axis)
    values : np.array
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
    x : np.array, optional
        the x axis, against which to plot the percolation outcomes; if x is not given, the x axis is constructed as linspace
        assuming that one node/edge is removed at each step
        

    Returns
    -------
    None.

    """
    n_percolation_steps = values.shape[1]
    assert n_percolation_steps > 1, 'only the values calculated for the full network given for plotting percolation curves'
    if len(x) > 0:
        assert len(x) == n_percolation_steps, "length of the given x axis does not match the number of percolation outcomes"
    else:
        if n_percolation_steps < total_n_nodes - 1:
            print('Warning: number of effective conductance values from percolation analysis does not match the number of nodes')
            x = np.linspace(0, 100 * (n_percolation_steps / total_n_nodes), n_percolation_steps)
        else:
            x = np.linspace(0, 100, total_n_nodes)
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    if len(np.unique(axindex)) > 1: #secondary y axis will be used 
        ax2 = ax.twinx()
        axis = [ax, ax2]
        for value, color, label, alpha, axind in zip(values, colors, labels, alphas, axindex):
            axis[axind].plot(x, value, color=color, label=label, alpha=alpha)
        ax.set_ylabel(y_labels[0])
        ax2.set_ylabel(y_labels[1])
        
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2, loc=0)
    else:    
        for value, color, label, alpha in zip(values, colors, labels, alphas):
            ax.plot(x, value, color=color, label=label, alpha=alpha)
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
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(vc[0], vc[1], color=color, alpha=alpha)
    if vc_type == 'physiological':
        x_label = 'Pressure difference'
    else:
        x_label = 'Spreading probability'
    ax.set_xlabel(x_label)
    if len(save_path) > 0:
        plt.savefig(save_path, format='pdf', bbox_inches='tight')