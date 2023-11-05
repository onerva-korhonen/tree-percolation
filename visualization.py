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
    
def plot_percolation_curve(total_n_nodes, values, colors=[], labels=[], alphas=[], y_labels=[]):
    """
    Plots the percolation analysis outcomes (largest connected component size and effective conductance) as a function
    of the fraction of nodes removed.
    
    #TODO: modify this function to plot both LCC and effective conductance and update documentation.

    Parameters
    ----------
    total_n_nodes : int
        number of nodes in the network subject to percolation analysis (used for constructing x axis)
    values : np.array
        each row corresponds to a set of percolation outcome values to be plotted against the fraction of
        removed pores. if values.shape[1] < total_n_nodes + 1, the assumption is that only total_n_nodes + 1 - len(effective_conductance)
        nodes have been removed
    colors : list of strs, optional
        colors used for plotting. len(colors) should equal to values.shape[0]
    labels : list of strs, optional
        labels of the curves plotted. len(labels) should equal to values.shape[0]
    alphas : iterable of floats
        transparency values of the plotted lines. len(alphas) should equal to values.shape[0]
    y_labels : list of strs, optional
        labels of the first and secondary y axis. only used if values.shape[0] == 2, in which case len(y_labels) should be 2

    Returns
    -------
    None.

    """
    #import pdb; pdb.set_trace()
    n_percolation_steps = values.shape[1]
    assert n_percolation_steps > 1, 'only the values calculated for the full network given for plotting percolation curves'
    if n_percolation_steps < total_n_nodes - 1:
        print('Warning: number of effective conductance values from percolation analysis does not match the number of nodes')
        x = np.arange(0, n_percolation_steps / 100, (n_percolation_steps / 100) / total_n_nodes)
    else:
        x = np.arange(0, 100, 100 / total_n_nodes)
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    if values.shape[0] == 2:
        ax2 = ax.twinx()
        ax.plot(x, values[0,:], color=colors[0], label=labels[0], alpha=alphas[0])
        ax2.plot(x, values[1,:], color=colors[1], label=labels[1], alpha=alphas[1])
        ax.set_ylabel(y_labels[0])
        ax2.set_ylabel(y_labels[1])
        ax2.legend()
    else:    
        for value, color, label, alpha in zip(values, colors, labels, alphas):
            ax.plot(x, value, color=color, label=label, alpha=alpha)
            ax.set_ylabel('Percolation outcome')
    
    ax.set_xlabel('Fraction of nodes removed')
    ax.legend()
