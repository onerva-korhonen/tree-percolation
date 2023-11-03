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
    
def plot_percolation_curve(total_n_nodes, effective_conductance, lcc=[], color=[], label=[]):
    """
    Plots the percolation analysis outcomes (largest connected component size and effective conductance) as a function
    of the fraction of nodes removed.
    
    #TODO: modify this function to plot both LCC and effective conductance and update documentation.

    Parameters
    ----------
    total_n_nodes : int
        number of nodes in the network subject to percolation analysis (used for constructing x axis)
    effective conductance : np.array
        effective conductance values in the full network and after each ode removal. 
        if len(effective_conductance) < total_n_nodes + 1, the assumption is that only total_n_nodes + 1 - len(effective_conductance)
        nodes have been removed
    color : str
        DESCRIPTION.
    label : str
        DESCRIPTION.

    Returns
    -------
    None.

    """
    assert len(effective_conductance) > 1, 'only the effective conductance of the full network given for plotting percolation curves'
    if len(effective_conductance) < total_n_nodes - 1:
        print('Warning: number of effective conductance values from percolation analysis does not match the number of nodes')
        x = np.arange(0, len(effective_conductance) / 100, (len(effective_conductance) / 100) / total_n_nodes)
    else:
        x = np.arange(0, 100, 100 / total_n_nodes)
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(x, effective_conductance, color=color, label=label)
    ax.set_xlabel('Fraction of nodes removed')
    ax.set_ylabel('Effective conductance')
