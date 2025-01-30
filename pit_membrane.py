#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A more realistic pit membrane model for calculating the bubble propagation pressure. Following model 1 from Kaack et al. 2021, New Phytologist
(https://nph.onlinelibrary.wiley.com/doi/10.1111/nph.17282 ), the model considers each pore in the pit as a set of constrictions that spans 
through the pit membrane with non-negligible thickness. The diameter of the constrictions is drawn from a truncated normal distribution,
the diameter of each pore corresponds to the diameter of its narrowest constriction, and the maximum pore diameter defines the bubble
propagation pressure across the pit membrane. 

Created on Thu Jan 30 11:56:59 2025

@author: onervak
"""
import numpy as np
from scipy.stats import truncnorm

import mrad_model
import params

def calculate_min_constriction_radius(N_constrictions, truncnorm_center, truncnorm_std, truncnorm_a, truncnorm_b=np.inf):
    """
    Draws a set of pore constriction radii from a truncated normal distribution and
    return the smallest one.

    Parameters:
    -----------
    N_constrictions : int
        number of values to be drawn from the distribution, i.e. the number of constrictions in the pore
    truncnorm_center : float
        center value of the distribution
    truncnorm_std : float
        standard deviation of the distribution
    truncnorm_a : float
        startpoint of the left truncation
    truncnorm_b : float, optional
        startpoint of the right truncation, default value np.inf gives left-only truncated distribution

    Returns
    -------
    min_constriction_radius : float, smallest of the radii drawn from the distribution
    """
    truncnorm_a = (truncnorm_a - truncnorm_center) / truncnorm_std
    truncnorm_b = (truncnorm_b - truncnorm_center) / truncnorm_std
    rv = truncnorm(truncnorm_a, truncnorm_b, loc=truncnorm_center, scale=truncnorm_std)
    radii = rv.rvs(size=N_constrictions)
    min_constriction_radius = np.amin(radii)
    return min_constriction_radius

def calculate_pit_BPP(average_pit_area, tf, N_constrictions, truncnorm_center, truncnorm_std, truncnorm_a, truncnorm_b=np.inf, 
                      surface_tension=params.surface_tension, pore_shape_correction=0.5, gas_contact_angle=0):
    """
    Calculates the bubble propagation pressure of a single pit based on the largest pore radius (defined
    as the smallest constriction radius for each pore, drawn from a truncated normal distribution) in the ICC, 
    following the modifed Young-Laplace equation given in the supplementary methods section of Kaack et al. 2021.
    
    Parameters:
    -----------
    average_pit_area : float
        the average area of a pit
    tf : float
        microfibril strand thickness
    N_constrictions : int
        the number of constrictions per pore
    truncnorm_center : float
        center value of the truncated normal distribution
    truncnorm_std : float
        standard deviation of the distribution
    truncnorm_a : float
        startpoint of the left truncation
    truncnorm_b : float, optional
        startpoint of the right truncation, default value np.inf gives left-only truncated distribution
    surface_tension : float, optional
        surface tension of sap in the xylem conduits (default: the value given in the parameter file)
    pore_shape_correction : float, optional
        correction factor applied to compensate for the inaccurately assumed round shape of all pores, default 0.5 (from Kaack et al. 2021)
    gas_contact_angle : float, optional
        the contact angle between gas and xylem sap in radians, default 0 (from Kaack et al. 2021)
        
    Returns:
    --------
    bpp : float
        bubble propagation pressure across the pit membrane
    """
    total_pore_area = 0
    pore_radii = []
    while total_pore_area <= average_pit_area:
        pore_radius = calculate_min_constriction_radius(N_constrictions, truncnorm_center, truncnorm_std, truncnorm_a)
        pore_radii.append(pore_radius)
        total_pore_area += np.pi * (pore_radius + tf / 2)**2
    max_radius = np.amax(np.array(pore_radii))
    bpp = (pore_shape_correction * 2 * surface_tension * np.cos(gas_contact_angle)) / max_radius
    return bpp
    
def calculate_BPP_for_constriction_pores(net, conduits, icc_mask, cfg):
    """
    Calculates bubble propagation pressure for each ICC using calculate_pit_BPP (see above) and selecting
    for each ICC the minimum BPP of its pits.

    Parameters
    ----------
    net : openpnm.network()
        pores correspond to conduit elements, throats to CECs and ICCs
    conduits : np.array
        three columns: 1) the first element of the conduit, 2) the last element of the conduit, 3) size of the conduit
    icc_mask : np.array
        for each throat of the network, contains 1 if the throat is an ICC and 0 otherwise
    cfg : dict
        contains: 
        average_pit_area : float, the average area of a pit
        conduit_element_length: float, length of a conduit element
        Dc: float, average conduit diameter (m)
        Dc_cv: float, coefficient of variation of conduit diameter
        fc: float, average contact fraction between two conduits
        fpf: float, average pit field fraction between two conduits
        tf: float, microfibril strand thickness
        N_constrictions: int, the number of constrictions per pore
        truncnorm_center: float, center value of the truncated normal distribution
        truncnorm_std : float, standard deviation of the distribution
        truncnorm_a: float, startpoint of the left truncation
        truncnorm_b: float, optional, startpoint of the right truncation, default value np.inf gives left-only truncated distribution
        surface_tension: float, optional, surface tension of sap in the xylem conduits (default: the value given in the parameter file)
        pore_shape_correction: float, correction factor applied to compensate for the inaccurately assumed round shape of all pores, default 0.5 (from Kaack et al. 2021)
        gas_contact_angle: float, optional, the contact angle between gas and xylem sap in radians, default 0 (from Kaack et al. 2021)
    Returns
    -------
    bpp : np.array
        bubble propagation pressure for each ICC of the network
    """    
    average_pit_area = cfg['average_pit_area']
    conduit_element_length = cfg['conduit_element_length']
    Dc = cfg['Dc']
    Dc_cv = cfg['Dc_cv']
    fc = cfg['fc']
    fpf = cfg['fpf']
    tf = cfg['tf']
    N_constrictions = cfg['N_constrictions']
    truncnorm_center = cfg['truncnorm_center']
    truncnorm_std = cfg['truncnorm_std']
    truncnorm_a = cfg['truncnorm_a']
    truncnorm_b = cfg.get('truncnorm_b', np.inf)
    surface_tension = cfg.get('surface_tension', params.surface_tension)
    pore_shape_correction = cfg.get('pore_shape_correction', 0.5)
    gas_contact_ange = cfg.get('gas_contact_angle', 0)
    conns = net['throat.conns']
    
    diameters_per_conduit, _ = mrad_model.get_conduit_diameters(net, 'inherit_from_net', conduits, Dc_cv=Dc_cv, Dc=Dc)
    conduit_areas = (conduits[:, 2] - 1) * conduit_element_length * np.pi * diameters_per_conduit
    iccs = conns[np.where(icc_mask)]
    icc_count = np.array([np.sum((conduit[0] <= iccs[:, 0]) & (iccs[:, 0] <= conduit[1])) + np.sum((conduit[0] <= iccs[:, 1]) & (iccs[:, 1] <= conduit[1])) for conduit in conduits])
    
    bpp = np.zeros(iccs.shape[0])
    
    for i, icc in enumerate(iccs):
        start_conduit = np.where((conduits[:, 0] <= icc[0]) & (icc[0] <= conduits[:, 1]))[0][0]
        end_conduit = np.where((conduits[:, 0] <= icc[1]) & (icc[1] <= conduits[:, 1]))[0][0] 
        Am = 0.5 * (conduit_areas[start_conduit] / icc_count[start_conduit] + conduit_areas[end_conduit] / icc_count[end_conduit]) * fc * fpf # Mrad et al. 2018, Eq. 2; surface area of the ICC
        pit_count = int(np.floor(Am / average_pit_area)) # Mrad et al. 2018, Eq. 3 (using average pit area as input instead of average pit diameter); number of pits in the ICC
        pit_bpps = [calculate_pit_BPP(average_pit_area, tf, N_constrictions, truncnorm_center, truncnorm_std, truncnorm_a,
                                      truncnorm_b=truncnorm_b, surface_tension=surface_tension, 
                                      pore_shape_correction=pore_shape_correction, gas_contact_angle=gas_contact_ange) for j in np.arange(pit_count)] #TODO: check if it's possible to parallelize across pits or if one could calculate directly for entire ICC instead of looping over pits (plot both!)
        bpp[i] = np.amin(pit_bpps)
        
    return bpp



