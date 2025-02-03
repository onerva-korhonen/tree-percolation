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
from scipy.stats import truncnorm, norm
import pickle

import mrad_model
import params

def calculate_min_constriction_radius(n_constrictions, truncnorm_center, truncnorm_std, truncnorm_a, truncnorm_b=np.inf):
    """
    Draws a set of pore constriction radii from a truncated normal distribution and
    return the smallest one.

    Parameters:
    -----------
    n_constrictions : int
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
    radii = rv.rvs(size=n_constrictions)
    min_constriction_radius = np.amin(radii)
    return min_constriction_radius

def calculate_pit_bpp(target_total_area, tf, n_constrictions, truncnorm_center, truncnorm_std, truncnorm_a, truncnorm_b=np.inf, 
                      surface_tension=params.surface_tension, pore_shape_correction=0.5, gas_contact_angle=0):
    """
    Calculates the bubble propagation pressure of a single pit based on the largest pore radius (defined
    as the smallest constriction radius for each pore, drawn from a truncated normal distribution) in the ICC, 
    following the modifed Young-Laplace equation given in the supplementary methods section of Kaack et al. 2021.
    
    Parameters:
    -----------
    target_total_area : float
        the aimed sum of pore areas; adding new pores ends when the total pore area reaches target_total_area
    tf : float
        microfibril strand thickness
    n_constrictions : int
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
    n_pores = 2 * int(np.ceil(target_total_area / (np.pi * (truncnorm_center + tf / 2)**2))) # factor 2 compensates for the fact that some drawn radii are smaller than the average radii and thus yield smaller area; thus more radii are needed than if they all were equal to the average; 2 is an arbitrary but empirically high enough factor

    truncnorm_a = (truncnorm_a - truncnorm_center) / truncnorm_std
    truncnorm_b = (truncnorm_b - truncnorm_center) / truncnorm_std
    rv = truncnorm(truncnorm_a, truncnorm_b, loc=truncnorm_center, scale=truncnorm_std)
    radii = rv.rvs(size=n_pores*n_constrictions)
    radii = np.reshape(radii, (n_constrictions, n_pores))
    radii = np.amin(radii, axis=0) # picking the smallest radii for each pore
    pore_areas = np.pi * (radii + tf/2)**2
    cum_pore_area = np.cumsum(pore_areas)
    cut_index = np.where(cum_pore_area > target_total_area)[0][0]
    max_radius = radii[cut_index]
    
    bpp = (pore_shape_correction * 2 * surface_tension * np.cos(gas_contact_angle)) / max_radius

    return bpp
    
def calculate_bpp_for_constriction_pores(cfg, Am_space=[], net=None, conduits=[], icc_mask=[], bpp_save_path=''):
    """
    Calculates bubble propagation pressure based on the pit membrane model of Kaack et al. 2021:
    each pore is a 3D structure with several objects, the pore diameter is defined by its smallest
    constriction, and the maximum pore diameter defines the BPP across the pit membrane. BPP can
    be calculated either for a given set of pit field area values or for all ICCs of a network. If a save path is given,
    BPPs are also saved as .pkl.

    Parameters
    ----------
    cfg : dict
        contains: 
        conduit_element_length: float, length of a conduit element
        Dc: float, average conduit diameter (m)
        Dc_cv: float, coefficient of variation of conduit diameter
        fc: float, average contact fraction between two conduits
        fpf: float, average pit field fraction between two conduits
        tf: float, microfibril strand thickness
        n_constrictions: int, the number of constrictions per pore
        truncnorm_center: float, center value of the truncated normal distribution
        truncnorm_std : float, standard deviation of the distribution
        truncnorm_a: float, startpoint of the left truncation
        truncnorm_b: float, optional, startpoint of the right truncation, default value np.inf gives left-only truncated distribution
        surface_tension: float, optional, surface tension of sap in the xylem conduits (default: the value given in the parameter file)
        pore_shape_correction: float, correction factor applied to compensate for the inaccurately assumed round shape of all pores, default 0.5 (from Kaack et al. 2021)
        gas_contact_angle: float, optional, the contact angle between gas and xylem sap in radians, default 0 (from Kaack et al. 2021)
    Am_space : np.array of floats, optional
        set of pit field area values, for which the BPP will be calculated, default: []
    net : openpnm.network(), optional
        pores correspond to conduit elements, throats to CECs and ICCs, default: None
    conduits : np.array, optional
        three columns: 1) the first element of the conduit, 2) the last element of the conduit, 3) size of the conduit, default: []
    icc_mask : np.array, optional
        for each throat of the network, contains 1 if the throat is an ICC and 0 otherwise, default: []
    bpp_save_path : str, optional
        path, to which save the BPPs, default: '' (no saving)

    Returns
    -------
    bpp : np.array
        bubble propagation pressure for each Am value or for each ICC of the network
    """
    assert (len(Am_space) > 0) or (net != None), 'please give either Am_space or network object for calculating bubble propagation pressure'
    conduit_element_length = cfg['conduit_element_length']
    Dc = cfg['Dc']
    Dc_cv = cfg['Dc_cv']
    fc = cfg['fc']
    fpf = cfg['fpf']
    tf = cfg['tf']
    n_constrictions = cfg['n_constrictions']
    truncnorm_center = cfg['truncnorm_center']
    truncnorm_std = cfg['truncnorm_std']
    truncnorm_a = cfg['truncnorm_a']
    truncnorm_b = cfg.get('truncnorm_b', np.inf)
    surface_tension = cfg.get('surface_tension', params.surface_tension)
    pore_shape_correction = cfg.get('pore_shape_correction', 0.5)
    gas_contact_angle = cfg.get('gas_contact_angle', 0)
    
    bpp_dic = {}
    
    if net != None:
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
            bpp[i] = calculate_pit_bpp(Am, tf, n_constrictions, truncnorm_center, truncnorm_std, truncnorm_a, truncnorm_b=truncnorm_b, surface_tension=surface_tension, 
                                       pore_shape_correction=pore_shape_correction, gas_contact_angle=gas_contact_angle)
            bpp_dic[Am] = bpp
            
    else:
        bpp = np.zeros(Am_space.shape)
        for Am in Am_space:
            bpp_dic[Am] = calculate_pit_bpp(Am, tf, n_constrictions, truncnorm_center, truncnorm_std, truncnorm_a, truncnorm_b=truncnorm_b, surface_tension=surface_tension, 
                                            pore_shape_correction=pore_shape_correction, gas_contact_angle=gas_contact_angle)
            
    if bpp_save_path != '':
        with open(bpp_save_path, 'wb') as f:
            pickle.dump(bpp_dic, f)
        f.close()
            
    return bpp

def read_constriction_bpp(bpp_data_path, net, conduits, icc_mask, cfg):
    """
    Reads previously calculated bubble propagation pressure data from .pkl file
    and pick the best-matching BPP for each ICC

    Parameters
    ----------
    bpp_data_path : str
        path, to which the BPP data has been saved
    net : openpnm.network(), optional
        pores correspond to conduit elements, throats to CECs and ICCs, default: None
    conduits : np.array, optional
        three columns: 1) the first element of the conduit, 2) the last element of the conduit, 3) size of the conduit, default: []
    icc_mask : np.array, optional
        for each throat of the network, contains 1 if the throat is an ICC and 0 otherwise, default: []
    cfg : dict
        contains: 
        conduit_element_length: float, length of a conduit element
        Dc: float, average conduit diameter (m)
        Dc_cv: float, coefficient of variation of conduit diameter
        fc: float, average contact fraction between two conduits
        fpf: float, average pit field fraction between two conduits

    Returns
    -------
    bpp : np.array
        bubble propagation pressure for each Am value or for each ICC of the network
    """
    with open(bpp_data_path, 'rb') as f:
        bpp_data = pickle.load(f)
    f.close()
    
    Am_space = np.array(list(bpp_data.keys()))
    
    conduit_element_length = cfg['conduit_element_length']
    Dc = cfg['Dc']
    Dc_cv = cfg['Dc_cv']
    fc = cfg['fc']
    fpf = cfg['fpf']
    
    conns = net['throat.conns']

    diameters_per_conduit, _ = mrad_model.get_conduit_diameters(net, 'inherit_from_net', conduits, Dc_cv=Dc_cv, Dc=Dc)
    conduit_areas = (conduits[:, 2] - 1) * conduit_element_length * np.pi * diameters_per_conduit
    iccs = conns[np.where(icc_mask)]
    icc_count = np.array([np.sum((conduit[0] <= iccs[:, 0]) & (iccs[:, 0] <= conduit[1])) + np.sum((conduit[0] <= iccs[:, 1]) & (iccs[:, 1] <= conduit[1])) for conduit in conduits])
    
    bpp = np.zeros(iccs.shape[0])
    Ams = np.zeros(iccs.shape[0])
    
    for i, icc in enumerate(iccs):
        start_conduit = np.where((conduits[:, 0] <= icc[0]) & (icc[0] <= conduits[:, 1]))[0][0]
        end_conduit = np.where((conduits[:, 0] <= icc[1]) & (icc[1] <= conduits[:, 1]))[0][0] 
        Am = 0.5 * (conduit_areas[start_conduit] / icc_count[start_conduit] + conduit_areas[end_conduit] / icc_count[end_conduit]) * fc * fpf # Mrad et al. 2018, Eq. 2; surface area of the ICC
        Ams[i] = Am
        Am_key = Am_space[np.argmin(np.abs(Am_space - Am))]
        bpp[i] = bpp_data[Am_key]
    

    return bpp

if __name__=='__main__':
    Am_space = np.logspace(np.log10(1E-9), np.log10(4E-8), num=100)
    
    cfg = {}
    cfg['net_size'] = params.net_size
    cfg['conduit_diameters'] = 'lognormal'#mrad_params.conduit_diameters
    cfg['Dc'] = params.Dc
    cfg['Dc_cv'] = params.Dc_cv
    cfg['conduit_element_length'] = params.Lce
    cfg['fc'] = params.fc
    cfg['average_pit_area'] = params.average_pit_membrane_area
    cfg['fpf'] = params.fpf
    cfg['tf'] = params.tf
    cfg['Dp'] = params.Dp
    cfg['Tm'] = params.Tm
    cfg['weibull_a'] = params.weibull_a
    cfg['weibull_b'] = params.weibull_b
    cfg['n_constrictions'] = params.n_constrictions
    cfg['truncnorm_center'] = params.truncnorm_center
    cfg['truncnorm_std'] = params.truncnorm_std
    cfg['truncnorm_a'] = params.truncnorm_a
    cfg['pore_shape_correction'] = params.pore_shape_correction
    cfg['gas_contact_angle'] = params.gas_contact_angle
    cfg['icc_length'] = params.icc_length
    cfg['seeds_NPc'] = params.seeds_NPc
    cfg['seeds_Pc'] = params.seeds_Pc
    cfg['seed_ICC_rad'] = params.seed_ICC_rad
    cfg['seed_ICC_tan'] = params.seed_ICC_tan
    cfg['si_length'] = params.si_length
    cfg['si_tolerance_length'] = params.si_tolerance_length
    cfg['si_type'] = params.si_type
    cfg['start_conduits'] = params.start_conduits
    cfg['surface_tension'] = params.surface_tension
    cfg['pressure'] = params.pressure
    cfg['nCPUs'] = params.nCPUs
    cfg['spontaneous_embolism'] = params.spontaneous_embolism
    cfg['bpp_type'] = params.bpp_type
    cfg['bpp_data_path'] = params.bubble_propagation_pressure_data_path
    
    bpps = calculate_bpp_for_constriction_pores(cfg, Am_space=Am_space, bpp_save_path=cfg['bpp_data_path'])




