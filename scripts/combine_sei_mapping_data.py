#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 12:05:40 2026

@author: Onerva Korhonen

A script for reading, combining and visualizing the SEI output space mapping
data.
"""
import numpy as np
import pickle
import os
import matplotlib.pylab as plt

import params

if __name__ == '__main__':
    mapping_data_save_folder, mapping_data_save_name_base = params.sei_mapping_data_save_path_base.rsplit('/', 1)
    
    data_files = [os.path.join(mapping_data_save_folder, file) for file in os.listdir(mapping_data_save_folder) if os.path.isfile(os.path.join(mapping_data_save_folder, file))]
    data_files = [data_file for data_file in data_files if mapping_data_save_name_base in data_file]
    
    unique_se_probabilities = []
    unique_ei_probabilities = []
    parameter_combinations = []
    plcs = []
    simulation_lengths = []
    
    for data_file in data_files:
        with open(data_file, 'rb') as f:
            data = pickle.load(f)
            f.close()
        se_probability = data['se_probability']
        ei_probabilities = data['ei_probabilities']
        plc = data['plcs']
        simulation_length = data['simulation_lengths']
        
        if not se_probability in unique_se_probabilities:
            unique_se_probabilities.append(se_probability)
            
        for ei_probability in ei_probabilities:
            if not ei_probability in unique_ei_probabilities:
                unique_ei_probabilities.append(ei_probability)
                parameter_combinations.append((se_probability, ei_probability))
                
        plcs.extend(plc)
        simulation_lengths.extend(simulation_length)
    
    unique_se_probabilities = np.sort(unique_se_probabilities)
    unique_ei_probabilities = np.sort(unique_ei_probabilities)
    n_se_probabilities = len(unique_se_probabilities)
    n_ei_probabilities = len(unique_ei_probabilities)
    realized_n_iterations = np.zeros((n_se_probabilities,  n_ei_probabilities))
    averaged_plcs = np.zeros((n_se_probabilities,  n_ei_probabilities))
    averaged_simulation_lengths = np.zeros((n_se_probabilities,  n_ei_probabilities))
    
    for parameter_combination, plc, simulation_length in zip(parameter_combinations, plcs, simulation_lengths):
        se_index = np.where(unique_se_probabilities == parameter_combination[0])[0]
        ei_index = np.where(unique_ei_probabilities == parameter_combination[1])[0]
        realized_n_iterations[se_index, ei_index] += 1
        averaged_plcs[se_index, ei_index] += plc
        averaged_simulation_lengths[se_index, ei_index] += simulation_length
        
    averaged_plcs /= realized_n_iterations
    averaged_simulation_lengths /= realized_n_iterations
    
    centers = [unique_se_probabilities.min(), unique_se_probabilities.max(), unique_ei_probabilities.min(), unique_ei_probabilities.max()]
    
    dx, = np.diff(centers[:2])/(data.shape[1]-1)
    dy, = -np.diff(centers[2:])/(data.shape[0]-1)
    extent = [centers[0]-dx/2, centers[1]+dx/2, centers[2]+dy/2, centers[3]-dy/2]
    
    plc_fig = plt.figure()
    plc_ax = plc_fig.add_subplot(111)
    
    # TODO: set vmin and vmax correctly in params
    plt.imshow(data, origin='lower', extent=extent, vmin=params.sei_plc_vmin, vmax=params.sei_plc_vmax)
    plt.colorbar(label='PLC (%)')
    plc_ax.set_yticks(unique_se_probabilities)
    plc_ax.set_xticks(unique_ei_probabilities)
    plc_ax.set_ylabel('E-I transition probability')
    plc_ax.set_xlabel('S-E transition probability')
    
    save_path = params.sei_mapping_plc_fig_save_path
    plt.savefig(save_path, format='pdf',bbox_inches='tight')
    
    length_fig = plt.figure()
    length_ax = length_fig.add_subplot(111)
    
    plt.imshow(data, origin='lower', extent=extent, vmin=params.sei_simulation_length_vmin, vmax=params.sei_simulation_length_vmax)
    plt.colorbar(label='Simulation length (steps)')
    length_ax.set_yticks(unique_se_probabilities)
    length_ax.set_xticks(unique_ei_probabilities)
    length_ax.set_ylabel('E-I transition probability')
    length_ax.set_xlabel('S-E transition probability')
    
    save_path = params.sei_mapping_simulation_fig_save_path
    plt.savefig(save_path, format='pdf',bbox_inches='tight')
    


