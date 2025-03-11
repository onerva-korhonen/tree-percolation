"""
A script for optimizing SI spreading probability based on previously calculated data.

After running this, run visualize_optimized_vc.py and visualize_single_param_spreading.py

Written by Onerva Korhonen
"""
import params
import percolation

max_n_iterations = 100

if __name__=='__main__':
    simulation_data_save_folder = params.optimized_spreading_probability_save_path_base.rsplit('/', 1)[0]
    simulation_data_save_name_base = params.optimized_spreading_probability_save_path_base.rsplit('/', 1)[1]
    pooled_data_save_path = simulation_data_save_folder + '/' + params.pooled_optimized_spreading_probability_vs_empirical_save_name
    empirical_physiological_data_path = params.empirical_vulnerability_save_path
    percolation.optimize_spreading_probability_against_empirical_vc(simulation_data_save_folder, simulation_data_save_name_base, pooled_data_save_path, empirical_physiological_data_path, max_n_iterations=max_n_iterations)
