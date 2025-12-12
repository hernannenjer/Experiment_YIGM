import sys
import os
import logging
import pandas as pd
import numpy as np

# Assuming 'src' is the main package directory containing models, utils, etc.
# Ensure that 'src/__init__.py' exists so that 'from src...' imports work.
sys.path.insert(0, 'src')

from src.utils.interface import get_paths_and_parameters, get_paths_and_parameters_1
from src.models.experiment import Experiment
from src.plotting.plot_deps_vs_sample import plot_peak_amplitudes_vs_sample

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


def main(experiment):
    # Get paths and parameters from the command-line interface
    # Adjust 'shared' as needed if your directory structure differs
    shared_path = os.path.dirname(os.path.abspath(__file__))

    # THEN create the full shared path
    data = "first_experiment_raw_data_channels"
   

    paths_and_params = get_paths_and_parameters_1(shared_path,data, experiment )


    shared_path = paths_and_params['shared_path']
    date = paths_and_params['date']
    experiment = paths_and_params['experiment']
    selected_samples = paths_and_params['selected_samples']
    fs = paths_and_params['fs']
    idx = paths_and_params['idx']
    figures_base_path = paths_and_params['figures_base_path']
    reg_par = paths_and_params['reg_par']
    tau_sampling = paths_and_params['tau_sampling']
    N_tau = paths_and_params['N_tau']
    tau_min = paths_and_params['tau_min']
    tau_max = paths_and_params['tau_max']


    

    # Build the experiment path
    experiment_path = os.path.join(shared_path, date, experiment)
    print(experiment_path)

    # Initialize the Experiment class
    experiment_obj = Experiment(experiment_dir=experiment_path, figures_base_path=figures_base_path)

    # Load data for the selected samples
    experiment_obj.load_data(idx=idx, fs=fs, parse=True, sample_list=selected_samples)

    # Subtract channels to create gradientometric data (example: channel 0 - channel 1)
    experiment_obj.subtract_channels_in_all_samples(channel_num_1=0, channel_num_2=1)


    baselines_ch_0, baselines_ch_1, baselines_ch_2 =experiment_obj.subtract_baseline_from_all_samples(method='mean_end', n_points=500)

    print(baselines_ch_0, baselines_ch_1, baselines_ch_2)

   
    

    # Create the full path for saving the file
    output_file = os.path.join(experiment_path, "information_baseline_ch0_exp_1_new.txt")
    print(output_file)
    # Create index array (0-based or 1-based depending on your preference)
    indices = np.arange(len(baselines_ch_0))
    
    # Create DataFrame with index and data
    df = pd.DataFrame({
        'index': indices,
        'baseline_value': baselines_ch_0
    })
    
    # Save to text file with tab separation
    df.to_csv(output_file, sep='\t', index=False)



if __name__ == "__main__":

    shared_1 = os.path.dirname(os.path.abspath(__file__))

    # THEN create the full shared path
    name_extesion = "first_experiment_raw_data_channels"

    date_path = os.path.join(shared_1, name_extesion)

    # # List currents in the date path
    experiments = [d for d in os.listdir(date_path) if os.path.isdir(os.path.join(date_path, d))]
    if not experiments:
        print(f"No experiments found in {date_path}")
        sys.exit(1)

    experiments.sort()  # Sort for consistent ordering

    print("\nAvailable experiments:")

    for i, exp in enumerate(experiments, start=1):
        print(f"{i}. {exp}")

    print("\nAnalizing the baselines for all currents:")

    # for number_current in experiments:
    #     print("\nFor obtaining the baselines in the experiments for current "+number_current)
    #     main(number_current)



    '''Concatenate all baselines obtained from each current and save it '''
    baseline_ch_0_list = []


    for number_current in experiments:
        experiment_path = os.path.join(date_path, number_current)
        addres_name = 'information_baseline_ch0_exp_1_new.txt'
        file_name = os.path.join(experiment_path, addres_name)
        signals = pd.read_csv(file_name,sep = '\s+').loc[:,["index","baseline_value"]]['baseline_value']
        baseline_ch_0_list.append(np.array(signals))
    
    baselines_experiment_1 = np.concatenate(baseline_ch_0_list)
    



    # Create the full path for saving the file
    output_file = os.path.join(date_path, "information_baselines_ch0_exp_1.txt")
   
    # Create index array (0-based or 1-based depending on your preference)
    indices = np.arange(len(baselines_experiment_1))
    
    # Create DataFrame with index and data
    df = pd.DataFrame({
        'index': indices,
        'baseline_value': baselines_experiment_1
    })
    
    # Save to text file with tab separation
    df.to_csv(output_file, sep='\t', index=False)
    print('The baselines extracted sucessfull:' , df['baseline_value'] )
    print('Baselines saving in ', output_file)
    

