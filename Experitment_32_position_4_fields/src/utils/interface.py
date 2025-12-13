import os
import logging
import datetime
from pathlib import Path
from .hyperparameters import (fs_default, 
                               t1_default, t2_default, 
                               dead_time_default, period_default, 
                               reg_par_default, N_tau_default, 
                               tau_sampling_default, 
                               tau_min_default, 
                               tau_max_default
                               )


def get_paths_and_parameters(shared_path):
    """
    Handles the command-line interface logic and returns the necessary paths and parameters.

    Parameters:
        shared_path (str): The base path to the shared directory.

    Returns:
        dict: A dictionary containing shared_path, date, experiment, selected_samples, fs, idx, and figures_path.
    """
    import datetime

    # List available dates (folders in shared_path)
    if not os.path.exists(shared_path):
        print(f"Shared path does not exist: {shared_path}")
        sys.exit(1)

    available_dates = [d for d in os.listdir(shared_path) if os.path.isdir(os.path.join(shared_path, d))]
    if not available_dates:
        print(f"No date folders found in shared path: {shared_path}")
        sys.exit(1)

    available_dates.sort()  # Sort the dates for consistent ordering

    print("\nAvailable dates:")
    for i, date_folder in enumerate(available_dates, start=1):
        print(f"{i}. {date_folder}")

    # Get today's date in "DD.MM.YY" format
    today = datetime.date.today()
    default_date = today.strftime('%d.%m.%y')  # Date in "DD.MM.YY" format

    # Ask user for date (default today)
    date_input = input(f"\nEnter date (number or name) [default: {default_date}]: ").strip()
    if date_input == '':
        date = default_date
    else:
        # Try to interpret input as an index
        try:
            date_index = int(date_input) - 1
            if 0 <= date_index < len(available_dates):
                date = available_dates[date_index]
            else:
                print("Invalid selection. Using default date.")
                date = default_date
        except ValueError:
            # Check if input matches a date name
            if date_input in available_dates:
                date = date_input
            else:
                print("Invalid date input. Using default date.")
                date = default_date

    # Build the date path
    date_path = os.path.join(shared_path, date)
    if not os.path.exists(date_path):
        print(f"No data found for date {date} at path {date_path}")
        sys.exit(1)

    # List experiments in the date path
    experiments = [d for d in os.listdir(date_path) if os.path.isdir(os.path.join(date_path, d))]
    if not experiments:
        print(f"No experiments found in {date_path}")
        sys.exit(1)

    experiments.sort()  # Sort for consistent ordering

    print("\nAvailable experiments:")
    for i, exp in enumerate(experiments, start=1):
        print(f"{i}. {exp}")

    default_experiment = experiments[0]
    experiment_input = input(f"\nSelect experiment (number or name) [default: {default_experiment}]: ").strip()
    if experiment_input == '':
        experiment = default_experiment
    else:
        # Try to interpret input as an index
        try:
            exp_index = int(experiment_input) - 1
            if 0 <= exp_index < len(experiments):
                experiment = experiments[exp_index]
            else:
                print("Invalid selection. Using default experiment.")
                experiment = default_experiment
        except ValueError:
            # Check if input matches an experiment name
            if experiment_input in experiments:
                experiment = experiment_input
            else:
                print("Invalid experiment input. Using default experiment.")
                experiment = default_experiment

    # Build the experiment path
    experiment_path = os.path.join(date_path, experiment)
    if not os.path.exists(experiment_path):
        print(f"Experiment path does not exist: {experiment_path}")
        sys.exit(1)

    # List samples in the experiment path
    samples = [d for d in os.listdir(experiment_path) if os.path.isdir(os.path.join(experiment_path, d))]
    if not samples:
        print(f"No samples found in {experiment_path}")
        sys.exit(1)

    samples.sort()  # Sort for consistent ordering

    print("\nAvailable samples:")
    for i, sample in enumerate(samples, start=1):
        print(f"{i}. {sample}")

    default_samples = 'all'
    samples_input = input(f"\nSelect samples (numbers or names, comma-separated, or 'all') [default: {default_samples}]: ").strip()
    if samples_input == '':
        selected_samples = samples
    elif samples_input.lower() == 'all':
        selected_samples = samples
    else:
        # Parse input
        selected_samples = []
        inputs = [s.strip() for s in samples_input.split(',')]
        for s in inputs:
            # Try to interpret each input as an index
            try:
                s_index = int(s) - 1
                if 0 <= s_index < len(samples):
                    selected_samples.append(samples[s_index])
                else:
                    print(f"Invalid sample index: {s}")
            except ValueError:
                # Check if input matches a sample name
                if s in samples:
                    selected_samples.append(s)
                else:
                    print(f"Invalid sample name: {s}")
        if not selected_samples:
            print("No valid samples selected. Exiting.")
            sys.exit(1)

    print("\nSelected samples:")
    for sample in selected_samples:
        print(f"- {sample}")

    # Ask user for 'fs' (sampling frequency)
    fs_default = 100000  # Default sampling frequency
    fs_input = input(f"\nEnter sampling frequency 'fs' [default: {fs_default}]: ").strip()
    if fs_input == '':
        fs = fs_default
    else:
        try:
            fs = float(fs_input)
        except ValueError:
            print("Invalid input. Using default fs.")
            fs = fs_default

    # Now ask for t1, t2, dead_time, with default values

    t1_input = input(f"\nEnter t1 (Width of one impulse in seconds) [default: {t1_default}]: ").strip()
    if t1_input == '':
        t1 = t1_default
    else:
        try:
            t1 = float(t1_input)
        except ValueError:
            print("Invalid input. Using default t1.")
            t1 = t1_default

    t2_input = input(f"\nEnter t2 (Distance between impulses in seconds) [default: {t2_default}]: ").strip()
    if t2_input == '':
        t2 = t2_default
    else:
        try:
            t2 = float(t2_input)
        except ValueError:
            print("Invalid input. Using default t2.")
            t2 = t2_default

    dead_time_input = input(f"\nEnter dead time (in seconds) [default: {dead_time_default}]: ").strip()
    if dead_time_input == '':
        dead_time = dead_time_default
    else:
        try:
            dead_time = float(dead_time_input)
        except ValueError:
            print("Invalid input. Using default dead_time.")
            dead_time = dead_time_default


    # Calculate t3
    t3 = period_default - 2 * t1 - t2

    # Calculate indices
    d_t = int(dead_time * fs)

    i0 = 0  # Starting index
    i1 = int(t1 * fs)
    i2 = int(t2 * fs)
    i3 = int(t3 * fs)
    di = int(fs * (t2 - dead_time))

    idx = [i0, i1, i2, i3, d_t, di]  # [i0, i1, i2, i3, i_dt, di]

    # Now ask for ME analysis parameters: reg_par, tau_sampling, N_tau

    reg_par_input = input(f"\nEnter the regularization parameter for multiexponential analysis [default: {reg_par_default}]: ").strip()
    if reg_par_input == '':
        reg_par = reg_par_default
    else:
        try:
            reg_par = float(reg_par_input)
        except ValueError:
            print("Invalid input. Using default regularization parameter.")
            reg_par = reg_par_default

    tau_sampling_input = input(f"\nEnter the tau sampling type ('uniform' or 'non_uniform') [default: {tau_sampling_default}]: ").strip()
    if tau_sampling_input == '':
        tau_sampling = tau_sampling_default
    elif tau_sampling_input not in ['uniform', 'non_uniform']:
        print("Invalid input. Using default tau sampling type ('uniform').")
        tau_sampling = tau_sampling_default
    else:
        tau_sampling = tau_sampling_input

    N_tau_input = input(f"\nEnter the number of tau points [default: {N_tau_default}]: ").strip()
    if N_tau_input == '':
        N_tau = N_tau_default
    else:
        try:
            N_tau = int(N_tau_input)
        except ValueError:
            print("Invalid input. Using default number of tau points.")
            N_tau = N_tau_default


    tau_min_input = input(f"\nEnter the minimum tau value [default: {tau_min_default}]: ").strip()
    if tau_min_input == '':
        tau_min = tau_min_default
    else:
        try:
            tau_min = float(tau_min_input)
        except ValueError:
            print("Invalid input. Using default tau_min.")
            tau_min = tau_min_default

    tau_max_input = input(f"\nEnter the maximum tau value [default: {tau_max_default}]: ").strip()
    if tau_max_input == '':
        tau_max = tau_max_default
    else:
        try:
            tau_max = float(tau_max_input)
        except ValueError:
            print("Invalid input. Using default tau_max.")
            tau_max = tau_max_default

    # Create figures_path
    figures_base_path = os.path.join(shared_path, 'Figures', date, experiment)
    os.makedirs(figures_base_path, exist_ok=True)


    # Return the collected paths and parameters
    paths_and_params = {
        'shared_path': shared_path,
        'date': date,
        'experiment': experiment,
        'selected_samples': selected_samples,
        'fs': fs,
        'idx': idx,
        'reg_par': reg_par,
        'figures_base_path': figures_base_path,
        'N_tau': N_tau, 
        'tau_sampling': tau_sampling,
        'tau_min': tau_min,
        'tau_max': tau_max
    }

    return paths_and_params





def get_paths_and_parameters_1(shared_path, data, experiment):
    """
    Handles the command-line interface logic and returns the necessary paths and parameters.

    Parameters:
        shared_path (str): The base path to the shared directory.

    Returns:
        dict: A dictionary containing shared_path, date, experiment, selected_samples, fs, idx, and figures_path.
    """
    import datetime

    shared_path = shared_path

    date = data
    experiment = experiment

    date_path = os.path.join(shared_path, date)

    experiment_path = os.path.join(date_path, experiment)

    if not os.path.exists(experiment_path):
        print(f"Experiment path does not exist: {experiment_path}")
        sys.exit(1)

    # List samples in the experiment path
    samples = [d for d in os.listdir(experiment_path) if os.path.isdir(os.path.join(experiment_path, d))]
    if not samples:
        print(f"No samples found in {experiment_path}")
        sys.exit(1)

    samples.sort()  # Sort for consistent ordering

    print("\nAvailable samples:")
    for i, sample in enumerate(samples, start=1):
        print(f"{i}. {sample}")

    default_samples = 'all'
    samples_input = input(f"\nSelect samples (numbers or names, comma-separated, or 'all') [default: {default_samples}]: ").strip()
    if samples_input == '':
        selected_samples = samples
    elif samples_input.lower() == 'all':
        selected_samples = samples
    else:
        # Parse input
        selected_samples = []
        inputs = [s.strip() for s in samples_input.split(',')]
        for s in inputs:
            # Try to interpret each input as an index
            try:
                s_index = int(s) - 1
                if 0 <= s_index < len(samples):
                    selected_samples.append(samples[s_index])
                else:
                    print(f"Invalid sample index: {s}")
            except ValueError:
                # Check if input matches a sample name
                if s in samples:
                    selected_samples.append(s)
                else:
                    print(f"Invalid sample name: {s}")
        if not selected_samples:
            print("No valid samples selected. Exiting.")
            sys.exit(1)

    print("\nSelected samples:")
    for sample in selected_samples:
        print(f"- {sample}")

    # Ask user for 'fs' (sampling frequency)
    fs_default = 100000  # Default sampling frequency
    fs_input = input(f"\nEnter sampling frequency 'fs' [default: {fs_default}]: ").strip()
    if fs_input == '':
        fs = fs_default
    else:
        try:
            fs = float(fs_input)
        except ValueError:
            print("Invalid input. Using default fs.")
            fs = fs_default

    # Now ask for t1, t2, dead_time, with default values

    t1_input = input(f"\nEnter t1 (Width of one impulse in seconds) [default: {t1_default}]: ").strip()
    if t1_input == '':
        t1 = t1_default
    else:
        try:
            t1 = float(t1_input)
        except ValueError:
            print("Invalid input. Using default t1.")
            t1 = t1_default

    t2_input = input(f"\nEnter t2 (Distance between impulses in seconds) [default: {t2_default}]: ").strip()
    if t2_input == '':
        t2 = t2_default
    else:
        try:
            t2 = float(t2_input)
        except ValueError:
            print("Invalid input. Using default t2.")
            t2 = t2_default

    dead_time_input = input(f"\nEnter dead time (in seconds) [default: {dead_time_default}]: ").strip()
    if dead_time_input == '':
        dead_time = dead_time_default
    else:
        try:
            dead_time = float(dead_time_input)
        except ValueError:
            print("Invalid input. Using default dead_time.")
            dead_time = dead_time_default


    # Calculate t3
    t3 = period_default - 2 * t1 - t2

    # Calculate indices
    d_t = int(dead_time * fs)

    i0 = 0  # Starting index
    i1 = int(t1 * fs)
    i2 = int(t2 * fs)
    i3 = int(t3 * fs)
    di = int(fs * (t2 - dead_time))

    idx = [i0, i1, i2, i3, d_t, di]  # [i0, i1, i2, i3, i_dt, di]

    # Now ask for ME analysis parameters: reg_par, tau_sampling, N_tau

    reg_par_input = input(f"\nEnter the regularization parameter for multiexponential analysis [default: {reg_par_default}]: ").strip()
    if reg_par_input == '':
        reg_par = reg_par_default
    else:
        try:
            reg_par = float(reg_par_input)
        except ValueError:
            print("Invalid input. Using default regularization parameter.")
            reg_par = reg_par_default

    tau_sampling_input = input(f"\nEnter the tau sampling type ('uniform' or 'non_uniform') [default: {tau_sampling_default}]: ").strip()
    if tau_sampling_input == '':
        tau_sampling = tau_sampling_default
    elif tau_sampling_input not in ['uniform', 'non_uniform']:
        print("Invalid input. Using default tau sampling type ('uniform').")
        tau_sampling = tau_sampling_default
    else:
        tau_sampling = tau_sampling_input

    N_tau_input = input(f"\nEnter the number of tau points [default: {N_tau_default}]: ").strip()
    if N_tau_input == '':
        N_tau = N_tau_default
    else:
        try:
            N_tau = int(N_tau_input)
        except ValueError:
            print("Invalid input. Using default number of tau points.")
            N_tau = N_tau_default


    tau_min_input = input(f"\nEnter the minimum tau value [default: {tau_min_default}]: ").strip()
    if tau_min_input == '':
        tau_min = tau_min_default
    else:
        try:
            tau_min = float(tau_min_input)
        except ValueError:
            print("Invalid input. Using default tau_min.")
            tau_min = tau_min_default

    tau_max_input = input(f"\nEnter the maximum tau value [default: {tau_max_default}]: ").strip()
    if tau_max_input == '':
        tau_max = tau_max_default
    else:
        try:
            tau_max = float(tau_max_input)
        except ValueError:
            print("Invalid input. Using default tau_max.")
            tau_max = tau_max_default

    # Create figures_path
    figures_base_path = os.path.join(shared_path, 'Figures', date, experiment)
    os.makedirs(figures_base_path, exist_ok=True)


    # Return the collected paths and parameters
    paths_and_params = {
        'shared_path': shared_path,
        'date': date,
        'experiment': experiment,
        'selected_samples': selected_samples,
        'fs': fs,
        'idx': idx,
        'reg_par': reg_par,
        'figures_base_path': figures_base_path,
        'N_tau': N_tau, 
        'tau_sampling': tau_sampling,
        'tau_min': tau_min,
        'tau_max': tau_max
    }

    return paths_and_params
