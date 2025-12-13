#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
process_3D_baseline.py

Compute and visualize baseline (DC offset) maps of magnetometer signals for
*every* sensor channel and for *every* excitation coil current that appears in
"information.txt".

The script relies *only* on functionality that already exists in the
project:
    - classes Experiment, Sample, Attempt, Channel
    - helpers in curve_preprocessing.py
    - helper function plot_3d_from_array

Workflow:
1. Build a gradiometer channel (id -1) by subtracting channel 1 from channel 0.
2. Parse "information.txt". Expected columns: index  X  Y  Z  current.
3. Remove baselines from all samples with a user specified method.
4. For each channel write a file "information_baseline_ch<channel>.txt" that
   repeats the original columns and appends a new column "baseline".
5. For every unique current value create a separate 3-D figure for each channel
   and save it as "baseline_ch<channel>_<current>A.png".

All parameters are requested interactively.  Comments are pure ASCII to avoid
encoding issues under any locale.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Dict, List
import os
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Ensure project root is on PYTHONPATH
# ---------------------------------------------------------------------------

sys.path.insert(0, "src")

# Project specific imports (type checkers can ignore them)
from src.utils.interface import get_paths_and_parameters, get_paths_and_parameters_1 # type: ignore
from src.models.experiment import Experiment              # type: ignore
from src.plotting.plot_3d import plot_3d_from_array       # type: ignore

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _to_scalar(value) -> float:
    """Convert a scalar or ndarray to a single float."""
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return float("nan")
        return float(np.mean(value))
    try:
        return float(value)
    except Exception:
        return float("nan")
    


def prompt_baseline_settings() -> tuple[str, int]:
    """Ask the user which baseline method and window size to use."""
    methods = [
        "mean_end", "median", "polynomial", "lowpass", "3points", "scalar", "None",
    ]
    default_method = "mean_end"

    entered = input(
        f"Baseline method {methods} [mean_end]: ").strip()
    method = entered if entered in methods else default_method

    while True:
        n_text = input("Number of trailing points for baseline [500]: ").strip() or "500"
        try:
            n_points = int(n_text)
            break
        except ValueError:
            print("Please enter an integer, for example 500.")

    return method, n_points


def prompt_baselines_settings() -> tuple[str, int]:
    """Ask the user which baseline method and window size to use."""
    
    default_method = "mean_end"

    method =  default_method

    
    n_text = "500"
    
    n_points = int(n_text)

    return method, n_points




def collect_baseline_per_channel(
    experiment: Experiment,
    channels: List[int],
) -> Dict[int, Dict[int, float]]:
    """Return nested dict {channel_num: {sample_index: baseline_scalar}}."""
    result: Dict[int, Dict[int, float]] = {ch: {} for ch in channels}

    for sample_name, sample in experiment.samples.items():
        try:
            idx = int(sample_name)
        except ValueError:
            # Skip samples whose names are not integers
            continue

        for attempt in sample.attempts:
            for ch in attempt.channels:
                if ch.channel_num not in result:
                    continue  # channel not requested
                baseline_val = getattr(ch, "baseline", None)
                if baseline_val is None:
                    continue
                if idx in result[ch.channel_num]:
                    continue  # already recorded for this sample
                result[ch.channel_num][idx] = _to_scalar(baseline_val)

    return result

# ---------------------------------------------------------------------------
# Main routine
# ---------------------------------------------------------------------------

def main(experiment) -> None:  # noqa: C901
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


    shared_path = os.path.dirname(os.path.abspath(__file__))

    # THEN create the full shared path
    data = "paa_w_50mcg_v_redone"
   

    params = get_paths_and_parameters_1(shared_path,data, experiment )




    # 1. Ask baseline settings
    baseline_method, n_points = prompt_baselines_settings()




    exp_dir = Path(params["shared_path"]) / params["date"] / params["experiment"]
    info_file = exp_dir / "information.txt"

    if not info_file.exists():
        logging.error("%s not found - aborting.", info_file)
        return

    # 3. Load raw data
    exp = Experiment(
        experiment_dir=str(exp_dir),
        figures_base_path=params["figures_base_path"],
    )
    exp.load_data(
        idx=params["idx"],
        fs=params["fs"],
        parse=True,
        sample_list=params["selected_samples"],
    )

    # 4. Build gradiometer channel and subtract baselines
    exp.subtract_channels_in_all_samples(channel_num_1=0, channel_num_2=1)
    exp.subtract_baseline_from_all_samples(method=baseline_method, n_points=n_points)

    # 5. Detect available channels (including new gradiometer "-1")
    all_channel_nums = sorted({
        ch.channel_num
        for s in exp.samples.values()
        for a in s.attempts
        for ch in a.channels
    })
    logging.info("Channels detected: %s", all_channel_nums)

    # 6. Load coordinate and current table
    raw_info = np.loadtxt(info_file)
    if raw_info.shape[1] < 5:
        logging.error(
            "information.txt must have at least 5 columns: index X Y Z current"
        )
        return

    meas_idx = raw_info[:, 0].astype(int)
    coords = raw_info[:, 1:4]
    currents = raw_info[:, 4]
    unique_currents = np.unique(currents)
    logging.info("Unique currents: %s", unique_currents)

    # 7. Collect baselines
    baseline_dicts = collect_baseline_per_channel(exp, all_channel_nums)

    # 8. Write per channel augmented tables
    header_txt = "X Y Z current baseline"
    for ch in all_channel_nums:
        baselines = np.array([
            baseline_dicts[ch].get(idx, np.nan) for idx in meas_idx
        ])
        out_path = info_file.with_name(f"information_baseline_ch{ch}.txt")
        combined = np.column_stack([raw_info[:, :5], baselines])
        np.savetxt(out_path, combined, fmt="%.6g", header=header_txt)
        logging.info("Saved %s", out_path.name)

    # # 9. Render 3-D plots for every current and channel
    # for ch in all_channel_nums:
    #     for cur in unique_currents:
    #         mask = np.isclose(currents, cur)
    #         if not mask.any():
    #             continue

    #         baselines = np.array([
    #             baseline_dicts[ch].get(idx, np.nan) for idx in meas_idx[mask]
    #         ])
    #         if np.all(np.isnan(baselines)):
    #             logging.info(
    #                 "Channel %d, current %.3g A: no baseline values - skipping.",
    #                 ch, cur,
    #             )
    #             continue

    #         data5 = np.column_stack([meas_idx[mask], coords[mask], baselines])
    #         fig_path = exp_dir / f"baseline_ch{ch}_{cur:.3g}A.png"

    #         try:
    #             plot_3d_from_array(
    #                 data_array=data5,
    #                 point_size=10,
    #                 cmap="viridis",
    #                 show=True,
    #                 save_path=str(fig_path),  # if helper supports it
    #                 view = 'xy'
    #             )
    #         except TypeError:
    #             # Fallback if helper lacks save_path
    #             import pyvista as pv  # type: ignore

    #             pl = pv.Plotter(off_screen=True)
    #             _ = plot_3d_from_array(data_array=data5, plotter=pl)  # type: ignore
    #             pl.screenshot(str(fig_path))
    #             pl.close()

    #         logging.info("Saved figure %s", fig_path.name)

    logging.info("All done.")


if __name__ == "__main__":
    # main()


    shared_1 = os.path.dirname(os.path.abspath(__file__))

    # THEN create the full shared path
    name_extesion = "paa_w_50mcg_v_redone"

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

    for number_current in experiments:
        print("\nFor obtaining the baselines in the experiments for current "+number_current)
        main(number_current)



    '''Concatenate all baselines obtained from each current and save it '''
    baseline_ch_0_list = []


    for number_current in experiments:
        experiment_path = os.path.join(date_path, number_current)
        addres_name = 'information_baseline_ch0.txt'
        file_name = os.path.join(experiment_path, addres_name)
        signals = pd.read_csv(file_name,sep = '\s+').loc[:,["#", "X", "Y", "Z", "current","baseline"]]['baseline']
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
