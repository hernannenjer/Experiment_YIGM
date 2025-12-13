import logging
import numpy as np
import matplotlib.pyplot as plt
from random import randint

def plot_channel_curves(channel, line_idx=0, label='', show=False, ax=None, highlight=False, use_baseline_subtracted=None, fig_dir=None, **plot_kwargs):
    if use_baseline_subtracted is None:
        data = channel.data_baseline_subtracted if channel.data_fit is not None else channel.raw_data
    elif use_baseline_subtracted:
        data = channel.data_baseline_subtracted
    else:
        data = channel.raw_data

    if data is None:
        logging.warning("No data available for plotting.")
        return None
    if channel.time is None:
        logging.warning("Time axis not set.")
        return None
    if not (0 <= line_idx < data.shape[0]):
        logging.warning(f"Invalid line index {line_idx}.")
        return None

    label = label or f'Channel {channel.channel_num}'
    linewidth = 2.5 if highlight else 1.0

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
        own_figure = True
    else:
        fig = ax.figure
        own_figure = False

    ax.plot(channel.time, data[line_idx], label=f'{label} Raw', linewidth=linewidth, **plot_kwargs)

    # Perform sparse refit if needed
    if channel.sparsified_amplitudes is not None and channel.sparsified_relaxation_times is not None and channel.data_fit_sparse is None:
        channel.perform_sparse_refit()

    # Plot fits if available
    if channel.data_fit is not None and line_idx < channel.data_fit.shape[0]:
        ax.plot(channel.time, channel.data_fit[line_idx], label=f'{label} Fit', linewidth=2.5, **plot_kwargs)
    if channel.data_fit_sparse is not None and line_idx < channel.data_fit_sparse.shape[0]:
        ax.plot(channel.time, channel.data_fit_sparse[line_idx], label=f'{label} Sparse Fit', linewidth=2.5, **plot_kwargs)

    if own_figure:
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Signal')
        ax.grid(True)
        ax.legend()
        if fig_dir:
            from pathlib import Path
            fig_dir_path = Path(fig_dir)
            fig_dir_path.mkdir(parents=True, exist_ok=True)
            fig_filename = fig_dir_path / f'channel_{channel.channel_num}_attempt_{channel.attempt_num}.png'
            fig.savefig(fig_filename)
            logging.info(f"Plot saved to {fig_filename}")
        if show:
            plt.show()
        plt.close(fig)

    return fig

def plot_sample_curves(sample, attempt=None, channel=None, line_idx=-1, fig_dir=None, show=False):
    # We reuse plot_channel_curves for each channel in the sample
    if attempt is not None:
        attempts_to_plot = [a for a in sample.attempts if a.attempt_num == attempt]
        if not attempts_to_plot:
            logging.warning(f"Attempt {attempt} not found in Sample {sample.sample_name}.")
            return
    else:
        attempts_to_plot = sample.attempts

    for attempt_obj in attempts_to_plot:
        if channel is not None:
            channels_to_plot = [ch for ch in attempt_obj.channels if ch.channel_num == channel]
            if not channels_to_plot:
                logging.warning(f"Channel {channel} not found in Attempt {attempt_obj.attempt_num}.")
                continue
        else:
            channels_to_plot = attempt_obj.channels

        if not channels_to_plot:
            logging.warning(f"No channels to plot in Attempt {attempt_obj.attempt_num}.")
            continue

        min_lines = None
        for ch in channels_to_plot:
            data = ch.data_baseline_subtracted if ch.data_type == 'gradientometer' else ch.raw_data
            if data is not None:
                num_lines = data.shape[0]
                if min_lines is None or num_lines < min_lines:
                    min_lines = num_lines
        if min_lines is None or min_lines == 0:
            logging.warning(f"No data available to plot in Attempt {attempt_obj.attempt_num}.")
            continue

        if line_idx == -1:
            line_idx = np.random.randint(0, min_lines)
            logging.info(f"Random line index {line_idx} selected.")
        else:
            if line_idx < 0 or line_idx >= min_lines:
                logging.warning(f"Invalid line index {line_idx}.")
                continue

        fig, ax = plt.subplots(figsize=(10, 6))

        for ch in channels_to_plot:
            ch_label = f'{attempt_obj.sample_name} - Channel {ch.channel_num}'
            highlight = (ch.data_type == 'gradientometer')
            plot_channel_curves(ch, line_idx=line_idx, label=ch_label, ax=ax, highlight=highlight, show=False)

        ax.set_title(f'Sample {sample.sample_name} - Attempt {attempt_obj.attempt_num}')
        ax.legend()
        ax.grid(True)
        plt.tight_layout()

        if fig_dir is not None:
            from pathlib import Path
            fig_dir_path = Path(fig_dir)
            fig_dir_path.mkdir(parents=True, exist_ok=True)
            plot_filename = f'sample_{sample.sample_name}_attempt_{attempt_obj.attempt_num}_combined_plot.png'
            fig.savefig(fig_dir_path / plot_filename)
            logging.info(f"Combined plot saved to {fig_dir_path / plot_filename}")

        if show:
            plt.show()

        plt.close(fig)

def plot_experiment_curves(experiment, sample=None, attempt=None, channel=None, 
                           fit_data=None, line_idx=-1, fig_dir=None, show=False):
    if sample is not None:
        sample_obj = experiment.samples.get(sample)
        if sample_obj is None:
            logging.warning(f"Sample '{sample}' not found.")
            return
        samples_to_plot = [sample_obj]
    else:
        samples_to_plot = experiment.samples.values()

    for sample_obj in samples_to_plot:
        plot_sample_curves(sample_obj, attempt=attempt, channel=channel, line_idx=line_idx, 
                           fig_dir=fig_dir, show=show)
