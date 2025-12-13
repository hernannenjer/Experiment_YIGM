import logging
import numpy as np
import matplotlib.pyplot as plt

def plot_channel_spectrum(channel, plot_individual=True, plot_sparsified=True, ax=None, show=False, set_labels=True):
    if channel.relaxation_times is None or channel.amplitudes is None:
        logging.warning("No relaxation times or amplitudes available for plotting.")
        return None

    if ax is None:
        fig, ax = plt.subplots(figsize=(10,6))
    else:
        fig = None

    if plot_individual:
        for i in range(channel.amplitudes.shape[0]):
            ax.plot(channel.relaxation_times, channel.amplitudes[i, :], color='gray', linewidth=0.5, alpha=0.5)

    if plot_sparsified and channel.sparsified_amplitudes is not None and channel.sparsified_relaxation_times is not None:
        mean_sparse = np.mean(channel.sparsified_amplitudes, axis=0) if channel.sparsified_amplitudes.ndim > 1 else channel.sparsified_amplitudes
        ax.plot(channel.sparsified_relaxation_times, mean_sparse, label=f'sample: {channel.sample_name}; channel {channel.channel_num}', linewidth=3)

    if set_labels:
        ax.set_xlabel('Relaxation Time')
        ax.set_ylabel('Amplitude')
        ax.set_title(f'Pseudo-Spectrum for Channel {channel.channel_num}')
        ax.legend()

    if fig is not None and show:
        plt.show()

    return ax

def plot_sample_spectra(sample, plot_individual=True, plot_sparsified=True, show=False, ax=None, channel_num=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
        created_fig = True
    else:
        created_fig = False

    for attempt in sample.attempts:
        for channel in attempt.channels:
            if channel_num is not None and channel.channel_num != channel_num:
                continue
            plot_channel_spectrum(channel, plot_individual=plot_individual, plot_sparsified=plot_sparsified, ax=ax, set_labels=False)

    if created_fig:
        ax.set_xlabel('Relaxation Time')
        ax.set_ylabel('Amplitude')
        ax.set_title(f'Pseudo-Spectra for Sample {sample.sample_name}')
        ax.legend()
        if show:
            plt.show()

def plot_experiment_spectra(experiment, plot_individual=True, plot_sparsified=True, show=False, ax=None, channel_num=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
        created_figure = True
    else:
        created_figure = False

    for sample in experiment.samples.values():
        plot_sample_spectra(sample, plot_individual=plot_individual, plot_sparsified=plot_sparsified, show=False, ax=ax, channel_num=channel_num)

    if created_figure:
        ax.set_xlabel('Relaxation Time')
        ax.set_ylabel('Amplitude')
        ax.set_title('Pseudo-Spectra for All Samples')
        ax.legend()
        if show:
            plt.show()
