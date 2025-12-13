import logging
import numpy as np
from pathlib import Path
from .attempt import Attempt
from ..plotting.plot_curves import plot_sample_curves
from ..plotting.plot_spectra import plot_sample_spectra
from ..utils.hyperparameters import (
    N_tau_default, 
    reg_par_default, 
    tau_min_default, 
    tau_max_default, 
    tau_sampling_default
)

class Sample:
    def __init__(self, sample_name, figures_path=None):
        self.sample_name = sample_name
        self.attempts = []
        self.figures_path = figures_path
        if self.figures_path:
            Path(self.figures_path).mkdir(exist_ok=True, parents=True)

    def compute_spectra(self, ch_num=None, use_parsed=False, fs=1.0, welch_params=None):
        """
        Compute Welch-based spectra for all Attempts in this Sample,
        but only for channel ch_num.
        """
        for attempt in self.attempts:
            attempt.compute_spectra(ch_num=ch_num, use_parsed=use_parsed, fs=fs, welch_params=welch_params)

    def subtract_baseline_from_all_attempts(self, method='median', **kwargs):
        channel_baselines = {}
        for attempt in self.attempts:
            attempt_baselines = attempt.subtract_baseline_from_all_channels(method=method, **kwargs)
            if attempt_baselines is None:
                continue
            for ch_num, baseline in attempt_baselines.items():
                if ch_num not in channel_baselines:
                    channel_baselines[ch_num] = []
                channel_baselines[ch_num].append(baseline)

        aggregated_baselines = {}
        for ch_num, baselines in channel_baselines.items():
            aggregated_baselines[ch_num] = np.median(baselines)
        return aggregated_baselines

    def get_baseline_stats(self):
        """
        Aggregate baselines from every attempt, then compute global
        statistics for each channel.
        """
        collected = {}  # channel_num -> list of baseline values
        for att in self.attempts:
            for ch in att.channels:
                if ch.baseline is None:
                    continue
                base = np.asarray(ch.baseline)
                if base.ndim > 1:
                    base = base.mean(axis=1)
                collected.setdefault(ch.channel_num, []).extend(base.tolist())

        stats = {
            ch_num: {
                "median": float(np.median(values)),
                "std":    float(np.std(values, ddof=1))
            }
            for ch_num, values in collected.items()
        }
        return stats

    def subtract_channels_in_all_attempts(self, channel_num_1, channel_num_2, new_channel_num=None):
        for attempt in self.attempts:
            attempt.subtract_channels_in_all_channels(channel_num_1, channel_num_2, new_channel_num)

    def denoise_all_attempts(self, method='spline', **kwargs):
        for attempt in self.attempts:
            attempt.denoise_all_channels(method=method, **kwargs)

    def plot_raw(self, attempt=None, channel=None, line_idx=-1, fig_dir=None, show=False):
        plot_sample_curves(self, attempt=attempt, channel=channel, line_idx=line_idx, fig_dir=fig_dir, show=show)

    def plot_pseudo_spectrum(self, plot_individual=True, plot_sparsified=True, show=False, ax=None, channel_num=None):
        plot_sample_spectra(self, plot_individual=plot_individual, plot_sparsified=plot_sparsified, show=show, ax=ax, channel_num=channel_num)

    def perform_multiexponential_analysis_all_attempts(
        self,
        method='l2',
        channel_num=None,
        fig_dir=None,
        line_idx=0,
        **kwargs
    ):
        """
        Unified method to perform multi-exponential analysis on all attempts.
        
        Parameters
        ----------
        method : str
            Analysis method ('l2', 'elastic', 'fixed_taus', 'fixed_taus_baseline')
        channel_num : int, optional
            If specified, only analyze this channel
        fig_dir : str, optional
            Directory for saving figures
        line_idx : int
            Line index for plotting
        **kwargs : dict
            Method-specific parameters
        """
        for attempt in self.attempts:
            attempt.perform_me_analysis_all_channels(
                method=method,
                channel_num=channel_num,
                fig_dir=fig_dir,
                line_idx=line_idx,
                **kwargs
            )
    
    # Backward compatibility wrappers
    def perform_multiexponential_analysis_all_attempts_l2(self, **kwargs):
        """L2 analysis (backward compatibility)."""
        return self.perform_multiexponential_analysis_all_attempts(method='l2', **kwargs)
    
    def perform_multiexponential_analysis_all_attempts_elastic(self, **kwargs):
        """Elastic Net analysis (backward compatibility)."""
        return self.perform_multiexponential_analysis_all_attempts(method='elastic', **kwargs)
    
    def perform_multiexponential_analysis_all_attempts_fixed_taus(self, **kwargs):
        """Fixed tau analysis (backward compatibility)."""
        fit_baseline = kwargs.get('fit_baseline', False)
        method = 'fixed_taus_baseline' if fit_baseline else 'fixed_taus'
        return self.perform_multiexponential_analysis_all_attempts(method=method, **kwargs)

    def extract_and_analyze_pronounced_peaks(self, N_peaks=3, statistic='mean'):
        peaks_list = []
        for attempt in self.attempts:
            for channel in attempt.channels:
                if not hasattr(channel, 'pronounced_peaks') or channel.pronounced_peaks is None:
                    result = channel.extract_pronounced_peaks(N_peaks=N_peaks)
                    if result is not None and result.ndim == 3 and result.shape[-1] == 2:
                        peaks_list.append(result)
                else:
                    if channel.pronounced_peaks is not None and channel.pronounced_peaks.ndim == 3 and channel.pronounced_peaks.shape[-1] == 2:
                        peaks_list.append(channel.pronounced_peaks)

        if len(peaks_list) == 0:
            logging.warning(f"No pronounced peaks found for Sample '{self.sample_name}'.")
            self.mean_peaks = None
            self.std_peaks = None
            return

        peaks_array = np.concatenate(peaks_list, axis=0)
        if statistic == 'mean':
            mean_peaks = np.mean(peaks_array, axis=0)
            std_peaks = np.std(peaks_array, axis=0)
        elif statistic == 'median':
            mean_peaks = np.median(peaks_array, axis=0)
            std_peaks = np.std(peaks_array, axis=0)
        else:
            logging.warning(f"Unknown statistic '{statistic}'. Using mean.")
            mean_peaks = np.mean(peaks_array, axis=0)
            std_peaks = np.std(peaks_array, axis=0)

        self.mean_peaks = mean_peaks
        self.std_peaks = std_peaks
        logging.info(f"Pronounced peaks analyzed for Sample '{self.sample_name}' using {statistic}.")
