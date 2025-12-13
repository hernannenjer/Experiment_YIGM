import os
import re
import logging
import numpy as np
import joblib

from .sample import Sample
from .attempt import Attempt
from ..plotting.plot_curves import plot_experiment_curves
from ..plotting.plot_spectra import plot_experiment_spectra
from ..plotting.plot_deps_vs_sample import (
    plot_peak_amplitudes_vs_sample, 
    plot_relaxation_times_vs_sample, 
    plot_baseline_vs_sample, 
    plot_integrals_vs_sample
)
from ..plotting.plot_3d import plot_3d_peak_heights
from ..utils.hyperparameters import (
    N_tau_default, 
    reg_par_default, 
    tau_min_default, 
    tau_max_default, 
    tau_sampling_default
)

class Experiment:
    def __init__(self, experiment_dir, figures_base_path=None):
        self.experiment_dir = experiment_dir
        self.samples = {}
        self.figures_base_path = figures_base_path

    def load_data(self, idx, fs, parse=True, sample_list=None):
        """
        Loads data from the experiment folder. Each subfolder is a sample, possibly with an attempt index.
        """
        dir_pattern = re.compile(r'^(.+?)(?:_(\d+))?$')
        for dir_name in os.listdir(self.experiment_dir):
            dir_path = os.path.join(self.experiment_dir, dir_name)
            if os.path.isdir(dir_path):
                match = dir_pattern.match(dir_name)
                if match:
                    sample_name, attempt_num_str = match.groups()
                    if sample_list is not None and sample_name not in sample_list:
                        continue
                    try:
                        attempt_num = int(attempt_num_str) if attempt_num_str else 1
                    except ValueError:
                        logging.warning(f"Cannot parse attempt number from '{dir_name}'")
                        continue
                    sample = self.samples.get(sample_name)
                    if sample is None:
                        sample_figures_path = (
                            os.path.join(self.figures_base_path, sample_name) 
                            if self.figures_base_path else None
                        )
                        sample = Sample(sample_name, figures_path=sample_figures_path)
                        self.samples[sample_name] = sample
                    attempt = Attempt(dir_path, sample_name, attempt_num)
                    attempt.load(idx=idx, fs=fs, parse=parse)
                    sample.attempts.append(attempt)
                    logging.info(f"Loaded attempt {attempt_num} for sample '{sample_name}'")

    def compute_spectra(self, ch_num=None, use_parsed=False, fs=1.0, welch_params=None):
        """
        Compute Welch-based spectra for all Samples in this Experiment,
        but only for the given channel number ch_num.
        """
        for sample in self.samples.values():
            sample.compute_spectra(ch_num=ch_num, use_parsed=use_parsed, fs=fs, welch_params=welch_params)

    def save(self, filename):
        try:
            joblib.dump(self, filename)
            logging.info(f"Experiment saved to {filename}")
        except Exception as e:
            logging.error(f"Failed to save Experiment: {e}")

    @classmethod
    def load(cls, filename):
        try:
            experiment = joblib.load(filename)
            logging.info(f"Experiment loaded from {filename}")
            return experiment
        except Exception as e:
            logging.error(f"Failed to load Experiment: {e}")
            return None

    def subtract_baseline_from_all_samples(self, method='median', **kwargs):
        if method == 'zero_sample':
            return self._subtract_baseline_zero_sample(**kwargs)
        else:
            return self._subtract_baseline_standard(method=method, **kwargs)

    def _subtract_baseline_standard(self, method='median', **kwargs):
        baselines = {}
        for sample in self.samples.values():
            sample_baselines = sample.subtract_baseline_from_all_attempts(method=method, **kwargs)
            baselines.update(sample_baselines)
        return baselines

    def _subtract_baseline_zero_sample(self, n_points=500):
        if '0' not in self.samples:
            logging.warning("Sample '0' not found. Cannot compute tail baseline.")
            return None
        sample_zero = self.samples['0']
        channel_baselines = {}
        channel_numbers = set()
        for attempt in sample_zero.attempts:
            for ch in attempt.channels:
                channel_numbers.add(ch.channel_num)

        if not channel_numbers:
            logging.warning("Sample '0' has no channels.")
            return None

        for channel_num in channel_numbers:
            tail_means = []
            for attempt in sample_zero.attempts:
                ch_zero = next((c for c in attempt.channels if c.channel_num == channel_num), None)
                if ch_zero is None:
                    logging.warning(f"Sample '0' - Attempt {attempt.attempt_num}: Channel {channel_num} not found.")
                    continue
                if ch_zero.raw_data is not None and ch_zero.raw_data.shape[1] >= n_points:
                    tail_segment = ch_zero.raw_data[:, -n_points:]
                    curve_tail_means = np.mean(tail_segment, axis=1)
                    tail_means.extend(curve_tail_means)
                else:
                    logging.warning(f"Sample '0' - Attempt {attempt.attempt_num}: Channel {channel_num} not enough data.")

            if not tail_means:
                logging.warning(f"Sample '0' - Channel {channel_num}: No tail means found.")
                continue
            tail_means = np.array(tail_means)
            median_baseline = np.median(tail_means)
            channel_baselines[channel_num] = median_baseline

            mean_val = np.mean(tail_means)
            median_val = np.median(tail_means)
            std_val = np.std(tail_means)
            logging.info(f"Channel {channel_num} Baseline Stats (Sample '0'):")
            logging.info(f"  Mean: {mean_val:.6f}, Median: {median_val:.6f}, Std: {std_val:.6f}")
            logging.info(f"  Computed Median Baseline: {median_baseline:.6f}")

        if not channel_baselines:
            logging.warning("No channel baselines computed.")
            return None

        for sample_name, sample in self.samples.items():
            for attempt in sample.attempts:
                for ch in attempt.channels:
                    if ch.channel_num in channel_baselines:
                        baseline = channel_baselines[ch.channel_num]
                        ch.subtract_baseline_scalar(baseline)
                    else:
                        logging.warning(f"Sample '{sample_name}' - Channel {ch.channel_num}: No baseline found.")

        logging.info("Baseline subtraction using 'zero_sample' method completed.")
        return channel_baselines

    def get_baseline_stats(self):
        """
        Return nested dict: stats[sample_name][channel_num]['median'|'std']
        """
        return {
            sample_name: sample.get_baseline_stats()
            for sample_name, sample in self.samples.items()
        }

    def subtract_channels_in_all_samples(self, channel_num_1, channel_num_2, new_channel_num=None):
        for sample in self.samples.values():
            sample.subtract_channels_in_all_attempts(channel_num_1, channel_num_2, new_channel_num)

    def denoise_all_samples(self, method='spline', **kwargs):
        for sample in self.samples.values():
            sample.denoise_all_attempts(method=method, **kwargs)

    def perform_multiexponential_analysis_all_samples(
        self,
        method='l2',
        N_tau=N_tau_default,
        alpha=reg_par_default,
        tau_min=tau_min_default,
        tau_max=tau_max_default,
        tau_sampling=tau_sampling_default,
        approx_taus=None,
        channel_num=None,
        fig_dir=None,
        line_idx=0,
        **kwargs  # Дополнительные параметры (weight, fit_baseline и т.д.)
    ):
        """
        Выполняет мультиэкспоненциальный анализ для всех образцов.
        
        Параметры
        ---------
        method : str
            Метод анализа ('l2', 'elastic', 'fixed_taus', 'fixed_taus_baseline')
        **kwargs : dict
            Дополнительные параметры для конкретных методов:
            - weight : float (для elastic) - вес между L1 и L2
            - fit_baseline : bool (для fixed_taus)
        """
        # Формируем общий набор параметров
        params = {
            'channel_num': channel_num,
            'N_tau': N_tau,
            'alpha': alpha,
            'tau_min': tau_min,
            'tau_max': tau_max,
            'tau_sampling': tau_sampling,
            'fig_dir': fig_dir,
            'line_idx': line_idx,
            **kwargs  # Добавляем дополнительные параметры
        }
        
        if method == 'l2':
            for sample in self.samples.values():
                sample.perform_multiexponential_analysis_all_attempts(
                    method='l2', **params
                )
        elif method == 'elastic':
            for sample in self.samples.values():
                sample.perform_multiexponential_analysis_all_attempts(
                    method='elastic', **params
                )
        elif method == 'fixed_taus' or method == 'fixed_taus_baseline':
            if not approx_taus:
                logging.warning(f"approx_taus must be specified for '{method}'.")
                return
            params['approx_taus'] = approx_taus
            if method == 'fixed_taus_baseline':
                params['fit_baseline'] = True
            for sample in self.samples.values():
                sample.perform_multiexponential_analysis_all_attempts(
                    method=method, **params
                )
        else:
            logging.warning(f"Unknown ME method '{method}' requested.")

        logging.info(f"Done ME analysis with method='{method}'.")

    def analyze_pronounced_peaks_per_sample(self, N_peaks=3, statistic='mean'):
        for sample in self.samples.values():
            sample.extract_and_analyze_pronounced_peaks(N_peaks=N_peaks, statistic=statistic)
        logging.info("Pronounced peaks analyzed for all samples.")

    def plot_raw(self, sample=None, attempt=None, channel=None, fit_data=None, line_idx=-1, fig_dir=None, show=False):
        plot_experiment_curves(
            self, sample=sample, attempt=attempt, channel=channel,
            fit_data=fit_data, line_idx=line_idx, fig_dir=fig_dir, show=show
        )

    def plot_pseudo_spectrum(self, plot_individual=True, plot_sparsified=True, show=False, ax=None, channel_num=None):
        plot_experiment_spectra(
            self, 
            plot_individual=plot_individual, 
            plot_sparsified=plot_sparsified, 
            show=show, 
            ax=ax, 
            channel_num=channel_num
        )

    def plot_peak_amplitudes_vs_sample(self, peak_idx=0, sample_properties=None, log_axes=True,
                                       exclude_zero_sample=False, use_sparsified=False,
                                       show_error=True, connect_points=True, x_label='mg'):
        plot_peak_amplitudes_vs_sample(
            self, 
            peak_idx, 
            sample_properties, 
            log_axes, 
            exclude_zero_sample, 
            use_sparsified, 
            show_error, 
            connect_points=connect_points, 
            x_label=x_label
        )

    def plot_relaxation_times_vs_sample(self, log_axes=True):
        plot_relaxation_times_vs_sample(self, log_axes)

    def plot_baseline_vs_sample(self, log_axes=True, ch_num=None, x_label=r'$\mu$g'):
        plot_baseline_vs_sample(self, log_axes, ch_num, x_label=x_label)

    def plot_integrals_vs_sample(self, log_axes=True):
        plot_integrals_vs_sample(self, log_axes)

    def plot_3d_peak_heights(self, coords_filename, N_peaks=1, point_size=10, cmap='viridis', show=True, rec=None, gt=None):
        return plot_3d_peak_heights(self, coords_filename, N_peaks, point_size, cmap, show, rec, gt)

    def get_SLAE_vectors(self, context_file, peak_index):
        # same logic as previously provided
        pass

    def print(self):
        print("\n=== Experiment Information ===\n")
        if not self.samples:
            print("No samples found.")
            logging.warning("No samples found in the experiment.")
            return

        for sample_name, sample in self.samples.items():
            num_attempts = len(sample.attempts)
            print(f"Sample '{sample_name}':")
            print(f"  Number of Attempts: {num_attempts}")
            logging.info(f"Sample '{sample_name}': {num_attempts} attempts.")

            if num_attempts == 0:
                print("    No attempts found for this sample.")
                logging.warning(f"Sample '{sample_name}' has no attempts.")
                continue

            for attempt in sample.attempts:
                attempt_num = attempt.attempt_num
                num_channels = len(attempt.channels)
                print(f"  Attempt {attempt_num}:")
                print(f"    Number of Channels: {num_channels}")
                logging.info(f"Attempt {attempt_num}: {num_channels} channels.")

                if num_channels == 0:
                    print("      No channels found for this attempt.")
                    logging.warning(f"Attempt {attempt_num} has no channels.")
                    continue

                for channel in attempt.channels:
                    channel_num = channel.channel_num
                    if channel.raw_data is not None:
                        if isinstance(channel.raw_data, np.ndarray) and channel.raw_data.ndim == 2:
                            num_curves, num_points = channel.raw_data.shape
                            print(f"    Channel {channel_num}:")
                            print(f"      Number of Curves: {num_curves}")
                            print(f"      Number of Points per Curve: {num_points}")
                            logging.info(f"Channel {channel_num}: {num_curves} curves, {num_points} points.")
                        else:
                            print(f"    Channel {channel_num}: raw_data unexpected shape/dimensions.")
                            logging.warning(f"Channel {channel_num}: raw_data unexpected dimensions.")
                    else:
                        print(f"    Channel {channel_num}: No raw_data loaded.")
                        logging.warning(f"Channel {channel_num}: No raw_data loaded.")

            print("")  
        print("=== End of Experiment Information ===\n")
        logging.info("Completed printing Experiment Information.")
