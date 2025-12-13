import logging
import numpy as np
from pathlib import Path
from scipy.signal import welch

from ..analysis.me_analysis import (
    perform_l2_analysis,
    perform_elastic_analysis,
    create_custom_basis,
    select_peaks, 
    sparcify_spectrum, 
    perform_three_exp_analysis
)
from ..analysis.curve_preprocessing import (
    subtract_baseline_mean_end,
    subtract_baseline_median,
    subtract_baseline_polynomial,
    subtract_baseline_lowpass,
    subtract_baseline_3points,
    subtract_baseline_scalar,
    denoise_spline_interpolation
)
from ..io.loader import load_raw_data
from ..io.parser import parse_channel_data
from ..utils.hyperparameters import (
    N_tau_default, 
    reg_par_default, 
    tau_min_default, 
    tau_max_default, 
    tau_sampling_default
)


class Channel:
    def __init__(self, channel_num, attempt_num, data_type='magnetometer', sample_name=None):
        self.channel_num = channel_num
        self.attempt_num = attempt_num
        self.data_type = data_type
        self.sample_name = sample_name

        # Store unparsed and parsed data
        self.unparsed_channel_data = None  # "long" signal or entire loaded data
        self.raw_data = None              # e.g., "lrels" or other parsed subset

        self.time_unparsed = None
        self.time = None

        # Fit data / results
        self.data_fit = None
        self.data_baseline_subtracted = None
        self.data_denoised = None

        # ME analysis results
        self.relaxation_times = None
        self.amplitudes = None
        self.sparsified_amplitudes = None
        self.sparsified_relaxation_times = None
        self.data_fit_sparse = None
        self.basis_norms = None
        self.custom_basis_norms = None
        self.baseline = None

        # Spectrum fields
        self.freqs_spectra = None
        self.psd_spectra = None  # e.g. PSD from Welch

    def get_baseline_stats(self):
        """
        Return median and standard deviation of the baseline vector.
        If baseline is a 2-D array (e.g. low-pass baseline), first
        reduce it to a 1-D vector by averaging along the time axis.
        """
        if self.baseline is None:
            return None

        base = np.asarray(self.baseline)
        if base.ndim > 1:                       # shape (N_curves, N_points)
            base = base.mean(axis=1)            # one value per curve

        return {
            "median": float(np.median(base)),
            "std":    float(np.std(base, ddof=1))
        }

    def compute_spectra(self, ch_num=None, use_parsed=False, fs=1.0, welch_params=None):
        """
        Compute the Welch spectrum for either:
         - unparsed_channel_data (use_parsed=False), or
         - raw_data (use_parsed=True).

        If ch_num is None, always compute spectra for this channel.
        If ch_num is not None, only compute if self.channel_num == ch_num.

        :param ch_num: Channel number to process or None (if None, this channel is always processed).
        :param use_parsed: If True, use self.raw_data; otherwise use self.unparsed_channel_data
        :param fs: Sampling frequency
        :param welch_params: Dictionary of parameters for scipy.signal.welch
        :return: (freqs, psd) or (None, None) if skipped
        """
        # If ch_num is specified and doesn't match this channel, skip.
        if ch_num is not None and self.channel_num != ch_num:
            return None, None

        if welch_params is None:
            welch_params = {}

        data_source = self.raw_data if use_parsed else self.unparsed_channel_data
        if data_source is None:
            logging.warning(f"[Channel {self.channel_num}] No data to compute spectra (use_parsed={use_parsed}).")
            return None, None

        # Ensure 2D: shape=(Ncurves, Npoints)
        data_2d = np.atleast_2d(data_source)

        from scipy.signal import welch
        psd_list = []
        freqs = None
        for row in data_2d:
            f, Pxx = welch(row, fs=fs, **welch_params)
            if freqs is None:
                freqs = f
            psd_list.append(Pxx)

        psd_array = np.array(psd_list)  # shape=(Ncurves, len(freqs))
        mean_psd = np.mean(psd_array, axis=0)

        self.freqs_spectra = freqs
        self.psd_spectra = mean_psd

        logging.info(f"[Channel {self.channel_num}] Spectrum computed (use_parsed={use_parsed}).")
        return freqs, mean_psd

    def perform_multiexponential_analysis(
        self,
        method='l2',
        fig_dir=None,
        line_idx=0,
        **kwargs
    ):
        """
        Unified method for multi-exponential analysis using Strategy pattern.
        
        Parameters
        ----------
        method : str
            Analysis method ('l2', 'elastic', 'fixed_taus', 'fixed_taus_baseline')
        fig_dir : str, optional
            Directory for saving figures
        line_idx : int
            Line index for plotting
        **kwargs : dict
            Method-specific parameters (N_tau, alpha, tau_min, tau_max, 
            tau_sampling, approx_taus, fit_baseline, etc.)
            
        Returns
        -------
        tuple
            (relaxation_times, amplitudes)
        """
        if self.time is None or self.data_baseline_subtracted is None:
            logging.warning(f"[Channel {self.channel_num}] Not enough data for analysis.")
            return None, None
        
        from ..analysis.analysis_strategy import perform_analysis
        
        try:
            tau, amplitudes, data_fit, norms = perform_analysis(
                method=method,
                time=self.time,
                data=self.data_baseline_subtracted,
                **kwargs
            )
            
            # Store results
            self.relaxation_times = tau
            self.amplitudes = amplitudes
            self.data_fit = data_fit
            self.basis_norms = norms
            
            # Handle baseline for fixed_taus_baseline method
            if method == 'fixed_taus_baseline' and 'fit_baseline' in kwargs and kwargs['fit_baseline']:
                # For this method, amplitudes might contain baseline as first column
                # But perform_analysis already handles this, so amplitudes should be clean
                pass
            
            logging.info(f"[Channel {self.channel_num}] {method} analysis completed.")
            return self.relaxation_times, self.amplitudes
            
        except Exception as e:
            logging.error(f"[Channel {self.channel_num}] Analysis failed: {e}")
            return None, None
    
    # Backward compatibility methods (delegate to unified method)
    def perform_multiexponential_analysis_l2(self, **kwargs):
        """L2 analysis (backward compatibility wrapper)."""
        return self.perform_multiexponential_analysis(method='l2', **kwargs)
    
    def perform_multiexponential_analysis_elastic(self, **kwargs):
        """Elastic Net analysis (backward compatibility wrapper)."""
        return self.perform_multiexponential_analysis(method='elastic', **kwargs)
    
    def perform_multiexponential_analysis_fixed_taus(self, **kwargs):
        """Fixed tau analysis (backward compatibility wrapper)."""
        fit_baseline = kwargs.get('fit_baseline', False)
        method = 'fixed_taus_baseline' if fit_baseline else 'fixed_taus'
        return self.perform_multiexponential_analysis(method=method, **kwargs)

    def perform_sparse_refit(self, tau_candidates=None, min_peak_height=1e-7, tau_sampling='non_uniform'):
        if self.data_baseline_subtracted is None or self.time is None:
            logging.warning("Baseline-subtracted data or time not available.")
            return None, None

        if tau_candidates is None:
            tau_candidates = select_peaks(self.relaxation_times, self.amplitudes, min_peak_height=min_peak_height)
            if tau_candidates is None:
                logging.warning("No tau candidates selected.")
                return None, None

        tau_candidates = np.sort(tau_candidates)
        Phi_reduced, norms = create_custom_basis(self.time, tau_candidates)
        self.custom_basis_norms = norms

        Phi_reduced_t = Phi_reduced.T
        N_curves = self.data_baseline_subtracted.shape[0]
        N_basis = tau_candidates.shape[0]
        sparse_amplitudes = np.zeros((N_curves, N_basis))

        for i in range(N_curves):
            data_i = self.data_baseline_subtracted[i]
            M_i, residuals, rank, s = np.linalg.lstsq(Phi_reduced, data_i, rcond=None)
            sparse_amplitudes[i] = M_i

        self.sparsified_amplitudes = sparse_amplitudes
        self.sparsified_relaxation_times = tau_candidates
        self.data_fit_sparse = sparse_amplitudes @ Phi_reduced_t

        logging.info("Sparse refit completed using selected candidate tau values.")
        return self.sparsified_relaxation_times, self.sparsified_amplitudes

    def extract_pronounced_peaks(self, N_peaks=3):
        if self.amplitudes is None or self.relaxation_times is None:
            logging.warning("Amplitudes or relaxation times not available. Perform ME analysis first.")
            self.pronounced_peaks = None
            return None

        N_curves, N_basis = self.amplitudes.shape
        peaks_array = np.zeros((N_curves, N_peaks, 2))
        sparse_amplitudes = np.zeros_like(self.amplitudes)

        for i in range(N_curves):
            amplitudes_i = self.amplitudes[i, :]
            peak_indices = np.argsort(-np.abs(amplitudes_i))[:N_peaks]
            peaks_array[i, :, 0] = amplitudes_i[peak_indices]
            peaks_array[i, :, 1] = self.relaxation_times[peak_indices]
            sparse_amplitudes[i, peak_indices] = amplitudes_i[peak_indices]

        self.pronounced_peaks = peaks_array
        self.sparse_amplitudes = sparse_amplitudes
        logging.info(f"Extracted {N_peaks} most pronounced peaks for Channel {self.channel_num}.")
        return peaks_array

    def load(self, filename, idx, fs, parse=True, show=False):
        try:
            self.unparsed_channel_data, time_unparsed = load_raw_data(filename, fs)

#             logging.info(f"Size of unparsed data = {self.unparsed_channel_data.shape}")
            self.time_unparsed = time_unparsed
            self.time = (np.arange(idx[5]) + idx[4]) / fs

            if parse:
                parse_result = parse_channel_data(self.unparsed_channel_data, idx, filename, show=show)
                if parse_result is not None:
                    self.raw_data = -(parse_result[2] - parse_result[3]) / 2
                else:
                    logging.warning("Parsing failed.")
                    return False
            else:
                parsed_data_filename = self.get_parsed_filename(filename)
                if Path(parsed_data_filename).exists():
                    self.raw_data = np.load(parsed_data_filename, allow_pickle=True)
                    logging.info(f"lrels data loaded from {parsed_data_filename}")
                else:
                    logging.info("Parsed data not found, parsing now...")
                    parse_result = parse_channel_data(self.unparsed_channel_data, idx, filename)
                    if parse_result is not None:
                        self.raw_data = parse_result[2]
                    else:
                        logging.warning("Parsing failed.")
                        return False

            return (self.raw_data is not None)
        except Exception as e:
            logging.error(f"Error loading channel from {filename}: {e}")
            return False

    def get_parsed_filename(self, filename):
        return filename.replace('.txt', '_parsed.npy')

    def parse(self, data, idx, filename, fig_path=None, show=False, result_path=None):
        return parse_channel_data(data, idx, filename, fig_path, show, result_path)

    def save(self, filename):
        result_dir = Path(filename).parent / 'results'
        result_dir.mkdir(parents=True, exist_ok=True)
        parsed_data_filename = self.get_parsed_filename(filename)
        np.save(parsed_data_filename, self.raw_data)
        logging.info(f"lrels data saved to {parsed_data_filename}")

    def subtract_baseline(self, method='median', **kwargs):
        if self.raw_data is None:
            logging.warning("Raw data not loaded. Cannot subtract baseline.")
            return None
        if method == 'mean_end':
            result = subtract_baseline_mean_end(self.raw_data, **kwargs)
        elif method == 'median':
            result = subtract_baseline_median(self.raw_data, **kwargs)
        elif method == 'polynomial':
            if self.time is None:
                logging.warning("No time axis available for polynomial baseline subtraction.")
                return None
            result = subtract_baseline_polynomial(self.raw_data, self.time, **kwargs)
        elif method == 'lowpass':
            result = subtract_baseline_lowpass(self.raw_data, **kwargs)
        elif method == '3points':
            result = subtract_baseline_3points(self.raw_data, **kwargs)
        elif method == 'scalar':
            if 'scalar' not in kwargs:
                logging.warning("For scalar baseline, provide 'scalar' argument.")
                return None
            scalar_val = kwargs['scalar']
            result = subtract_baseline_scalar(self.raw_data, scalar_val)
        elif method == 'None':
            result = (self.raw_data, np.zeros(self.raw_data.shape[0]))
        else:
            logging.warning(f"Unknown baseline subtraction method: {method}")
            return None

        if result is not None:
            self.data_baseline_subtracted, self.baseline = result
        return result

    def subtract_baseline_scalar(self, scalar):
        result = subtract_baseline_scalar(self.raw_data, scalar)
        if result is not None:
            self.data_baseline_subtracted, _ = result
        return result

    def denoise_spline_interpolation(self, s=None):
        result = denoise_spline_interpolation(self.data_baseline_subtracted, self.time, s)
        if result is not None:
            self.data_denoised = result
        return result

    def print(self):
        logging.info(f"Channel Number: {self.channel_num}")
        logging.info(f"Attempt Number: {self.attempt_num}")
        logging.info(f"Data Type: {self.data_type}")

        if self.unparsed_channel_data is not None:
            logging.info(f"Unparsed Data Shape: {self.unparsed_channel_data.shape}")
        else:
            logging.info("Unparsed Data: None")

        if self.raw_data is not None:
            logging.info(f"Parsed Data Shape (e.g. lrels): {self.raw_data.shape}")
        else:
            logging.info("Parsed Data: None")

        if self.time is not None:
            logging.info(f"Time Axis Length: {self.time.shape[0]}")
        else:
            logging.info("Time Axis: None")
