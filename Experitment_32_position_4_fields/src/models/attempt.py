import os
import logging
from .channel import Channel
from ..utils.hyperparameters import (
    N_tau_default, 
    reg_par_default, 
    tau_min_default, 
    tau_max_default, 
    tau_sampling_default
)

class Attempt:
    def __init__(self, attempt_dir, sample_name, attempt_num):
        self.attempt_dir = attempt_dir
        self.sample_name = sample_name
        self.attempt_num = attempt_num
        self.channels = []

    def load(self, idx, fs, parse=True):
        for filename in os.listdir(self.attempt_dir):
            file_path = os.path.join(self.attempt_dir, filename)
            if os.path.isfile(file_path):
                if filename.endswith(".txt"):
                    channel_name = os.path.splitext(filename)[0]
                    try:
                        channel_num = int(channel_name)
                    except ValueError:
                        logging.warning(f"Cannot parse channel number from '{filename}'")
                        continue
                    channel = Channel(
                        channel_num=channel_num,
                        attempt_num=self.attempt_num,
                        data_type='magnetometer',
                        sample_name=self.sample_name
                    )
                    if channel.load(filename=file_path, idx=idx, fs=fs, parse=parse):
                        self.channels.append(channel)
                        logging.info(f"Channel {channel.channel_num} in Attempt {self.attempt_num} loaded successfully.")
                    else:
                        logging.warning(f"Failed to load channel from {filename}")
                else:
                    logging.debug(f"Skipping non-txt file '{filename}'")
            else:
                logging.debug(f"Skipping directory '{filename}'")

    def compute_spectra(self, ch_num=None, use_parsed=False, fs=1.0, welch_params=None):
        """
        Compute Welch-based spectra for all channels in this Attempt,
        but only process the channel that matches ch_num.
        """
        for ch in self.channels:
            ch.compute_spectra(ch_num=ch_num, use_parsed=use_parsed, fs=fs, welch_params=welch_params)

    def subtract_baseline_from_all_channels(self, method='median', **kwargs):
        baselines = {}
        for ch in self.channels:
            result = ch.subtract_baseline(method=method, **kwargs)
            if result is not None:
                adjusted_data, baseline = result
                baselines[ch.channel_num] = baseline
        return baselines
    
    def get_baseline_stats(self):
        """
        Collect baseline statistics for every channel in the attempt.
        """
        stats = {}
        for ch in self.channels:
            ch_stats = ch.get_baseline_stats()
            if ch_stats is not None:
                stats[ch.channel_num] = ch_stats
        return stats

    def subtract_channels(self, channel_num_1, channel_num_2, new_channel_num=None):
        ch1 = next((ch for ch in self.channels if ch.channel_num == channel_num_1), None)
        ch2 = next((ch for ch in self.channels if ch.channel_num == channel_num_2), None)
        if ch1 and ch2:
            if ch1.data_baseline_subtracted is not None and ch2.data_baseline_subtracted is not None:
                data1 = ch1.data_baseline_subtracted
                data2 = ch2.data_baseline_subtracted
                use_baseline = True
            elif ch1.raw_data is not None and ch2.raw_data is not None:
                data1 = ch1.raw_data
                data2 = ch2.raw_data
                use_baseline = False
            else:
                logging.warning("Insufficient data for subtraction.")
                return None

            if data1.shape != data2.shape:
                logging.warning("Data shapes do not match. Cannot subtract.")
                return None

            subtracted_data = data1 - data2
            if new_channel_num is None:
                existing_nums = [ch.channel_num for ch in self.channels]
                new_channel_num = max(existing_nums) + 1 if existing_nums else 0

            new_data_type = 'gradientometer'
            new_channel = Channel(channel_num=new_channel_num, attempt_num=self.attempt_num,
                                  data_type=new_data_type, sample_name=self.sample_name)
            new_channel.time = ch1.time
            if use_baseline:
                new_channel.data_baseline_subtracted = subtracted_data
            else:
                new_channel.raw_data = subtracted_data

            self.channels.append(new_channel)
            logging.info(f"Created new {new_data_type} channel {new_channel_num} by subtracting Channel {channel_num_2} from Channel {channel_num_1}")
            return new_channel
        else:
            missing = []
            if not ch1:
                missing.append(channel_num_1)
            if not ch2:
                missing.append(channel_num_2)
            logging.warning(f"Channels {', '.join(map(str, missing))} not found in Attempt {self.attempt_num}")
            return None

    def subtract_channels_in_all_channels(self, channel_num_1, channel_num_2, new_channel_num=None):
        return self.subtract_channels(channel_num_1, channel_num_2, new_channel_num)

    def denoise_all_channels(self, method='spline', **kwargs):
        for ch in self.channels:
            if method == 'spline':
                ch.denoise_spline_interpolation(**kwargs)
            else:
                logging.warning(f"Denoising method '{method}' not implemented.")

    def perform_me_analysis_all_channels(
        self,
        method='l2',
        channel_num=None,
        fig_dir=None,
        line_idx=0,
        **kwargs
    ):
        """
        Unified method to perform multi-exponential analysis on all channels.
        
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
        for ch in self.channels:
            if channel_num is not None and ch.channel_num != channel_num:
                continue
            ch.perform_multiexponential_analysis(
                method=method,
                fig_dir=fig_dir,
                line_idx=line_idx,
                **kwargs
            )
    
    # Backward compatibility wrappers
    def perform_me_analysis_all_channels_l2(self, **kwargs):
        """L2 analysis (backward compatibility)."""
        return self.perform_me_analysis_all_channels(method='l2', **kwargs)
    
    def perform_me_analysis_all_channels_elastic(self, **kwargs):
        """Elastic Net analysis (backward compatibility)."""
        return self.perform_me_analysis_all_channels(method='elastic', **kwargs)
    
    def perform_me_analysis_all_channels_fixed_taus(self, **kwargs):
        """Fixed tau analysis (backward compatibility)."""
        fit_baseline = kwargs.get('fit_baseline', False)
        method = 'fixed_taus_baseline' if fit_baseline else 'fixed_taus'
        return self.perform_me_analysis_all_channels(method=method, **kwargs)
