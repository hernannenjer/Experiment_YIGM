import logging
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress



def plot_peak_amplitudes_vs_sample(
    experiment, 
    peak_idx=0, 
    sample_properties=None, 
    log_axes=True, 
    exclude_zero_sample=False, 
    use_sparsified=False, 
    show_error=True,
    connect_points=False,         # <--- Optional argument
    x_label='mg',                # <--- Optional argument (by default 'mg')
    fit=True                     # <--- New optional argument (True by default)
):
    """
    Plots the absolute amplitudes (and errors, if show_error=True) of a chosen peak index (peak_idx)
    against a sample property. The property can be numeric or string-like.

    :param experiment: object containing the samples and relevant data
    :param peak_idx: which peak to plot (int)
    :param sample_properties: dict {sample_name: numeric_value} or None
    :param log_axes: if True, attempt log-log scaling (requires positive values)
    :param exclude_zero_sample: if True, skip samples where the property is 0 (only works if property is numeric)
    :param use_sparsified: if True, use the 'sparsified_amplitudes' from channels instead of mean_peaks
    :param show_error: if True, plot error bars
    :param connect_points: if True, draw a line connecting the data points
    :param x_label: label for the x axis (default 'mg')
    :param fit: if True, show a linear (or power-law) fit; if False, skip fitting
    """
    samples = []
    amplitudes = []
    errors = []
    sample_props = []
    
    for sample_name, sample in experiment.samples.items():
        if use_sparsified:
            peak_values = []
            for attempt in sample.attempts:
                for channel in attempt.channels:
                    if channel.sparsified_amplitudes is not None and channel.sparsified_relaxation_times is not None:
                        for i in range(channel.sparsified_amplitudes.shape[0]):
                            amps_i = channel.sparsified_amplitudes[i, :]
                            sorted_indices = np.argsort(-np.abs(amps_i))
                            if peak_idx < len(sorted_indices):
                                chosen_amp = amps_i[sorted_indices[peak_idx]]
                                peak_values.append(chosen_amp)
            if len(peak_values) == 0:
                continue
            mean_amp = np.mean(peak_values)
            std_amp = np.std(peak_values)
            amplitude = mean_amp
            amplitude_error = std_amp
        else:
            if not hasattr(sample, 'mean_peaks') or sample.mean_peaks is None:
                logging.warning(f"No pronounced peaks found for sample '{sample_name}'. Skipping.")
                continue
            if peak_idx >= sample.mean_peaks.shape[0]:
                logging.warning(f"Peak index {peak_idx} out of range for sample '{sample_name}'. Skipping.")
                continue
            amplitude = sample.mean_peaks[peak_idx, 0]
            amplitude_error = sample.std_peaks[peak_idx, 0]

        if sample_properties and sample_name in sample_properties:
            prop = sample_properties[sample_name]
        else:
            try:
                prop = float(sample_name)
            except ValueError:
                prop = sample_name

        if exclude_zero_sample and isinstance(prop, (int, float)) and prop == 0:
            continue

        samples.append(sample_name)
        amplitudes.append(abs(amplitude))
        errors.append(amplitude_error)
        sample_props.append(prop)

    if not samples:
        logging.warning("No peak data available to plot after filtering.")
        return

    amplitudes = np.array(amplitudes)
    errors = np.array(errors)
    sample_props = np.array(sample_props, dtype=object)
    samples = np.array(samples)

    numeric_mask = []
    for val in sample_props:
        if isinstance(val, (int, float)):
            numeric_mask.append(True)
        else:
            try:
                float(val)
                numeric_mask.append(True)
            except ValueError:
                numeric_mask.append(False)

    if all(numeric_mask):
        sample_props = sample_props.astype(float)
        use_numeric_axis = True
    else:
        use_numeric_axis = False
        unique_cats = np.unique(sample_props)
        cat_to_num = {cat: i for i, cat in enumerate(unique_cats)}
        numeric_vals = np.array([cat_to_num[val] for val in sample_props])
        sample_props = numeric_vals

    x_values = sample_props

    if log_axes and use_numeric_axis:
        min_val = np.min(x_values)
        if min_val <= 0:
            shift_amount = 1e-12 - min_val
            x_values = x_values + shift_amount
            logging.info(f"Shifting x_values by {shift_amount} for positivity on log scale.")

    if use_numeric_axis:
        sorted_indices = np.argsort(x_values)
    else:
        sorted_indices = np.argsort(x_values)

    x_values = x_values[sorted_indices]
    samples = samples[sorted_indices]
    amplitudes = amplitudes[sorted_indices]
    errors = errors[sorted_indices]

    # -- �������: �������� �������� � ������ �����, ���� ������� ���-������� � ��� --
    if log_axes and fit and use_numeric_axis and len(amplitudes) > 0:
        first_amp = amplitudes[0]
        min_amp = np.min(amplitudes)
#         amplitudes = amplitudes - min_amp
        print('INFO: Amplitudes for plotting vs samples: '+str(amplitudes))
        # ����� �������� ������������� ��� ������� �������� ����� ���������
        min_amp = np.min(amplitudes)
        if min_amp <= 0:
            shift_amount = 1e-12 - min_amp
            amplitudes += shift_amount
            logging.info(f"Shifting amplitudes by {shift_amount} after subtracting the first amplitude for positivity on log scale.")

    plt.figure(figsize=(10, 6))
    plt.scatter(x_values, amplitudes, label=f'Peak {peak_idx} Amplitude')
    if show_error:
        plt.errorbar(x_values, amplitudes, yerr=errors, fmt='o', capsize=5, label='Error')

    # Connect points if requested
    if connect_points:
        plt.plot(x_values, amplitudes, '-o', color='blue', label='Line between points')

    # Fit (only if fit=True)
    if fit and use_numeric_axis:
        if log_axes:
            if np.all(x_values > 0) and np.all(amplitudes > 0):
                log_x = np.log10(x_values)
                log_y = np.log10(amplitudes)
                slope, intercept, r_value, p_value, std_err = linregress(log_x, log_y)
                fit_line = 10 ** (intercept + slope * log_x)
                plt.plot(x_values, fit_line, 'r--', label=f'Fit: y={10**intercept:.2e}x^{slope:.2f}')
                logging.info(f"Power-law fit: y={10**intercept:.2e}x^{slope:.2f}, R2={r_value**2:.2f}")
            else:
                logging.warning("Not all values are positive. Keeping linear scale.")
        else:
            slope, intercept, r_value, p_value, std_err = linregress(x_values, amplitudes)
            fit_line = slope * x_values + intercept
            plt.plot(x_values, fit_line, 'r--', label=f'Fit: y={slope:.2e}x + {intercept:.2e}')
            logging.info(f"Linear fit: slope={slope:.2e}, intercept={intercept:.2e}, R2={r_value**2:.2f}")

    plt.xlabel(x_label)
    plt.ylabel('Amplitude (absolute value)')
    plt.title(f'Absolute Amplitude of Peak {peak_idx} vs. Sample Property')

    if log_axes:
        if np.all(x_values > 0) and np.all(amplitudes > 0):
            plt.xscale('log')
            plt.yscale('log')

    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.tight_layout()
    plt.show()
    logging.info(f"Amplitude of Peak {peak_idx} vs. Sample plot displayed.")

def plot_relaxation_times_vs_sample(
    experiment, 
    log_axes=True,
    connect_points=False, 
    x_label='mg',
    fit=True    # <--- New optional argument
):
    """
    Plots the mean effective relaxation times (and errors) for each sample.

    :param experiment: object containing the samples and relevant data
    :param log_axes: if True, use log scale for axes (where possible)
    :param connect_points: if True, draw a line connecting the data points
    :param x_label: label for the x axis (default 'mg')
    :param fit: if True, show a linear (or power-law) fit; if False, skip fitting
    """
    relaxation_times_per_sample = {}
    for sample_name, sample in experiment.samples.items():
        tau_eff_list = []
        for attempt in sample.attempts:
            for channel in attempt.channels:
                if channel.relaxation_times is not None and channel.amplitudes is not None:
                    for i in range(channel.amplitudes.shape[0]):
                        M_i = channel.amplitudes[i, :]
                        M_sum = np.sum(M_i)
                        if M_sum > 0:
                            tau_eff = np.sum(M_i * channel.relaxation_times)/M_sum
                            tau_eff_list.append(tau_eff)
        if tau_eff_list:
            mean_tau = np.mean(tau_eff_list)
            std_tau = np.std(tau_eff_list)
            relaxation_times_per_sample[sample_name] = (mean_tau, std_tau)

    if not relaxation_times_per_sample:
        logging.warning("No relaxation times to plot.")
        return

    samples = list(relaxation_times_per_sample.keys())
    mean_taus = [relaxation_times_per_sample[name][0] for name in samples]
    errors = [relaxation_times_per_sample[name][1] for name in samples]

    use_numeric_axis = True
    try:
        sample_numeric_values = [float(name) for name in samples]
    except ValueError:
        use_numeric_axis = False

    if use_numeric_axis:
        sample_numeric_values = np.array(sample_numeric_values)
        sorted_indices = np.argsort(sample_numeric_values)
        x_values = sample_numeric_values[sorted_indices]
        samples = np.array(samples)[sorted_indices]
        mean_taus = np.array(mean_taus)[sorted_indices]
        errors = np.array(errors)[sorted_indices]
    else:
        x_values = np.arange(len(samples))
        sorted_indices = np.argsort(samples)
        samples = np.array(samples)[sorted_indices]
        mean_taus = np.array(mean_taus)[sorted_indices]
        errors = np.array(errors)[sorted_indices]
        x_values = x_values[sorted_indices]

    plt.figure(figsize=(10,6))
    plt.errorbar(x_values, mean_taus, yerr=errors, fmt='o', label='Mean Effective Relaxation Time')
    # Connect points if requested
    if connect_points:
        plt.plot(x_values, mean_taus, '-o', color='blue', label='Line between points')

    # Attempt to fit (only if fit=True)
    if fit and use_numeric_axis:
        if log_axes:
            pos_mask = (x_values>0) & (mean_taus>0)
            if np.any(pos_mask):
                lx = np.log10(x_values[pos_mask])
                ly = np.log10(mean_taus[pos_mask])
                slope, intercept, r_value, p_value, std_err = linregress(lx, ly)
                fit_line = 10 ** (intercept + slope * lx)
                plt.plot(x_values[pos_mask], fit_line, 'r--', label=f'Fit: y={10**intercept:.2e}x^{slope:.2f}')
                logging.info(f"Power-law fit: R2={r_value**2:.2f}")
        else:
            slope, intercept, r_value, p_value, std_err = linregress(x_values, mean_taus)
            fit_line = slope*x_values + intercept
            plt.plot(x_values, fit_line, 'r--', label=f'Fit: y={slope:.2e}x+{intercept:.2e}')
            logging.info(f"Linear fit: R2={r_value**2:.2f}")

    plt.xlabel(x_label)
    plt.ylabel('Mean Effective Relaxation Time')
    plt.title('Mean Effective Relaxation Time vs. Sample Property')

    if log_axes:
        plt.xscale('log')
        plt.yscale('log')

    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.tight_layout()
    plt.show()


def plot_baseline_vs_sample(
    experiment, 
    log_axes=True, 
    ch_num=None,
    connect_points=False,
    x_label=r'$\mu$g',
    fit=True
):
    """
    Plots the baseline vs. sample property for each channel.
    Uses only the 'channel.baseline' field, ignoring data_baseline_subtracted.
    
    :param experiment: object containing the samples
    :param log_axes: if True, use log scale for x,y (where possible)
    :param ch_num: if specified, filter by channel number
    :param connect_points: if True, draw a line connecting the data points
    :param x_label: label for the x axis (default 'mg')
    :param fit: if True, attempt a linear or power-law fit (on log scale)
    """
    baselines_per_channel = {}

    # --- �������� baseline �� ���� ������� ---
    for sample_name, sample in experiment.samples.items():
        for attempt in sample.attempts:
            for channel in attempt.channels:
                # 1) ������ �� ������ ������
                if ch_num is not None and channel.channel_num != ch_num:
                    continue

                # 2) ���������, ��� channel.baseline ������ ���� (�� None)
                if getattr(channel, 'baseline', None) is None:
                    continue

                # ������� ��������� ��������
                if channel.channel_num not in baselines_per_channel:
                    baselines_per_channel[channel.channel_num] = {
                        'sample_props': [],
                        'baselines': []
                    }
                
                # ����������� ��� ������ � �����, ���� ��������
                try:
                    sample_prop = float(sample_name)
                except ValueError:
                    sample_prop = sample_name

                # 3) ���� baseline - ��� ������ (�� ������)
                #    �� �� ������ ������ ���� � ��� �� sample_prop.
                #    ���� baseline - ������, ������� ������ ���� �����.
                baseline_val = channel.baseline
                if isinstance(baseline_val, np.ndarray):
                    # ��������, baseline_val.shape = (Ncurves,)
                    for val in baseline_val:
                        baselines_per_channel[channel.channel_num]['sample_props'].append(sample_prop)
                        baselines_per_channel[channel.channel_num]['baselines'].append(val)
                else:
                    # baseline - float ��� int
                    baselines_per_channel[channel.channel_num]['sample_props'].append(sample_prop)
                    baselines_per_channel[channel.channel_num]['baselines'].append(baseline_val)

    # ���� ������ ������ �� ������� - �������
    if not baselines_per_channel:
        logging.warning("No valid baseline values found for any channels.")
        return

    plt.figure(figsize=(12, 8))

    # --- ������ �� ������� ������ ---
    for c_num, data in baselines_per_channel.items():
        sample_props = np.array(data['sample_props'], dtype=object)
        baseline_vals = np.array(data['baselines'], dtype=float)

        # ���������, ����� �� �� ��� X ���������������� ��� ��������
        use_numeric_axis = True
        try:
            sample_props_numeric = sample_props.astype(float)
        except ValueError:
            use_numeric_axis = False

        if use_numeric_axis:
            # ��������� �� ����������� X
            sorted_idx = np.argsort(sample_props_numeric)
            x = sample_props_numeric[sorted_idx]
            y = baseline_vals[sorted_idx]
        else:
            # �������������� ��� X
            sorted_idx = np.argsort(sample_props)
            x = np.arange(len(sample_props))[sorted_idx]
            y = baseline_vals[sorted_idx]
            sample_props = sample_props[sorted_idx]

        # ����������� ����� � ���������� x (��������, ��������� ������ ������ sName)
        unique_x, group_idx = np.unique(x, return_inverse=True)
        median_baselines = np.array([np.median(y[group_idx == i]) for i in range(len(unique_x))])
        std_baselines = np.array([np.std(y[group_idx == i]) for i in range(len(unique_x))])

        x_plot = unique_x
        y_plot = np.abs(median_baselines) #FIXME
        print(y_plot)
        y_plot -= y_plot[0]
        y_err = std_baselines


        # --- ������ ---
#         plt.errorbar(x_plot, y_plot, yerr=y_err, fmt='o', capsize=5, label=f'Channel {c_num}')
        plt.scatter(x_plot, y_plot, label=f'Channel {c_num}')
        if connect_points:
            plt.plot(x_plot, y_plot, '-o', color='blue')

        # --- ���, ���� ���� ---
        if fit and use_numeric_axis:
            # ��ң� ������ ������������� ����� (���� ����� power-law �� log scale)
            if log_axes:
                pos_mask = (x_plot > 0) & (y_plot > 0)
                if np.any(pos_mask):
                    lx = np.log10(x_plot[pos_mask])
                    ly = np.log10(y_plot[pos_mask])
                    slope, intercept, r_value, p_value, std_err = linregress(lx, ly)
                    fit_line = 10 ** (intercept + slope * lx)
                    plt.plot(x_plot[pos_mask], fit_line, 'r--')
                    plt.text(
                        x_plot[pos_mask][-1], 
                        fit_line[-1], 
                        f'y={10**intercept:.2e}x^{slope:.2f}', 
                        fontsize=9, 
                        color='r'
                    )
                    logging.info(f"Power-law fit for Channel {c_num}: R2={r_value**2:.2f}")
            else:
                pos_mask = (x_plot >= 0) & (y_plot >= 0)
                # �������� ���
                if np.any(pos_mask):
                    slope, intercept, r_value, p_value, std_err = linregress(x_plot[pos_mask], y_plot[pos_mask])
                    fit_line = slope * x_plot[pos_mask] + intercept
                    plt.plot(x_plot[pos_mask], fit_line, 'r--')
                    plt.text(
                        x_plot[pos_mask][-1], 
                        fit_line[-1], 
                        f'y={slope:.2e}x+{intercept:.2e}', 
                        fontsize=9, 
                        color='r'
                    )
                    logging.info(f"Linear fit for Channel {c_num}: R2={r_value**2:.2f}")

    plt.xlabel(x_label)
    plt.ylabel('Baseline')
    plt.title('Baseline vs. Sample Property (from channel.baseline only)')
    if log_axes and use_numeric_axis:
        plt.xscale('log')
        plt.yscale('log')

    plt.legend()
    plt.grid(True, which="both", ls="--", linewidth=0.5)
    plt.tight_layout()
    plt.show()
    logging.info("Baseline vs. Sample plot displayed.")

def plot_integrals_vs_sample(
    experiment, 
    log_axes=True,
    connect_points=False,
    x_label='mg',
    fit=True  # <--- New optional argument
):
    """
    Plots the median integrals (and errors) of raw data vs. sample property.

    :param experiment: object containing the samples
    :param log_axes: if True, use log scale for axes (where possible)
    :param connect_points: if True, draw a line connecting the data points
    :param x_label: label for the x axis (default 'mg')
    :param fit: if True, show a linear (or power-law) fit; if False, skip fitting
    """
    integrals_per_sample = {}
    for sample_name, sample in experiment.samples.items():
        sample_integrals = []
        for attempt in sample.attempts:
            for channel in attempt.channels:
                if channel.raw_data is not None:
                    integrals = np.trapz(np.abs(channel.raw_data), channel.time, axis=1)
                    channel.integrals = integrals
                    sample_integrals.extend(integrals)
        if sample_integrals:
            sample_median_integral = np.median(sample_integrals)
            sample_std_integral = np.std(sample_integrals)
            integrals_per_sample[sample_name] = (sample_median_integral, sample_std_integral)

    if not integrals_per_sample:
        logging.warning("No valid samples to plot.")
        return

    samples = list(integrals_per_sample.keys())
    integrals = [integrals_per_sample[name][0] for name in samples]
    errors = [integrals_per_sample[name][1] for name in samples]

    use_numeric_axis = True
    try:
        sample_numeric_values = [float(name) for name in samples]
    except ValueError:
        use_numeric_axis = False

    if use_numeric_axis:
        sample_numeric_values = np.array(sample_numeric_values)
        sorted_indices = np.argsort(sample_numeric_values)
        x_values = sample_numeric_values[sorted_indices]
        samples = np.array(samples)[sorted_indices]
        integrals = np.array(integrals)[sorted_indices]
        errors = np.array(errors)[sorted_indices]
    else:
        x_values = np.arange(len(samples))
        sorted_indices = np.argsort(samples)
        samples = np.array(samples)[sorted_indices]
        integrals = np.array(integrals)[sorted_indices]
        errors = np.array(errors)[sorted_indices]
        x_values = x_values[sorted_indices]

    plt.figure(figsize=(10,6))
    plt.errorbar(x_values, integrals, yerr=errors, fmt='o', label='Median Integral')
    # Connect points if requested
    if connect_points:
        plt.plot(x_values, integrals, '-o', color='blue', label='Line between points')

    # Attempt to fit (only if fit=True)
    if fit and use_numeric_axis:
        from scipy.stats import linregress
        pos_mask = (x_values>0) & (integrals>0)
        if log_axes:
            if np.any(pos_mask):
                lx = np.log10(x_values[pos_mask])
                ly = np.log10(integrals[pos_mask])
                slope, intercept, r_value, p_value, std_err = linregress(lx, ly)
                fit_line = 10 ** (intercept + slope * lx)
                plt.plot(x_values[pos_mask], fit_line, 'r--', label=f'Fit: y={10**intercept:.2e}x^{slope:.2f}')
                logging.info(f"Power-law fit: R2={r_value**2:.2f}")
        else:
            slope, intercept, r_value, p_value, std_err = linregress(x_values[pos_mask], integrals[pos_mask])
            fit_line = slope*x_values[pos_mask] + intercept
            plt.plot(x_values[pos_mask], fit_line, 'r--', label=f'Fit: y={slope:.2e}x + {intercept:.2e}')
            logging.info(f"Linear fit: R2={r_value**2:.2f}")

    plt.xlabel(x_label)
    plt.ylabel('Median Integral')
    plt.title('Median Integral vs. Sample Property')

    if log_axes:
        plt.xscale('log')
        plt.yscale('log')

    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.tight_layout()
    plt.show()
    logging.info("Integral vs. Sample plot displayed.")








def plot_psd_peak_vs_sample(
    experiment,
    freq_range=(70, 90),
    sample_properties=None,
    log_axes=True,
    exclude_zero_sample=False,
    show_error=True,
    connect_points=False,
    x_label='mg',
    fit=True
):
    """
    Plots the maximum PSD amplitude found within a specified frequency window (freq_range)
    vs. a sample property (e.g., mass).

    :param experiment: object containing the samples and relevant data
    :param freq_range: (f_min, f_max) tuple specifying the frequency window
    :param sample_properties: dict {sample_name: numeric_value} or None
    :param log_axes: if True, attempt log-log scaling (requires positive values)
    :param exclude_zero_sample: if True, skip samples where the property is 0 (for numeric properties)
    :param show_error: if True, plot error bars (std. deviation across attempts/channels)
    :param connect_points: if True, draw a line connecting the data points
    :param x_label: label for the x axis (default 'mg')
    :param fit: if True, attempt a linear or power-law fit (on log scale)
    """
    samples = []
    amplitudes = []
    errors = []
    sample_props = []

    f_min, f_max = freq_range

    # Loop over each sample
    for sample_name, sample in experiment.samples.items():
        psd_peak_values = []  # will collect the maximum PSD in freq_range for each attempt/channel

        # Loop over attempts/channels
        for attempt in sample.attempts:
            for channel in attempt.channels:
                # Make sure channel has PSD data
                if channel.freqs_spectra is None or channel.psd_spectra is None:
                    continue
                freqs = channel.freqs_spectra
                psd = channel.psd_spectra  # 1D array after averaging across curves

                # Select the portion of PSD within freq_range
                mask = (freqs >= f_min) & (freqs <= f_max)
                if not np.any(mask):
                    continue  # no data in that freq window

                sub_psd = psd[mask]
                if sub_psd.size == 0:
                    continue

                # Find the maximum PSD in this window
                peak_val = np.max(sub_psd)
                psd_peak_values.append(peak_val)

        # If we found no peak values, skip this sample
        if len(psd_peak_values) == 0:
            logging.warning(f"No PSD data in freq range {freq_range} for sample '{sample_name}'. Skipping.")
            continue

        # Compute mean & std
        mean_amp = np.mean(psd_peak_values)
        std_amp = np.std(psd_peak_values)

        # Determine the sample property (mass, or numeric from the folder name, etc.)
        if sample_properties and sample_name in sample_properties:
            prop = sample_properties[sample_name]
        else:
            # Try converting sample_name to float (if your folder is named '12.5', etc.)
            try:
                prop = float(sample_name)
            except ValueError:
                prop = sample_name

        # Possibly skip sample if property is zero and exclude_zero_sample is True
        if exclude_zero_sample and isinstance(prop, (int, float)) and prop == 0:
            continue

        samples.append(sample_name)
        amplitudes.append(mean_amp)
        errors.append(std_amp)
        sample_props.append(prop)

    # If no samples found after filtering, just return
    if not samples:
        logging.warning("No PSD peak data available to plot after filtering.")
        return

    # Convert lists to numpy arrays
    amplitudes = np.array(amplitudes)
    errors = np.array(errors)
    sample_props = np.array(sample_props, dtype=object)
    samples = np.array(samples)

    # Determine whether sample_props are numeric or categorical
    numeric_mask = []
    for val in sample_props:
        if isinstance(val, (int, float)):
            numeric_mask.append(True)
        else:
            try:
                float(val)
                numeric_mask.append(True)
            except ValueError:
                numeric_mask.append(False)

    if all(numeric_mask):
        sample_props = sample_props.astype(float)
        use_numeric_axis = True
    else:
        use_numeric_axis = False
        unique_cats = np.unique(sample_props)
        cat_to_num = {cat: i for i, cat in enumerate(unique_cats)}
        numeric_vals = np.array([cat_to_num[val] for val in sample_props])
        sample_props = numeric_vals

    x_values = sample_props

    # If log_axes requested, ensure positivity
    if log_axes and use_numeric_axis:
        min_val = np.min(x_values)
        if min_val <= 0:
            shift_amount = 1e-12 - min_val
            x_values = x_values + shift_amount
            logging.info(f"Shifting x_values by {shift_amount} for positivity on log scale.")

    # Sort data by x_values so plot lines connect in ascending order
    sorted_indices = np.argsort(x_values)
    x_values = x_values[sorted_indices]
    samples = samples[sorted_indices]
    amplitudes = np.sqrt(amplitudes[sorted_indices])
    errors = errors[sorted_indices]

    # Plot
    plt.figure(figsize=(10, 6))
    plt.scatter(x_values, amplitudes, label=f'PSD Peak in [{f_min}, {f_max}] Hz')
    if show_error:
        plt.errorbar(x_values, amplitudes, yerr=errors, fmt='o', capsize=5, label='Error')

    if connect_points:
        plt.plot(x_values, amplitudes, '-o', color='blue', label='Line between points')

    # Attempt a fit if requested
    if fit and use_numeric_axis:
        # Check positivity if log_axes
        if log_axes and np.all(x_values > 0) and np.all(amplitudes > 0):
            log_x = np.log10(x_values)
            log_y = np.log10(amplitudes)
            slope, intercept, r_value, p_value, std_err = linregress(log_x, log_y)
            fit_line = 10 ** (intercept + slope * log_x)
            plt.plot(x_values, fit_line, 'r--',
                     label=f'Fit: y={10**intercept:.2e}x^{slope:.2f}')
            logging.info(f"Power-law fit: y={10**intercept:.2e}x^{slope:.2f}, R2={r_value**2:.2f}")
        else:
            slope, intercept, r_value, p_value, std_err = linregress(x_values, amplitudes)
            fit_line = slope * x_values + intercept
            plt.plot(x_values, fit_line, 'r--',
                     label=f'Fit: y={slope:.2e}x + {intercept:.2e}')
            logging.info(f"Linear fit: slope={slope:.2e}, intercept={intercept:.2e}, R2={r_value**2:.2f}")

    plt.xlabel(x_label)
    plt.ylabel('Max PSD in freq window')
    plt.title(f'PSD Peak in [{f_min}, {f_max}] Hz vs. Sample Property')

    if log_axes and np.all(x_values > 0) and np.all(amplitudes > 0):
        plt.xscale('log')
        plt.yscale('log')

    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.tight_layout()
    plt.show()
    logging.info(f"PSD peak in freq window {freq_range} vs. sample property plot displayed.")
