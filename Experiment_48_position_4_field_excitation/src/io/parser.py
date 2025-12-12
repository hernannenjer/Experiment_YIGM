import numpy as np
import logging
import os
import matplotlib.pyplot as plt

def parse_channel_data(data, idx, filename, fig_path=None, show=False, result_path=None):
    try:
        i0, i1, i2, i3, i_dt, di = idx
        ddi = 2 * i1 + i2 + i3

        lpeaks, rpeaks, lrels, rrels, noise = [], [], [], [], []
        mask_lpeaks = np.zeros_like(data)
        mask_rpeaks = np.zeros_like(data)
        mask_rrels = np.zeros_like(data)
        mask_lrels = np.zeros_like(data)
        mask_noise = np.zeros_like(data)

        start_lpeak = i0
        end_cons = i0 + ddi
        i = 1
        while end_cons < data.shape[0]:
            end_lpeak = start_lpeak + i1
            start_rpeak = end_lpeak + i2
            end_rpeak = start_rpeak + i1

            lpeaks.append(data[start_lpeak:end_lpeak])
            rpeaks.append(data[start_rpeak + 1:end_rpeak])
            lrels.append(data[end_lpeak + i_dt:end_lpeak + i_dt + di])
            rrels.append(data[end_rpeak + i_dt:end_rpeak + i_dt + di])
            noise.append(data[end_rpeak + di + 1: end_cons - i_dt])

            mask_lpeaks[start_lpeak:end_lpeak] = 1
            mask_rpeaks[start_rpeak + 1:end_rpeak] = 1
            mask_lrels[end_lpeak + i_dt:end_lpeak + i_dt + di] = 1
            mask_rrels[end_rpeak + i_dt:end_rpeak + i_dt + di] = 1
            mask_noise[end_rpeak + di + 1: end_cons - i_dt] = 1

            start_lpeak = i0 + i * ddi
            end_cons = start_lpeak + ddi
            i += 1

        if fig_path is not None or show:
            fig = plt.figure()
            plt.plot(data[0: i0 + 3 * ddi], label='data')
            plt.plot(mask_lpeaks[0:i0 + 3 * ddi], label='lpeaks')
            plt.plot(mask_rpeaks[0:i0 + 3 * ddi], label='rpeaks')
            plt.plot(mask_rrels[0:i0 + 3 * ddi], label='rrels')
            plt.plot(mask_lrels[0:i0 + 3 * ddi], label='lrels')
            plt.plot(mask_noise[0:i0 + 3 * ddi], label='noise')
            plt.legend(loc='upper right')
            if fig_path is not None:
                fig.savefig(os.path.join(fig_path, 'signal_masks.png'))
            if show:
                plt.show()
            plt.close()

        lpeaks = np.array(lpeaks)
        rpeaks = np.array(rpeaks)
        lrels = np.array(lrels)
        rrels = np.array(rrels)
        noise = np.array(noise)

        if result_path is not None:
            base_filename = os.path.splitext(os.path.basename(filename))[0]
            np.save(os.path.join(result_path, base_filename + '_lpeaks.npy'), lpeaks)
            np.save(os.path.join(result_path, base_filename + '_rpeaks.npy'), rpeaks)
            np.save(os.path.join(result_path, base_filename + '_lrels.npy'), lrels)
            np.save(os.path.join(result_path, base_filename + '_rrels.npy'), rrels)
            np.save(os.path.join(result_path, base_filename + '_noise.npy'), noise)
            np.save(os.path.join(result_path, base_filename + '_data.npy'), data)
            logging.info('Parsed data saved.')

        logging.info('Data parsed successfully.')
        return lpeaks, rpeaks, lrels, rrels, noise, data
    except Exception as e:
        logging.error(f"Error during parsing: {e}")
        return None
