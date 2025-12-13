import numpy as np
import logging
from pathlib import Path

def load_raw_data(filename, fs):
    results_dir = Path(filename).parent / 'results'
    npy_filename = Path(filename).with_suffix('.npy')
    npy_file_path = results_dir / npy_filename.name

    if npy_file_path.exists():
        logging.info(f"Loading raw data from {npy_file_path}")
        unparsed_channel_data = np.load(npy_file_path)
    else:
        logging.info(f"Loading raw data from {filename}")
        with open(filename, 'r') as file:
            unparsed_channel_data = np.loadtxt(file)
        results_dir.mkdir(parents=True, exist_ok=True)
        np.save(npy_file_path, unparsed_channel_data)
        logging.info(f"Raw data saved to {npy_file_path}")

    time_unparsed = np.arange(unparsed_channel_data.shape[0]) / fs
    return unparsed_channel_data, time_unparsed
