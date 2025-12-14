# Experiment: 48 Sensor Positions with 4 Field Excitations

This folder has the code and scripts to reproduce the results for the **48-sensor, 3-face experimental configuration** described in the manuscript. 

The experiment reconstructs Magnetic particle distributions using 48 sensor positions across three faces of the Region of Interest (ROI) and 4 optimized excitation magnetic fields.

## Folder Contents

| File Name | Purpose |
| :--- | :--- |
| `donwload_extract_Raw_data.py` | Downloads and decompresses the raw experimental measurement data. |
| `me_analysis.py` | Calculates baseline signal levels from the raw data. |
| `Experiments_paper.ipynb` | **Main notebook:** Calculates the lead field matrix and performs Fused Lasso reconstruction. |
| `functions_finite_diferences.py` | Contains core functions for the forward model (used by the notebook). |
| `algorithm_solution_FL.py` | Contains the Fused Lasso inverse solver implementation (used by the notebook). |
| `parallel_fused_lasso_4.py` | Enables parallel computation for faster reconstruction (used by the notebook). |
| `plot_function_vista.py` | Helper functions for generating figures and plots. |
| `Difference_finite_devices_clases.py` | Helper functiond for construct the Gain Matrix. |
| `src`| Folder with additional scripts for extract the baselines. |





## How to Run the Experiment

Follow these steps in order to reproduce the results:

1.  **Download the Raw Data**
    Run the following command to download and prepare the required dataset:
    ```bash
    donwload_extract_Raw_data.py
    ```
    **Expected Output**: Creates a folder (e.g., `first_experiment_raw_data_channels/folder_fields`) containing the raw signal files for channels 0 and 1, we use channel 0.

2. **Extract the baselines**
    Processes the raw data to establish baseline signal levels for each sensor channel, use the default parameters.
    ```bash
    me_analysis.py
    ```
    **Expected Output**: Generates baseline files in the `first_experiment_raw_data_channels/information_baselines_ch0_exp_1.txt`.


3.  **Reconstruct MNP Distribution (Core Step)**

    Runs the forward model and solves the inverse problem using Fused Lasso regularization
    ```bash
    Experiments_paper.ipynb
    ```

## Additional Notes

*   **Interactive Analysis**: Open `Experiments_paper.ipynb` in Jupyter Lab/Notebook for a step-by-step walkthrough and interactive result exploration.
*   **Data Source**: The raw data is hosted on https://drive.google.com/file/d/1c35FB6UP9WApqOPpuBbGNr2qjjJOcpR_/view?usp=sharing. The `download_raw_data.py` script handles authentication and download automatically. If you encounter access issues, please contact the corresponding author.
*   **Dependencies**: Ensure all libraries installed before running these scripts (pyvista for visuyalization and cvxpy for the optimization solver).





