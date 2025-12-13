import numpy as np
import logging
import os
import matplotlib.pyplot as plt
import random

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


# def parse_channel_data(data, idx, filename, fig_path=None, show=False,result_path=None):
#     """
#     Parse channel data by extracting specific segments (peaks, release phases, noise)
#     from randomly selected cycles instead of all cycles.
    
#     Parameters:
#     - data: Input signal data array
#     - idx: Tuple containing indices (i0, i1, i2, i3, i_dt, di) for segmentation
#     - filename: Name of the source file
#     - fig_path: Optional path to save visualization figures
#     - show: Boolean to display visualization
#     - result_path: Optional path to save extracted segments
#     - num_cycles: Number of random cycles to extract (default: 15)
    
#     Returns:
#     - Extracted segments (lpeaks, rpeaks, lrels, rrels, noise) and original data
#     - Returns None if an error occurs
#     """
    
#     try:
#         num_cycles=15
        
#         # Unpack index parameters from the idx tuple
#         i0, i1, i2, i3, i_dt, di = idx
#         # Calculate the total length of one complete cycle/segment
#         ddi = 2 * i1 + i2 + i3

#         # Initialize empty lists to store extracted segments
#         lpeaks, rpeaks, lrels, rrels, noise = [], [], [], [], []
#         # Create binary masks to visualize segment locations (same shape as input data)
#         mask_lpeaks = np.zeros_like(data)
#         mask_rpeaks = np.zeros_like(data)
#         mask_rrels = np.zeros_like(data)
#         mask_lrels = np.zeros_like(data)
#         mask_noise = np.zeros_like(data)
        
#         # Store all valid cycle start positions
#         cycle_starts = []
#         start_lpeak = i0
#         end_cons = i0 + ddi
        
#         i=1
#         # print('hhhhhh',end_cons, data.shape[0])
#         # First pass: Find all possible cycle start positions
#         while end_cons < data.shape[0]:
#             cycle_starts.append(start_lpeak)
#             start_lpeak = i0 + i * ddi
#             end_cons = start_lpeak + ddi
#             i += 1
#             # print('hhhhhh',end_cons, cycle_starts)
        
#         # Check if we have enough cycles
#         total_cycles = len(cycle_starts)

        
#         if total_cycles < num_cycles:
#             logging.warning(f"Only {total_cycles} cycles available, extracting all of them")
#             num_cycles = total_cycles
        
#         # Randomly select cycle start positions (true random each time)
#         if total_cycles > 0:
#             selected_starts = random.sample(cycle_starts, min(num_cycles, len(cycle_starts)))
#             selected_starts.sort()  # Sort for sequential processing
#         else:
#             selected_starts = []
        
#         logging.info(f"Selected {len(selected_starts)} random cycles out of {total_cycles} total cycles")
        
#         # Second pass: Extract only from selected cycles
#         for cycle_num, start_lpeak in enumerate(selected_starts):
#             # Define boundaries for different segments within the current cycle
#             end_lpeak = start_lpeak + i1  # End of left peak
#             start_rpeak = end_lpeak + i2  # Start of right peak (gap after left peak)
#             end_rpeak = start_rpeak + i1  # End of right peak

#             end_cons = start_lpeak + ddi  # End of current cycle

#             # Extract and store each segment from the data
#             lpeaks.append(data[start_lpeak:end_lpeak])  # Left peak data
#             rpeaks.append(data[start_rpeak + 1:end_rpeak])  # Right peak data (starting 1 index later)
#             lrels.append(data[end_lpeak + i_dt:end_lpeak + i_dt + di])  # Left release phase
#             rrels.append(data[end_rpeak + i_dt:end_rpeak + i_dt + di])  # Right release phase
#             noise.append(data[end_rpeak + di + 1: end_cons - i_dt])  # Noise segment between cycles

#             # Update masks to mark where each segment occurs in the original data
#             mask_lpeaks[start_lpeak:end_lpeak] = 1
#             mask_rpeaks[start_rpeak + 1:end_rpeak] = 1
#             mask_lrels[end_lpeak + i_dt:end_lpeak + i_dt + di] = 1
#             mask_rrels[end_rpeak + i_dt:end_rpeak + i_dt + di] = 1
#             mask_noise[end_rpeak + di + 1: end_cons - i_dt] = 1

#         # Create visualization if requested (fig_path provided or show=True)
#         if fig_path is not None or show:
#             fig = plt.figure()
            
#             # Determine visualization range: show selected cycles in context
#             if selected_starts:
#                 min_start = max(0, selected_starts[0] - ddi)  # Show one cycle before first
#                 max_end = min(data.shape[0], selected_starts[-1] + 2 * ddi)  # Show two cycles after last
                
#                 plt.plot(data[min_start:max_end], label='data', alpha=0.7)
#                 plt.plot(mask_lpeaks[min_start:max_end], label='lpeaks', alpha=0.8)
#                 plt.plot(mask_rpeaks[min_start:max_end], label='rpeaks', alpha=0.8)
#                 plt.plot(mask_rrels[min_start:max_end], label='rrels', alpha=0.8)
#                 plt.plot(mask_lrels[min_start:max_end], label='lrels', alpha=0.8)
#                 plt.plot(mask_noise[min_start:max_end], label='noise', alpha=0.8)
                
#                 # Mark selected cycles with vertical lines
#                 for i, start in enumerate(selected_starts):
#                     if min_start <= start <= max_end:
#                         plt.axvline(x=start - min_start, color='red', alpha=0.3, linestyle='--')
#                         if i % 5 == 0:  # Label every 5th cycle for clarity
#                             plt.text(start - min_start, plt.ylim()[1] * 0.95, 
#                                     f'Cycle {i+1}', rotation=90, alpha=0.5)
#             else:
#                 # Fallback if no cycles selected
#                 plt.plot(data[0: i0 + 3 * ddi], label='data')
#                 plt.plot(mask_lpeaks[0:i0 + 3 * ddi], label='lpeaks')
#                 plt.plot(mask_rpeaks[0:i0 + 3 * ddi], label='rpeaks')
#                 plt.plot(mask_rrels[0:i0 + 3 * ddi], label='rrels')
#                 plt.plot(mask_lrels[0:i0 + 3 * ddi], label='lrels')
#                 plt.plot(mask_noise[0:i0 + 3 * ddi], label='noise')
            
#             plt.title(f'Randomly Selected {len(selected_starts)} Cycles (out of {total_cycles})')
#             plt.legend(loc='upper right')
#             plt.grid(True, alpha=0.3)
            
#             # Save figure if path provided
#             if fig_path is not None:
#                 base_filename = os.path.splitext(os.path.basename(filename))[0]
#                 fig.savefig(os.path.join(fig_path, f'{base_filename}_random_cycles.png'), 
#                            dpi=150, bbox_inches='tight')
#             # Display figure if requested
#             if show:
#                 plt.show()
#             plt.close()  # Close the figure to free memory

#         # Convert lists of segments to numpy arrays for easier handling
#         lpeaks = np.array(lpeaks)
#         rpeaks = np.array(rpeaks)
#         lrels = np.array(lrels)
#         rrels = np.array(rrels)
#         noise = np.array(noise)

#         # Save extracted segments and original data if result_path provided
#         if result_path is not None:
#             # Extract base filename without extension
#             base_filename = os.path.splitext(os.path.basename(filename))[0]
            
#             # Save metadata about the random selection
#             metadata = {
#                 'total_cycles': total_cycles,
#                 'selected_cycles': len(selected_starts),
#                 'selected_indices': selected_starts,
#                 'cycle_length': ddi,
#                 'random_selection': True
#             }
            
#             # Save segments

#             np.save(os.path.join(result_path, base_filename + '_random_lpeaks.npy'), lpeaks)
#             np.save(os.path.join(result_path, base_filename + '_random_rpeaks.npy'), rpeaks)
#             np.save(os.path.join(result_path, base_filename + '_random_lrels.npy'), lrels)
#             np.save(os.path.join(result_path, base_filename + '_random_rrels.npy'), rrels)
#             np.save(os.path.join(result_path, base_filename + '_random_noise.npy'), noise)
#             np.save(os.path.join(result_path, base_filename + '_data.npy'), data)
#             np.save(os.path.join(result_path, base_filename + '_random_metadata.npy'), metadata)
            
            

            
#             logging.info(f'Random {len(selected_starts)} cycles parsed and saved.')

#         # Log success and return all extracted segments
#         logging.info(f'Data parsed successfully. Extracted {len(selected_starts)} random cycles.')
#         return lpeaks, rpeaks, lrels, rrels, noise, data
        
#     except Exception as e:
#         # Log any errors that occur during parsing
#         logging.error(f"Error during parsing: {e}")
#         return None