import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numpy import random
from numpy import linalg as LA
from scipy.linalg import svd

from scipy.sparse import kron, diags, vstack
from scipy.sparse.linalg import spsolve
from scipy import sparse
import numpy.linalg as LA

# Import utils for graphs
import matplotlib.pyplot as plt

import pandas as pd

# Misc
from tqdm import tqdm
import time
import warnings
warnings.filterwarnings('ignore')


from PIL import Image
import os

import pyvista as pv

from os import makedirs
from os.path import join, exists

def plot_volume_pdf_1(grid, domain ,s=None, coils=None, axis=True, nb=True, num_max_voxels=3,
                              save_filename=None, camera_position=None):
    """
    Function to plot sensors (red points) and coils (blue points) in a 3D grid.
    
    Parameters:
    - grid: The main 3D dataset to plot (e.g., a PyVista grid).
    - s (list of PyVista datasets, optional): Sensors to plot as red points.
    - coils (list of PyVista datasets, optional): Coils to plot as blue points.
    - axis (bool, default=True): Whether to show the axes.
    - nb (bool, default=True): Whether to use notebook mode (for Jupyter Notebook).
    - save_filename (str, optional): Filename to save the plot (e.g., "plot.png" or "plot.pdf").
    - camera_position (tuple or list, optional): Camera position for the plot 
      (e.g., [(x, y, z), (target_x, target_y, target_z), (up_x, up_y, up_z)]).
      
    Returns:
    - None
    """

    # Create the Plotter instance
    plotter = pv.Plotter(notebook=nb)

    # Add the main grid to the plot
    # Get scalar values
    values = grid.cell_data["values"]
    max_val = values.max()
    min_val = values.min()
    min_val = 0
    
    # Find threshold for top N voxels
    threshold = np.sort(values.flatten())[-num_max_voxels]
    thresholded = grid.threshold([threshold, max_val])
    

    import matplotlib.colors as mcolors
    from matplotlib.cm import viridis
    
    # Create custom colormap: viridis colors with alpha proportional to value
    cmap = viridis
    norm = mcolors.Normalize(vmin=min_val, vmax=max_val)
    
    # Create custom colormap with transparency
    custom_cmap = []
    for value in np.linspace(min_val, max_val, 256):
        # Get viridis color
        rgba = cmap(norm(value))
        # Modify alpha: 0 at min_val, 1 at max_val
        alpha = norm(value)  # Linear alpha from 0 to 1
        custom_cmap.append([rgba[0], rgba[1], rgba[2], alpha])
    
    custom_cmap = mcolors.ListedColormap(custom_cmap)
    
    # Add thresholded volume with custom colormap (color + transparency)
    plotter.add_mesh(thresholded, 
                   show_edges=True, 
                   cmap=custom_cmap,
                   clim=[min_val, max_val],
                   scalar_bar_args={
                       'title': '', 
                       'vertical': False,
                       'n_labels': 5,
                       'fmt': '%.2f',
                       'title_font_size': 20,
                       'label_font_size': 16,
                       'shadow': True
                   })
    
    # Highlight absolute maximum voxel
    max_voxel = grid.threshold([max_val - 1e-9, max_val])
    plotter.add_mesh(max_voxel,
                   color='yellow',
                   opacity=1.0,
                   edge_color='white',
                   line_width=3,
                   label=f'Max Value: {max_val:.4f}')
    
    # Add wireframe of full volume (transparent)
    plotter.add_mesh(grid, 
                   style='wireframe', 
                   color='gray', 
                   opacity=0.1,
                   line_width=0.5)







    # # Create opacity mapping for thresholded volume
    # thresholded_values = thresholded.cell_data["values"]
    # normalized_values = (thresholded_values - threshold) / (max_val - threshold)
    # opacity = 0.2 + 0.8 * normalized_values  # Range from 0.2 to 1.0
    
    # # Add thresholded volume with value-dependent opacity
    # plotter.add_mesh(thresholded, 
    #                show_edges=True, 
    #                cmap='viridis',
    #                clim=[min_val, max_val],
    #                opacity=opacity)  # Apply custom opacity
    
    # # Highlight absolute maximum voxel
    # max_voxel = grid.threshold([max_val - 1e-9, max_val])  # Small epsilon to catch max
    # plotter.add_mesh(max_voxel,
    #                color='yellow',
    #                opacity=1.0,
    #                edge_color='white',
    #                line_width=3)
    
    # # Add wireframe of full volume (transparent)
    # plotter.add_mesh(grid, 
    #                style='wireframe', 
    #                color='gray', 
    #                opacity=0.1,  # More transparent
    #                line_width=0.5)
    







    # Add sensors (if provided)
    if s is not None:
        for l in range(len(s)):
            plotter.add_mesh(
                s[l],
                opacity=1,
                color='red',
                render_points_as_spheres=True,
                point_size=8
            )

    # Add coils (if provided)
    if coils is not None:
        for l in range(len(coils)):
            plotter.add_mesh(
                coils[l],
                opacity=1,
                color='blue',
                render_points_as_spheres=True,
                point_size=8
            )

    # Create and add the bounding box/cube
    bounds = [
        domain['x'][0], domain['x'][1],
        domain['y'][0], domain['y'][1], 
        domain['z'][0], domain['z'][1]]
    
    # Create a wireframe box representing the domain
    box = pv.Box(bounds=bounds)
    plotter.add_mesh(box, 
               style='wireframe', 
               color='black', 
               line_width=1,
               opacity=0.5)
    
    

    # Show or hide axes
    if axis:
        plotter.show_axes()

    # Set the camera position if provided
    if camera_position is not None:
        plotter.camera_position = camera_position 



    # Save the plot to a file if a filename is provided
    if save_filename is not None:
        # Extract the desired file extension
        if save_filename.lower().endswith('.png'):
            plotter.show(screenshot=save_filename)  # Save as PNG with high resolution
        elif save_filename.lower().endswith('.pdf'):
            plotter.scalar_bar.GetLabelTextProperty().SetFontSize(35)  # Increase font size
            plotter.scalar_bar.SetTitle(" ")
            plotter.scalar_bar.GetTitleTextProperty().SetFontSize(25)  # Increase title size
            plotter.save_graphic(save_filename) # Save as vectorized PDF
        else:
            raise ValueError("Unsupported file format. Use '.png' or '.pdf' for save_filename.")
    # # Customize the scalar bar (legend)
    
    # # plotter.scalar_bar.SetNumberOfLabels(5)  # Adjust number of labels
    # plotter.scalar_bar.GetLabelTextProperty().SetFontSize(35)  # Increase font size
    # plotter.scalar_bar.SetTitle(" ")
    # plotter.scalar_bar.GetTitleTextProperty().SetFontSize(25)  # Increase title size
    plotter.show()

def plot_volume_max_voxels_1(grid, domain, s=None, labels=None, coils=None, axis=True, nb=False, num_max_voxels=3):
    '''Plot with opacity scaled by values for thresholded volume
    
    Parameters:
    -----------
    grid : pyvista.UnstructuredGrid
        The volume grid to plot
    domain : dict
        Dictionary with domain boundaries {'x': [min, max], 'y': [min, max], 'z': [min, max]}
    s : list of pyvista.PolyData, optional
        List of sensor point clouds
    labels : list of str, optional
        List of labels for each sensor (must match length of s)
    coils : list of pyvista.PolyData, optional
        List of coil point clouds
    axis : bool, optional
        Whether to show axes
    nb : bool, optional
        Whether plotting in notebook
    num_max_voxels : int, optional
        Number of top voxels to highlight
    '''
    plotter = pv.Plotter(notebook=nb)
    
    # Get scalar values
    values = grid.cell_data["values"]
    max_val = values.max()
    min_val = values.min()
    # min_val = 0
    
    # Find threshold for top N voxels
    threshold = np.sort(values.flatten())[-num_max_voxels]
    thresholded = grid.threshold([threshold, max_val])
    
    # Create a custom colormap with transparency
    # Viridis-like colors but with alpha channel
    import matplotlib.colors as mcolors
    from matplotlib.cm import viridis
    
    # Create custom colormap: viridis colors with alpha proportional to value
    cmap = viridis
    norm = mcolors.Normalize(vmin=min_val, vmax=max_val)
    
    # Create custom colormap with transparency
    custom_cmap = []
    for value in np.linspace(min_val, max_val, 256):
        # Get viridis color
        rgba = cmap(norm(value))
        # Modify alpha: 0 at min_val, 1 at max_val
        alpha = norm(value)  # Linear alpha from 0 to 1
        custom_cmap.append([rgba[0], rgba[1], rgba[2], alpha])
    
    custom_cmap = mcolors.ListedColormap(custom_cmap)
    
    # Add thresholded volume with custom colormap (color + transparency)
    plotter.add_mesh(thresholded, 
                   show_edges=True, 
                   cmap=custom_cmap,
                   clim=[min_val, max_val],
                   scalar_bar_args={
                       'title': '', 
                       'vertical': True,
                       'n_labels': 5,
                       'fmt': '%.2f',
                       'title_font_size': 20,
                       'label_font_size': 16,
                       'shadow': True
                   })
    
    # Highlight absolute maximum voxel
    max_voxel = grid.threshold([max_val - 1e-9, max_val])
    plotter.add_mesh(max_voxel,
                   color='yellow',
                   opacity=1.0,
                   edge_color='white',
                   line_width=3,
                   label=f'Max Value: {max_val:.4f}')
    
    # Add wireframe of full volume (transparent)
    plotter.add_mesh(grid, 
                   style='wireframe', 
                   color='gray', 
                   opacity=0.1,
                   line_width=0.5)
    
    # Add sensors with labels
    if s is not None:
        for l in range(len(s)):
            plotter.add_mesh(s[l],
                           opacity=1, 
                           color='red',
                           render_points_as_spheres=True, 
                           point_size=8)
            
            if labels is not None and l < len(labels):
                center = s[l].center
                plotter.add_point_labels(
                    center,
                    labels=[labels[l]],
                    font_size=19,
                    text_color='red',
                    shadow=True,
                    shape=None,
                    always_visible=True
                )
    
    # Add coils
    if coils is not None:
        for l in range(len(coils)):
            plotter.add_mesh(coils[l],
                           opacity=1, 
                           color='blue',
                           render_points_as_spheres=True, 
                           point_size=8)
    
    # Create and add the bounding box
    bounds = [
        domain['x'][0], domain['x'][1],
        domain['y'][0], domain['y'][1], 
        domain['z'][0], domain['z'][1]]
    
    box = pv.Box(bounds=bounds)
    plotter.add_mesh(box, 
               style='wireframe', 
               color='black', 
               line_width=1,
               opacity=0.5)
    
    if axis:
        plotter.show_axes()
    
    # plotter.add_legend()
    
    plotter.show()

    
def plot_volume_pdf(grid, domain ,s=None, coils=None, axis=True, nb=True, num_max_voxels=3,
                              save_filename=None, camera_position=None):
    """
    Function to plot sensors (red points) and coils (blue points) in a 3D grid.
    
    Parameters:
    - grid: The main 3D dataset to plot (e.g., a PyVista grid).
    - s (list of PyVista datasets, optional): Sensors to plot as red points.
    - coils (list of PyVista datasets, optional): Coils to plot as blue points.
    - axis (bool, default=True): Whether to show the axes.
    - nb (bool, default=True): Whether to use notebook mode (for Jupyter Notebook).
    - save_filename (str, optional): Filename to save the plot (e.g., "plot.png" or "plot.pdf").
    - camera_position (tuple or list, optional): Camera position for the plot 
      (e.g., [(x, y, z), (target_x, target_y, target_z), (up_x, up_y, up_z)]).
      
    Returns:
    - None
    """

    # Create the Plotter instance
    plotter = pv.Plotter(notebook=nb)

    # Add the main grid to the plot
    # Get scalar values
    values = grid.cell_data["values"]
    max_val = values.max()
    min_val = values.min()
    
    # Find threshold for top N voxels
    threshold = np.sort(values.flatten())[-num_max_voxels]
    thresholded = grid.threshold([threshold, max_val])
    
    # Create opacity mapping for thresholded volume
    thresholded_values = thresholded.cell_data["values"]
    normalized_values = (thresholded_values - threshold) / (max_val - threshold)
    opacity = 0.2 + 0.8 * normalized_values  # Range from 0.2 to 1.0
    
    # Add thresholded volume with value-dependent opacity
    plotter.add_mesh(thresholded, 
                   show_edges=True, 
                   cmap='viridis',
                   clim=[min_val, max_val],
                   opacity=opacity)  # Apply custom opacity
    
    # Highlight absolute maximum voxel
    max_voxel = grid.threshold([max_val - 1e-9, max_val])  # Small epsilon to catch max
    plotter.add_mesh(max_voxel,
                   color='yellow',
                   opacity=1.0,
                   edge_color='white',
                   line_width=3)
    
    # Add wireframe of full volume (transparent)
    plotter.add_mesh(grid, 
                   style='wireframe', 
                   color='gray', 
                   opacity=0.1,  # More transparent
                   line_width=0.5)
    




    # Add sensors (if provided)
    if s is not None:
        for l in range(len(s)):
            plotter.add_mesh(
                s[l],
                opacity=1,
                color='red',
                render_points_as_spheres=True,
                point_size=8
            )

    # Add coils (if provided)
    if coils is not None:
        for l in range(len(coils)):
            plotter.add_mesh(
                coils[l],
                opacity=1,
                color='blue',
                render_points_as_spheres=True,
                point_size=8
            )

    # Create and add the bounding box/cube
    bounds = [
        domain['x'][0], domain['x'][1],
        domain['y'][0], domain['y'][1], 
        domain['z'][0], domain['z'][1]]
    
    # Create a wireframe box representing the domain
    box = pv.Box(bounds=bounds)
    plotter.add_mesh(box, 
               style='wireframe', 
               color='black', 
               line_width=1,
               opacity=0.5)
    
    

    # Show or hide axes
    if axis:
        plotter.show_axes()

    # Set the camera position if provided
    if camera_position is not None:
        plotter.camera_position = camera_position 



    # Save the plot to a file if a filename is provided
    if save_filename is not None:
        # Extract the desired file extension
        if save_filename.lower().endswith('.png'):
            plotter.show(screenshot=save_filename)  # Save as PNG with high resolution
        elif save_filename.lower().endswith('.pdf'):
            plotter.scalar_bar.GetLabelTextProperty().SetFontSize(35)  # Increase font size
            plotter.scalar_bar.SetTitle(" ")
            plotter.scalar_bar.GetTitleTextProperty().SetFontSize(25)  # Increase title size
            plotter.save_graphic(save_filename) # Save as vectorized PDF
        else:
            raise ValueError("Unsupported file format. Use '.png' or '.pdf' for save_filename.")
    # # Customize the scalar bar (legend)
    
    # # plotter.scalar_bar.SetNumberOfLabels(5)  # Adjust number of labels
    # plotter.scalar_bar.GetLabelTextProperty().SetFontSize(35)  # Increase font size
    # plotter.scalar_bar.SetTitle(" ")
    # plotter.scalar_bar.GetTitleTextProperty().SetFontSize(25)  # Increase title size
    plotter.show()



def plot_volume_max_voxels(grid, domain, s=None, labels=None, coils=None, axis=True, nb=False, num_max_voxels=3):
    '''Plot with opacity scaled by values for thresholded volume
    
    Parameters:
    -----------
    grid : pyvista.UnstructuredGrid
        The volume grid to plot
    domain : dict
        Dictionary with domain boundaries {'x': [min, max], 'y': [min, max], 'z': [min, max]}
    s : list of pyvista.PolyData, optional
        List of sensor point clouds
    labels : list of str, optional
        List of labels for each sensor (must match length of s)
    coils : list of pyvista.PolyData, optional
        List of coil point clouds
    axis : bool, optional
        Whether to show axes
    nb : bool, optional
        Whether plotting in notebook
    num_max_voxels : int, optional
        Number of top voxels to highlight
    '''
    plotter = pv.Plotter(notebook=nb)
    
    # Get scalar values
    values = grid.cell_data["values"]
    max_val = values.max()
    min_val = values.min()
    
    # Find threshold for top N voxels
    threshold = np.sort(values.flatten())[-num_max_voxels]
    thresholded = grid.threshold([threshold, max_val])
    
    # Create opacity mapping for thresholded volume
    thresholded_values = thresholded.cell_data["values"]
    normalized_values = (thresholded_values - threshold) / (max_val - threshold)
    opacity = 0.2 + 0.8 * normalized_values  # Range from 0.2 to 1.0
    
    # Add thresholded volume with value-dependent opacity
    plotter.add_mesh(thresholded, 
                   show_edges=True, 
                   cmap='viridis',
                   clim=[min_val, max_val],
                   opacity=opacity)  # Apply custom opacity
    
    # Highlight absolute maximum voxel
    max_voxel = grid.threshold([max_val - 1e-9, max_val])  # Small epsilon to catch max
    plotter.add_mesh(max_voxel,
                   color='yellow',
                   opacity=1.0,
                   edge_color='white',
                   line_width=3)
    
    # Add wireframe of full volume (transparent)
    plotter.add_mesh(grid, 
                   style='wireframe', 
                   color='gray', 
                   opacity=0.1,  # More transparent
                   line_width=0.5)
    
    # Add sensors with labels
    if s is not None:
        for l in range(len(s)):
            # Add sensor point
            plotter.add_mesh(s[l],
                           opacity=1, 
                           color='red',
                           render_points_as_spheres=True, 
                           point_size=8)
            
            # Add label if labels are provided
            if labels is not None and l < len(labels):
                # Get the center point of the sensor
                center = s[l]
                plotter.add_point_labels(
                    center,
                    labels=[labels[l]],
                    font_size=19,
                    text_color='red',
                    shadow=True,
                    shape=None,  # No background shape
                    always_visible=True,
                    name=f"sensor_label_{l}"
                )
    
    # Add coils
    if coils is not None:
        for l in range(len(coils)):
            plotter.add_mesh(coils[l],
                           opacity=1, 
                           color='blue',
                           render_points_as_spheres=True, 
                           point_size=8)
    
    # Create and add the bounding box/cube
    bounds = [
        domain['x'][0], domain['x'][1],
        domain['y'][0], domain['y'][1], 
        domain['z'][0], domain['z'][1]]
    
    # Create a wireframe box representing the domain
    box = pv.Box(bounds=bounds)
    plotter.add_mesh(box, 
               style='wireframe', 
               color='black', 
               line_width=1,
               opacity=0.5)
    
    if axis:
        plotter.show_axes()
    
    plotter.show()



#for create a gift 


def create_rotation_gif(grid, domain, s=None, coils=None, axis=True, num_max_voxels=3,  output_file='rotation.gif', 
                       duration=100, resolution=(800, 600)):
    """
    Creates a rotating GIF of the 3D plot.
    
    Args:
        grid: PyVista dataset to visualize
        sensors: List of sensor meshes (optional)
        coils: List of coil meshes (optional)
        output_file: Output GIF filename
        duration: Frame duration in ms
        resolution: (width, height) of output
    """
    plotter = pv.Plotter(off_screen=True)  # Critical for headless rendering


    
    # Get scalar values
    values = grid.cell_data["values"]
    max_val = values.max()
    min_val = values.min()
    
    # Find threshold for top N voxels
    threshold = np.sort(values.flatten())[-num_max_voxels]
    thresholded = grid.threshold([threshold, max_val])
    
    # Create opacity mapping for thresholded volume
    thresholded_values = thresholded.cell_data["values"]
    normalized_values = (thresholded_values - threshold) / (max_val - threshold)
    opacity = 0.2 + 0.8 * normalized_values  # Range from 0.2 to 1.0
    
    # Add thresholded volume with value-dependent opacity
    plotter.add_mesh(thresholded, 
                   show_edges=True, 
                   cmap='viridis',
                   clim=[min_val, max_val],
                   opacity=opacity)  # Apply custom opacity
    
    # Highlight absolute maximum voxel
    max_voxel = grid.threshold([max_val - 1e-9, max_val])  # Small epsilon to catch max
    plotter.add_mesh(max_voxel,
                   color='yellow',
                   opacity=1.0,
                   edge_color='white',
                   line_width=3)
    
    # Add wireframe of full volume (transparent)
    plotter.add_mesh(grid, 
                   style='wireframe', 
                   color='gray', 
                   opacity=0.1,  # More transparent
                   line_width=0.5)
    
    # Add sensors
    if s is not None:
        for l in range(len(s)):
            plotter.add_mesh(s[l],
                           opacity=1, 
                           color='red',
                           render_points_as_spheres=True, 
                           point_size=8)
    
    # Add coils
    if coils is not None:
        for l in range(len(coils)):
            plotter.add_mesh(coils[l],
                           opacity=1, 
                           color='blue',
                           render_points_as_spheres=True, 
                           point_size=8)
    
 
    # Create and add the bounding box/cube
    bounds = [
        domain['x'][0], domain['x'][1],
        domain['y'][0], domain['y'][1], 
        domain['z'][0], domain['z'][1]]
    
    # Create a wireframe box representing the domain
    box = pv.Box(bounds=bounds)
    plotter.add_mesh(box, 
               style='wireframe', 
               color='black', 
               line_width=1,
               opacity=0.5)
    
    if axis:
        plotter.show_axes()


    
    # Open a temporary directory
    temp_dir = "\home\hernan\Pictures"
    os.makedirs(temp_dir, exist_ok=True)
    
    # Capture frames while rotating
    frame_files = []
    for i, angle in enumerate(range(0, 360, 5)):  # 10° steps (36 frames)
        plotter.camera_position = 'yz'
        plotter.camera.azimuth = angle  # Rotate around Z-axis
        frame_path = f"{temp_dir}/frame_{i:03d}.png"
        plotter.screenshot(frame_path, window_size=resolution)
        frame_files.append(frame_path)
    
    # Convert frames to GIF using PIL
    images = [Image.open(f) for f in frame_files]
    images[0].save(output_file, save_all=True, append_images=images[1:], 
                  duration=duration, loop=0)
    
    # Cleanup
    for f in frame_files:
        os.remove(f)
    os.rmdir(temp_dir)
    
    print(f"GIF saved to {output_file}")








def plot_point_figure(u_0, s,coordinates1):
    # Draw the solution
    # filename_rec = join(fig_dir, 'rec.png')
    cube = pv.PolyData(coordinates1)
    sensors = pv.PolyData(s)
    pl = pv.Plotter(notebook=False)
    pl.add_mesh(cube, scalars=u_0, opacity=np.clip(u_0, 0.8, 1), render_points_as_spheres=True, point_size=30)
    pl.add_mesh(sensors, render_points_as_spheres=True, point_size=10)
    pl.show_axes()
    pl.show()
    # pl.show(screenshot=filename_rec)
# plot_point_figure(cj, s)







def plot_volume_sensors_coils1(grid, s=None, coils=None, axis=True, nb=True, 
                              save_filename=None, camera_position=None):
    """
    Function to plot sensors (red points) and coils (blue points) in a 3D grid.
    
    Parameters:
    - grid: The main 3D dataset to plot (e.g., a PyVista grid).
    - s (list of PyVista datasets, optional): Sensors to plot as red points.
    - coils (list of PyVista datasets, optional): Coils to plot as blue points.
    - axis (bool, default=True): Whether to show the axes.
    - nb (bool, default=True): Whether to use notebook mode (for Jupyter Notebook).
    - save_filename (str, optional): Filename to save the plot (e.g., "plot.png" or "plot.pdf").
    - camera_position (tuple or list, optional): Camera position for the plot 
      (e.g., [(x, y, z), (target_x, target_y, target_z), (up_x, up_y, up_z)]).
      
    Returns:
    - None
    """

    # Create the Plotter instance
    plotter = pv.Plotter(notebook=nb)

    # Add the main grid to the plot
    plotter.add_mesh(grid, show_edges=True, opacity=0.9)

    # Add sensors (if provided)
    if s is not None:
        for l in range(len(s)):
            plotter.add_mesh(
                s[l],
                opacity=1,
                color='red',
                render_points_as_spheres=True,
                point_size=8
            )

    # Add coils (if provided)
    if coils is not None:
        for l in range(len(coils)):
            plotter.add_mesh(
                coils[l],
                opacity=1,
                color='blue',
                render_points_as_spheres=True,
                point_size=8
            )

    # Show or hide axes
    if axis:
        plotter.show_axes()

    # Set the camera position if provided
    if camera_position is not None:
        plotter.camera_position = camera_position 

    # Save the plot to a file if a filename is provided
    if save_filename is not None:
        # Extract the desired file extension
        if save_filename.lower().endswith('.png'):
            plotter.show(screenshot=save_filename)  # Save as PNG with high resolution
        elif save_filename.lower().endswith('.pdf'):
            plotter.scalar_bar.GetLabelTextProperty().SetFontSize(35)  # Increase font size
            plotter.scalar_bar.SetTitle(" ")
            plotter.scalar_bar.GetTitleTextProperty().SetFontSize(25)  # Increase title size
            plotter.save_graphic(save_filename) # Save as vectorized PDF
        else:
            raise ValueError("Unsupported file format. Use '.png' or '.pdf' for save_filename.")
    # # Customize the scalar bar (legend)
    
    # # plotter.scalar_bar.SetNumberOfLabels(5)  # Adjust number of labels
    # plotter.scalar_bar.GetLabelTextProperty().SetFontSize(35)  # Increase font size
    # plotter.scalar_bar.SetTitle(" ")
    # plotter.scalar_bar.GetTitleTextProperty().SetFontSize(25)  # Increase title size
    plotter.show()






def domain_mesh_2volumes(h,domain,cotas,nx=None,ny=None):
    '''This function returns the coordinates of the center of mass of doamin discretized by finite differences in size h, 
    the volume of each voxel and the mask of  the cotas of  cube of cuadricula form volume the volume that contains the concentration of the MN particles'''    
    ax,bx=domain['x']

    ay,by=domain['y']

    az,bz=domain['z']

    if nx == None:
      nx=2
    if ny == None:
      ny=2
  
    nz=int((bz-az)/h)+1
    dx = (bx-(ax))/(nx-1)
    dy = (by-(ay))/(ny-1)
    dz = (bz-(az))/(nz-1)

    # the coordinates of the center mass is rest a half of the size in order to obtain the half part 

    x = np.linspace(ax+dx/2, bx-dx/2, nx-1)
    y = np.linspace(ay+dy/2, by-dy/2, ny-1)
    z = np.linspace(az+dz/2, bz-dz/2, nz-1)

    ###source positions
    #coordinates1 = np.array([(a, b, c)  for a in x for b in y for c in z])
    xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
    coordinates = np.column_stack((xx.ravel(), yy.ravel(), zz.ravel()))

    nc=len(coordinates)
    #volume for every single volume
    V=(bx-ax)*(by-ay)*(bz-az)
    
    dV_x=V/nc


    eps = 1e-8 


    #mask by using cotas

    if cotas['cube']==False:
        radius=cotas['radius']
        center=np.array(cotas['center'])
        mask= (LA.norm((coordinates[:]-center),axis=1)<=radius+eps)
        

    else:
        x1,x2=cotas['x']
        y1,y2=cotas['y']
        z1,z2=cotas['z']
        mask= (((coordinates[:,0]+dx/2<=x2+eps)&(x1-eps<=coordinates[:,0]-dx/2))  & ((coordinates[:,1]+dy/2<=y2+eps)&(y1-eps<=coordinates[:,1]-dy/2)) & (coordinates[:,2]+dz/2<=z2+eps) & (z1-eps<=coordinates[:,2]-dz/2))

    return mask,coordinates,dV_x,nx,ny,nz

def domain_mesh_n(domain,cotas,nx=None,ny=None,nz=None):
    '''This function returns the coordinates of the center of mass of doamin discretized by finite differences in size h, 
    the volume of each voxel and the mask of  the cotas of  cube of cuadricula form volume the volume that contains the concentration of the MN particles'''    
    ax,bx=domain['x']

    ay,by=domain['y']

    az,bz=domain['z']

    if nx == None:
      nx=2
    if ny == None:
      ny=2
    if nz == None:
      nz=2
  
    # nz=int((bz-az)/h)+1
    dx = (bx-(ax))/(nx-1)
    dy = (by-(ay))/(ny-1)
    dz = (bz-(az))/(nz-1)

    # the coordinates of the center mass is rest a half of the size in order to obtain the half part 

    x = np.linspace(ax+dx/2, bx-dx/2, nx-1)
    y = np.linspace(ay+dy/2, by-dy/2, ny-1)
    z = np.linspace(az+dz/2, bz-dz/2, nz-1)

    ###source positions
    #coordinates1 = np.array([(a, b, c)  for a in x for b in y for c in z])
    xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
    coordinates = np.column_stack((xx.ravel(), yy.ravel(), zz.ravel()))

    nc=len(coordinates)

    #volume for every single volume
    V=(bx-ax)*(by-ay)*(bz-az)
    
    dV_x=V/nc


    eps = 5e-6


    #mask by using cotas

    if cotas['cube']==False:
        radius=cotas['radius']
        center=np.array(cotas['center'])
        mask= (LA.norm((coordinates[:]-center),axis=1)<=radius+eps)
        

    else:
        x1,x2=cotas['x']
        y1,y2=cotas['y']
        z1,z2=cotas['z']
        mask = (((coordinates[:,0]<=x2) & (x1<=coordinates[:,0])) &
                ((coordinates[:,1]<=y2) & (y1<=coordinates[:,1])) &
                ((coordinates[:,2]<=z2) & (z1<=coordinates[:,2])))
        # mask= (((coordinates[:,0]+dx/2<=x2+eps)&(x1-eps<=coordinates[:,0]-dx/2))  & ((coordinates[:,1]+dy/2<=y2+eps)&(y1-eps<=coordinates[:,1]-dy/2)) & (coordinates[:,2]+dz/2<=z2+eps) & (z1-eps<=coordinates[:,2]-dz/2))

    return mask,coordinates,dV_x,nx,ny,nz


def grid_volume_n(cj,domain,nx=None,ny=None,nz=None):
  "for plotting the domain"
  ax,bx=domain['x']

  ay,by=domain['y']

  az,bz=domain['z']
  #Fucion para graficar
  if nx == None:
      nx=2
  if ny == None:
      ny=2
  if nz == None:
      nz=2
  #   nz=int((bz-az)/h)+1
  dx = (bx-(ax))/(nx-1)
  dy = (by-(ay))/(ny-1)
  dz = (bz-(az))/(nz-1)

  matrixcjexa = np.reshape(cj, (nx-1, ny-1, nz-1))

  values = matrixcjexa
  values.shape

  # Create the spatial reference
  grid = pv.ImageData()

  # Set the grid dimensions: shape + 1 because we want to inject our values on
  #   the CELL data
  grid.dimensions = np.array(values.shape)+1

  # Edit the spatial reference
  grid.origin = (ax,ay,az)  # The bottom left corner of the data set
  grid.spacing = (dx, dy, dz)  # These are the cell sizes along each axis

  # Add the data values to the cell data
  grid.cell_data["values"] = values.flatten(order="F")  # Flatten the array
  
  return grid




def grid_volume_2volumes(cj,h,domain,nx=None,ny=None):
  "for plotting the domain"
  ax,bx=domain['x']

  ay,by=domain['y']

  az,bz=domain['z']
  #Fucion para graficar
  if nx == None:
      nx=2
  if ny == None:
      ny=2
  nz=int((bz-az)/h)+1
  dx = (bx-(ax))/(nx-1)
  dy = (by-(ay))/(ny-1)
  dz = (bz-(az))/(nz-1)

  matrixcjexa = np.reshape(cj, (nx-1, ny-1, nz-1))

  values = matrixcjexa
  values.shape

  # Create the spatial reference
  grid = pv.ImageData()

  # Set the grid dimensions: shape + 1 because we want to inject our values on
  #   the CELL data
  grid.dimensions = np.array(values.shape)+1

  # Edit the spatial reference
  grid.origin = (ax,ay,az)  # The bottom left corner of the data set
  grid.spacing = (dx, dy, dz)  # These are the cell sizes along each axis

  # Add the data values to the cell data
  grid.cell_data["values"] = values.flatten(order="F")  # Flatten the array
  
  return grid


def plot_volume_sensors_coils(grid,s=None,coils = None ,axis=True,nb=True):
    '''Fucion para graficar los coils and the sensors inpyu'''
    plotter = pv.Plotter(notebook=nb)
    plotter.add_mesh(grid,show_edges=True, opacity=0.9)
    if s is not None:
        for l in range(len(s)):
            plotter.add_mesh(s[l],opacity=1, color='red',render_points_as_spheres=True, point_size=8)
    if coils is not None:
        for l in range(len(coils)):
            plotter.add_mesh(coils[l],opacity=1, color='blue',render_points_as_spheres=True, point_size=8)
    if axis==True:
        plotter.show_axes()
    plotter.show()


def increase_angles_matrix(G_global1_cs):
    # compute the all angles between all rows
    angles = np.zeros((G_global1_cs.shape[0], G_global1_cs.shape[0]))
    norm_G = np.linalg.norm(G_global1_cs, axis=1)
    #for choose the first two rows
    for i in range(G_global1_cs.shape[0]):
        for j in range(i+1, G_global1_cs.shape[0]):
            dot_product = np.dot(G_global1_cs[i], G_global1_cs[j])
            norm_i = norm_G[i] #np.linalg.norm(G_global1_cs[i])
            norm_j = norm_G[j] #np.linalg.norm(G_global1_cs[j])
            # print(dot_product,(norm_i * norm_j))
            angles[i, j] = np.arccos(dot_product / (norm_i * norm_j))
            angles[j, i] = angles[i, j]

    # print(angles)
    # we get the first index in the matrix  where we get max value of the angle, it starts with
    indices = [int(np.where(angles==np.max(angles))[0][0]),int(np.where(angles==np.max(angles))[0][1])]
  
    # result = G_global1_cs[indices]
    # Initialize the set of remaining rows
    remaining_rows = set(range(0, G_global1_cs.shape[0]))
    remaining_rows.remove(indices[0])
    remaining_rows.remove(indices[1])
    
    # print(norm_G)
    # While there are still remaining rows
    while remaining_rows:
        # Calculate the angle with the last row in the result
        # last_row_idx = indices[-1]
        B_sum = np.sum(G_global1_cs[indices], axis=0)
        angles_with_last_row = [np.arccos((np.dot(B_sum,G_global1_cs[i]))/(np.linalg.norm(B_sum)*norm_G[i]))  for i in remaining_rows]
        # print(angles_with_last_row)
        # Choose the row with the maximum angle
        max_angle_idx = np.argmax(angles_with_last_row)
        # print(max_angle_idx)
        max_angle_row_idx = list(remaining_rows)[max_angle_idx]
        
        # Add the row with the maximum angle to the result
        # result = np.concatenate((result, G_global1_cs[max_angle_row_idx:max_angle_row_idx+1]), axis=0)
        indices.append(max_angle_row_idx)
        # print(indices)
        # Remove the row from the set of remaining rows
        remaining_rows.remove(max_angle_row_idx)
    return indices


