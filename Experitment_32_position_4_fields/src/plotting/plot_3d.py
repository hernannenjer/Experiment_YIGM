import logging
import numpy as np
import pyvista as pv

# def plot_3d_from_array(data_array,
#                        point_size=10,
#                        cmap='viridis',
#                        show=True,
#                        rec=None,
#                        gt=None,
#                        point_label_fs=0,
#                        use_convex_hull=False):
#     """
#     Plots 3D points from a pre-assembled array of shape (N, 5),
#     where each row is: [index, X, Y, Z, value].
# 
#     Arguments:
#         data_array (np.ndarray): Must be of shape (N, 5) -> [idx, X, Y, Z, value].
#         point_size (int): Sphere size for points.
#         cmap (str): Colormap name.
#         show (bool): Whether to show the PyVista window.
#         rec (array-like or None): Additional points to show in red (optional).
#         gt (array-like or None): Additional points to show in green (optional).
#         point_label_fs (int): Font size for labeling points; if 0 or less, no labels.
#         use_convex_hull (bool): If True, use a convex hull for mesh.
#                                 If False, use 3D Delaunay + extract_surface.
# 
#     Returns:
#         (indices, coords, values): indices -> (N, ), coords -> (N, 3), values -> (N, )
#     """
#     if data_array.shape[1] != 5:
#         logging.error("data_array must have 5 columns: [index, X, Y, Z, value].")
#         return None, None, None
# 
#     # Extract columns
#     indices = data_array[:, 0].astype(int)
#     coords = data_array[:, 1:4]  # X, Y, Z
#     values = data_array[:, 4]
# 
#     # Build a PyVista PolyData
#     point_cloud = pv.PolyData(coords)
#     point_cloud['value'] = values
# 
#     # Choose how to build the surface
#     if use_convex_hull:
#         # Build a convex hull around the points (all outer faces)
#         surface = point_cloud.convex_hull()
#         # convex_hull() already returns a surface mesh
#         surface['value'] = values
#     else:
#         # 3D Delaunay (tetrahedrons), then extract outer surface
#         volume_mesh = point_cloud.delaunay_3d()
#         surface = volume_mesh.extract_surface()
#         surface['value'] = values
# 
#     # Create a Plotter and add meshes
#     plotter = pv.Plotter()
#     plotter.add_mesh(surface, scalars='value', cmap=cmap, opacity=1)
# 
#     # Also show the point cloud as spheres
#     plotter.add_mesh(point_cloud,
#                      scalars='value',
#                      point_size=point_size,
#                      render_points_as_spheres=True,
#                      cmap=cmap)
# 
#     plotter.add_axes()
# 
#     # Optional: plot 'rec' and 'gt' points
#     if rec is not None:
#         sphere_rec = pv.PolyData(np.array(rec))
#         plotter.add_mesh(sphere_rec, color='red', point_size=30, render_points_as_spheres=True)
# 
#     if gt is not None:
#         sphere_gt = pv.PolyData(np.array(gt))
#         plotter.add_mesh(sphere_gt, color='green', point_size=30, render_points_as_spheres=True)
# 
#     # If labeling is requested
#     if point_label_fs > 0:
#         labels = [str(idx) for idx in indices]
#         plotter.add_point_labels(coords,
#                                  labels,
#                                  show_points=False,
#                                  font_size=point_label_fs,
#                                  name="point_labels")
# 
#     # Show plot
#     if show:
#         plotter.show()
# 
#     return indices, coords, values

# src/plotting/plot_3d.py  (������� plot_3d_from_array)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# from typing import Optional
# import numpy as np
# import pyvista as pv
# 
# 
# def plot_3d_from_array(  # noqa: C901
#     data_array,
#     show: bool = True,
#     title: Optional[str] = None,
#     clim: Optional[tuple] = None,
#     cmap: str = "viridis",
#     point_size: float = 10.0,
#     add_axes: bool = True,
#     plotter: Optional[pv.Plotter] = None,
#     save_path: Optional[str] = None,
# ) -> pv.Plotter:
#     # --- input check --------------------------------------------------
#     arr = np.asarray(data_array, dtype=np.float32)
#     if arr.ndim != 2 or arr.shape[1] != 5:
#         raise ValueError("data_array must have shape (N, 5)")
# 
#     indices = arr[:, 0].astype(int)
#     coords = arr[:, 1:4]
#     values = arr[:, 4]
#     clim = clim or (float(values.min()), float(values.max()))
# 
#     # --- plotter ------------------------------------------------------
#     if plotter is None:
#         off = (save_path is not None) and (not show)
#         pl = pv.Plotter(off_screen=off)
#     else:
#         pl = plotter
# 
#     # --- point cloud --------------------------------------------------
#     cloud = pv.PolyData(coords)
#     cloud["value"] = values
#     pl.add_mesh(
#         cloud,
#         scalars="value",
#         cmap=cmap,
#         clim=clim,
#         render_points_as_spheres=True,
#         point_size=point_size,
#         name="points",
#     )
# 
#     # --- surface reconstruction --------------------------------------
#     try:
#         volume = cloud.delaunay_3d()           # no alpha -> more robust
#         surface = volume.extract_surface()
# 
#         # interpolate transfers 'value' from cloud to surface vertices
#         surface = surface.interpolate(cloud)
# 
#         pl.add_mesh(
#             surface,
#             scalars="value",
#             cmap=cmap,
#             clim=clim,
#             opacity=1,
#             show_scalar_bar=False,
#             name="surface",
#         )
#     except Exception as exc:
#         print(f"Surface reconstruction failed: {exc}")
# 
#     # --- decorations --------------------------------------------------
#     if title:
#         pl.add_text(title, position="upper_edge")
#     if add_axes:
#         pl.show_axes()
# 
#     # --- show / save --------------------------------------------------
#     if save_path:
#         if show:
#             pl.show(screenshot=save_path, auto_close=plotter is None)
#         else:
#             pl.screenshot(save_path)
#             if plotter is None:
#                 pl.close()
#     else:
#         if show and plotter is None:
#             pl.show()
# 
#     return pl

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_3d_from_array - now with camera presets

New keyword
-----------
view : str, optional
    "iso" (default) | "xy" | "xz" | "yz"
    Sets the camera so the user sees the chosen plane.
"""

from typing import Optional
import numpy as np
import pyvista as pv


# def _apply_camera(p: pv.Plotter, view: str) -> None:
#     """Internal helper - align camera to a principal plane."""
#     view = view.lower()
#     if view == "xy":
#         p.view_xy()
#     elif view == "xz":
#         p.view_xz()
#     elif view == "yz":
#         p.view_yz()
#     else:                       # any unknown string -> iso
#         p.view_isometric()

# ----------------------------------------------------------------------
# Camera presets   (all-ASCII comments, safe for any locale)
# ----------------------------------------------------------------------
from typing import Tuple, Union
import numpy as np
import pyvista as pv

Vec3  = Tuple[float, float, float]
CamPos = Tuple[Vec3, Vec3, Vec3]          # (pos, focal, up)


def _safe_view_up(cam: pv.Camera) -> Vec3:
    """Return view-up using VTK API - works on any PyVista version."""
    try:
        # Modern PyVista (�0.41) exposes .view_up
        return tuple(cam.view_up)          # type: ignore[attr-defined]
    except AttributeError:
        # Older versions: VTK getter
        return tuple(cam.GetViewUp())


def _apply_camera(pl: pv.Plotter, view: Union[str, CamPos, None]) -> None:
    """
    Align the camera to a principal plane or to a custom position.

    Parameters
    ----------
    pl   : pyvista.Plotter
    view : str | CamPos | None
        * str   -  \"iso\", \"xy\", \"xy-\", \"xz\", \"xz-\", \"yz\", \"yz-\".
        * tuple -  (pos, focal, up)     -> forwarded to camera_position.
        * None  -  same as \"iso\".
    """
    # ------------------------------------------------------------------
    # Custom camera position (3x3 tuple) -------------------------------
    # ------------------------------------------------------------------
    if isinstance(view, tuple):
        if len(view) == 3 and all(len(v) == 3 for v in view):  # rudimentary check
            pl.camera_position = view                           # type: ignore[arg-type]
        else:
            pl.view_isometric()
        return

    # ------------------------------------------------------------------
    # String presets ----------------------------------------------------
    # ------------------------------------------------------------------
    name = (view or "iso").lower()

    # Map code to PyVista method + sign for flipping
    preset = {
        "iso":  ("view_isometric", +1),
        "xy":   ("view_xy",        +1),
        "xy-":  ("view_xy",        -1),
        "xz":   ("view_xz",        +1),
        "xz-":  ("view_xz",        -1),
        "yz":   ("view_yz",        +1),
        "yz-":  ("view_yz",        -1),
    }

    if name not in preset:
        pl.view_isometric()
        return

    method_name, sign = preset[name]
    getattr(pl, method_name)()            # call view_xy / view_xz / view_yz / iso

    # Flip to the opposite side if sign is negative
    if sign < 0:
        cam = pl.camera
        fp  = np.asarray(cam.focal_point)
        pos = np.asarray(cam.position)
        up  = np.asarray(_safe_view_up(cam))

        # Mirror the camera position through the focal point
        cam.position = tuple(fp - (pos - fp))
        cam.focal_point = tuple(fp)
        cam.view_up = tuple(-up)
        pl.camera = cam

def plot_3d_from_array(  # noqa: C901
    data_array,
    show: bool = True,
    title: Optional[str] = None,
    clim: Optional[tuple] = None,
    cmap: str = "viridis",
    point_size: float = 10.0,
    add_axes: bool = True,
    plotter: Optional[pv.Plotter] = None,
    save_path: Optional[str] = None,
    view: str = "iso",          # <�� NEW PARAMETER
) -> pv.Plotter:
    """Visualise an (N, 5) array - now with camera presets."""
    arr = np.asarray(data_array, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != 5:
        raise ValueError("data_array must have shape (N, 5)")

    coords = arr[:, 1:4]
    values = arr[:, 4]
    clim = clim or (float(values.min()), float(values.max()))

    # create or reuse plotter
    if plotter is None:
        off = (save_path is not None) and (not show)
        pl = pv.Plotter(off_screen=off)
    else:
        pl = plotter

    # points
    cloud = pv.PolyData(coords)
    cloud["value"] = values
    pl.add_mesh(
        cloud,
        scalars="value",
        clim=clim,
        cmap=cmap,
        render_points_as_spheres=True,
        point_size=point_size,
        name="points",
    )

    # surface (same code as your latest working version)
    try:
        surface = cloud.delaunay_3d().extract_surface().interpolate(cloud)
        pl.add_mesh(
            surface,
            scalars="value",
            clim=clim,
            cmap=cmap,
            opacity=1,
            show_scalar_bar=False,
            name="surface",
        )
    except Exception as exc:
        print(f"Surface reconstruction failed: {exc}")

    # decorations
    if title:
        pl.add_text(title, position="upper_edge")
    if add_axes:
        pl.show_axes()

    # apply camera preset
    _apply_camera(pl, view)

    # display / save
    if save_path:
        if show:
            pl.show(screenshot=save_path, auto_close=plotter is None)
        else:
            pl.screenshot(save_path)
            if plotter is None:
                pl.close()
    else:
        if show and plotter is None:
            pl.show()

    return pl
def plot_3d_peak_heights(experiment,
                         coords_filename,
                         N_peaks=1,
                         point_size=10,
                         cmap='viridis',
                         show=True,
                         rec=None,
                         gt=None,
                         point_label_fs=0,
                         filename=None,
                         use_convex_hull=False):
    """
    Gathers data from 'experiment' and a coordinates file (with 4 columns:
    [point_index, X, Y, Z]). Then it computes the first peak amplitude for each sample
    and creates a 5-column array [index, X, Y, Z, value], where 'value' is abs(peak_height).
    If 'filename' is not None, the array is saved in that file as text.

    Finally, it calls 'plot_3d_from_array()' to visualize the results.

    Arguments:
        experiment: An Experiment object containing multiple samples.
        coords_filename: Path to a file with 4 columns: [point_index, X, Y, Z].
        N_peaks (int): How many peaks to search for per sample.
        point_size (int): Size of spheres for points in the plot.
        cmap (str): Colormap name.
        show (bool): Whether to show the plot.
        rec (array-like or None): Additional points to mark in red.
        gt (array-like or None): Additional points to mark in green.
        point_label_fs (int): Font size for labeling points (0 -> no labels).
        filename (str or None): If not None, saves the array [index, X, Y, Z, value].
        use_convex_hull (bool): If True, draws the convex hull. Otherwise uses 3D Delaunay.

    Returns:
        sample_indices (np.ndarray), peak_heights (np.ndarray) or (None, None) if error.
    """

    # Attempt to load coordinates
    try:
        coords_raw = np.loadtxt(coords_filename)
        logging.info(f"Loaded {coords_raw.shape[0]} lines from '{coords_filename}'.")
    except Exception as e:
        logging.error(f"Failed to load coordinates: {e}")
        return None, None

    # coords_raw should have shape (N_s, 4): [point_index, X, Y, Z]
    # Build a dict: sample_idx -> (X, Y, Z)
    coords_dict = {}
    for row in coords_raw:
        point_idx = int(row[0])
        x, y, z = row[1], row[2], row[3]
        coords_dict[point_idx] = (x, y, z)

    sample_indices = []
    peak_heights = []
    coords_list = []

    # Collect data from the experiment
    for sample_name, sample in experiment.samples.items():
        try:
            sample_idx = int(sample_name)
        except ValueError:
            logging.warning(f"Sample name '{sample_name}' is not an integer; skipping.")
            continue

        if sample_idx not in coords_dict:
            logging.warning(f"No coordinates found for sample index {sample_idx}; skipping.")
            continue

        # Make sure the sample has peak data
        if not hasattr(sample, 'mean_peaks') or sample.mean_peaks is None:
            sample.perform_multiexponential_analysis_all_attempts()
            sample.extract_and_analyze_pronounced_peaks(N_peaks=N_peaks)

        if sample.mean_peaks is not None:
            peak_height = sample.mean_peaks[0, 0]
            sample_indices.append(sample_idx)
            peak_heights.append(abs(peak_height))
            coords_list.append(coords_dict[sample_idx])
        else:
            logging.warning(f"No pronounced peaks found for sample '{sample_name}'.")

    if not peak_heights:
        logging.warning("No peak heights to plot.")
        return None, None

    sample_indices = np.array(sample_indices, dtype=int)
    peak_heights = np.array(peak_heights, dtype=float)
    coords_to_plot = np.array(coords_list, dtype=float)  # shape (N, 3)

    # Build the 5-column array: [index, X, Y, Z, value]
    data_5col = np.column_stack((sample_indices, coords_to_plot, peak_heights))

    # If a filename was specified, save the array
    if filename is not None:
        header_line = "index X Y Z value"
        try:
            np.savetxt(filename, data_5col, header=header_line, fmt="%.6g")
            logging.info(f"Saved data to '{filename}' in format: {header_line}")
        except Exception as e:
            logging.error(f"Failed to save '{filename}': {e}")

    # Now call the universal plotting function
    plot_3d_from_array(
        data_array=data_5col,
        point_size=point_size,
        cmap=cmap,
        show=show,
        rec=rec,
        gt=gt,
        point_label_fs=point_label_fs,
        use_convex_hull=use_convex_hull
    )

    return sample_indices, peak_heights
