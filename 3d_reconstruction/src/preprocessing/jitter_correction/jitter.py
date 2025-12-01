# ================================================================
# 0. Section: Imports
# ================================================================
import os

import numpy as np
import matplotlib.pyplot as plt

from functools import singledispatch
from copy import deepcopy

from .object_pca import get_object_orientation
from ...scans import Scans, Scan



# ================================================================
# 1. Section: Jitter Correction via PCA Alignment
# ================================================================
@singledispatch
def fix_jitter(mask):
    """
    Align masks or collections of masks to their principal axes to correct jitter.

    This is a generic function with multiple implementations registered via
    ``functools.singledispatch``:

    - ``fix_jitter(mask: np.ndarray)``:
        Align a single 2D binary/boolean mask using PCA.
        Computes a PCA-based alignment (rotation) around the mask centroid,
        applies the inverse rotation to a centered sampling grid, and resamples
        the input mask using nearest-neighbour interpolation to produce an
        aligned mask.

    - ``fix_jitter(mask: Scans)``:
        Align all scans in a ``Scans`` collection. Each scan's mask is aligned
        as above, optionally saved to disk for inspection, and a new
        ``Scans`` object containing the aligned masks is returned.

    Parameters
    ----------
    mask : np.ndarray or Scans
        - If ``np.ndarray``: a 2D array of shape ``(H, W)``. Non-zero or
          True values are treated as foreground pixels used to compute the
          PCA alignment. The array can be of any numeric or boolean dtype.
        - If ``Scans``: a container of scan objects, each providing a 2D
          mask (e.g. via ``scan.scan``). The collection is iterated and
          each mask is aligned individually.

    Returns
    -------
    For ``mask: np.ndarray``:
        mask_aligned : np.ndarray
            The resampled mask after PCA-based alignment. The output has the
            same shape ``(H, W)`` as the input.
        centroid : tuple[float, float]
            The computed centroid of the input mask in (row, column)
            coordinates (i.e. (y, x)). Values are floating-point and refer to
            the original mask coordinate system.
        eigvecs : np.ndarray
            A ``(2, 2)`` array containing the principal eigenvectors of the
            mask covariance. Columns correspond to principal axes (sorted by
            descending eigenvalue), useful to recover the rotation used for
            alignment.

    For ``mask: Scans``:
        aligned_scans : Scans
            A new ``Scans`` instance containing the aligned masks (and
            associated positions or metadata). The individual aligned masks
            may also be saved to disk for visualization.

    Raises
    ------
    TypeError
        If ``mask`` is of an unsupported type (i.e. not handled by any
        registered implementation).
    ValueError
        If the NumPy mask does not have exactly two dimensions.

    Examples
    --------
    Single mask
    ^^^^^^^^^^^
    >>> import numpy as np
    >>> mask = np.zeros((100, 150), dtype=bool)
    >>> mask[40:60, 70:100] = True
    >>> aligned, centroid, eigvecs = fix_jitter(mask)
    >>> aligned.shape
    (100, 150)

    Collection of scans
    ^^^^^^^^^^^^^^^^^^^
    >>> scans = Scans(...)  # collection of Scan objects
    >>> aligned_scans = fix_jitter(scans)

    Notes
    -----
    For the NumPy implementation:

    - The algorithm:
        1. Computes PCA on foreground pixels to obtain rotation angle,
           principal eigenvectors, and centroid.
        2. Generates a centered sampling grid for the mask dimensions.
        3. Applies the inverse rotation about the centroid to the grid.
        4. Samples the input mask at rotated coordinates using nearest
           neighbour interpolation.

    - The function returns the centroid and eigenvectors to allow further
      analysis or inverse transformations if needed.

    - Nearest-neighbour sampling is used to preserve binary mask values;
      if smoother interpolation is desired, resampling must be changed
      accordingly.
    """
    raise TypeError("Input must be a 2D NumPy array or a Scans or a Scan.")

@fix_jitter.register
def _(mask: np.ndarray):
    H, W = mask.shape

    # Get PCA Alignment Parameters
    theta, eigvecs, centroid = get_pca_alignment(mask)

    # Generate Centered Grid    
    grid = generate_grid(H, W)

    # Inverse Rotation
    rotated_grid = apply_inverse_rotation(theta, grid, centroid)

    # NN Sampling
    mask_aligned = nearest_neighbour_sampling(mask, rotated_grid, H, W)

    return mask_aligned, centroid, eigvecs

@fix_jitter.register
def _(mask: Scans):
    aligned_masks = []
    aligned_positions = []
    scans = mask.scans
    for idx, scan in enumerate(scans):
        masked = scan.scan
        aligned_mask, new_centroid, new_eigvecs = fix_jitter(masked)

        folder_path = "./figures/demo/aligned/"
        os.makedirs(folder_path, exist_ok=True)
        file_path = os.path.join(folder_path, f"aligned_scan_{scan.position:03d}.jpg")

        plt.imsave(file_path, aligned_mask, cmap='gray')
        aligned_masks.append(Scan(aligned_mask, scan.position))
        aligned_positions.append(scan.position)

        print(f"    ✔ Saved aligned scan {scan.position} to {file_path} ({idx + 1}/{len(scans)})")

    aligned_scans = Scans(np.array(aligned_masks), np.array(aligned_positions))
    print("✅ All aligned scans saved.")

    return aligned_scans

# ──────────────────────────────────────────────────────
# 1.1 Subsection: PCA Alignment to Image Axes
# ──────────────────────────────────────────────────────
def get_pca_alignment(mask: np.ndarray) -> tuple:
    centroid, eigvals, eigvecs = get_object_orientation(mask)

    # Principal axis as (y, x)
    v = eigvecs[:, 0]
    vx, vy = v
    vx = -vx 

    theta = np.arctan2(vy, vx)

    return theta, eigvecs, centroid


# ──────────────────────────────────────────────────────
# 1.2 Subsection: Generate Grid Centered at Given Point
# ──────────────────────────────────────────────────────
def generate_grid(H: int, W: int) -> tuple:
    cy_im = (H - 1) / 2.0
    cx_im = (W - 1) / 2.0

    Y, X = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')

    x0 = X - cx_im
    y0 = Y - cy_im

    return x0, y0


# ──────────────────────────────────────────────────────
# 1.3 Subsection: Inverse Rotation
# ──────────────────────────────────────────────────────
def apply_inverse_rotation(theta: float, grid: tuple, centroid: tuple) -> tuple:
    # Unpacks the data
    x0, y0 = grid
    cy, cx = centroid

    # Inverse rotation: map output -> input
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    # [x_rel; y_rel] = R(theta) * [x0; y0]
    x_rel = cos_t * x0 - sin_t * y0
    y_rel = sin_t * x0 + cos_t * y0

    # Add original centroid to get input coords
    x_in = x_rel + cx
    y_in = y_rel + cy

    return x_in, y_in


# ──────────────────────────────────────────────────────
# 1.4 Subsection: NN Sampling
# ──────────────────────────────────────────────────────
def nearest_neighbour_sampling(mask: np.ndarray, rotated_grid: tuple, H: int, W: int) -> np.ndarray:
    x_in, y_in = rotated_grid

    # Nearest-neighbor sampling
    x_in_round = np.round(x_in).astype(int)
    y_in_round = np.round(y_in).astype(int)

    mask_aligned = np.zeros_like(mask, dtype=bool)
    inside = (
        (x_in_round >= 0) & (x_in_round < W) &
        (y_in_round >= 0) & (y_in_round < H)
    )

    mask_aligned[inside] = mask[y_in_round[inside], x_in_round[inside]]

    return mask_aligned

