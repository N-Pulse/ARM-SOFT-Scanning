# ================================================================
# 0. Section: Imports
# ================================================================
import numpy as np



# ================================================================
# 1. Section: PCA for Object Orientation
# ================================================================
def get_object_orientation(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the centroid and principal axes of a foreground region in a binary mask
    by performing PCA on the coordinates of non-zero pixels.

    Parameters
    ----------
    mask : np.ndarray
        Binary/boolean mask of arbitrary dimensionality. Non-zero (or True)
        elements are treated as foreground. Coordinates are collected using
        ``np.nonzero``, so the coordinate order is (row, column, ...) for 2-D
        images and follows the array axis order for higher dimensions.

    Returns
    -------
    centroid : np.ndarray
        The mean coordinate of the foreground pixels. Shape is (D,), where D is
        the number of dimensions of ``mask`` (i.e., ``mask.ndim``). Coordinates
        are in the same axis order as produced by ``np.nonzero``.
    eigvals : np.ndarray
        Eigenvalues of the covariance matrix of the centered coordinates,
        sorted in decreasing order. Shape is (D,).
    eigvecs : np.ndarray
        Eigenvectors of the covariance matrix as columns. Shape is (D, D).
        Column i corresponds to the eigenvector associated with eigvals[i].

    Raises
    ------
    None
        This function does not explicitly raise on empty foreground. If no
        foreground pixels are present, NumPy may emit runtime warnings and the
        returned arrays will contain NaNs.

    Examples
    --------
    >>> import numpy as np
    >>> # 2D rectangular region
    >>> mask = np.zeros((100, 100), dtype=bool)
    >>> mask[40:60, 45:65] = True
    >>> centroid, eigvals, eigvecs = get_object_orientation(mask)
    >>> centroid.shape
    (2,)
    >>> eigvecs.shape
    (2, 2)

    Notes
    -----
    - The PCA is performed on integer pixel coordinates (row, column, ...).
    - Eigenvalues are returned in descending order; corresponding eigenvectors
      are provided as the columns of ``eigvecs``.
    - For consistent geometric interpretation, remember that the first coordinate
      axis corresponds to the array row index (vertical image axis) and the
      second to the column index (horizontal image axis).
    """
    # 1. Get coordinates of foreground pixels
    coords = np.column_stack(np.nonzero(mask))

    # 2. Compute centroid and center data
    centroid = coords.mean(axis=0)
    X = coords - centroid

    # 3. Covariance matrix and Eigen decomposition
    C = np.cov(X, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(C) 

    # 4. Sort by decreasing eigenvalue
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    return centroid, eigvals, eigvecs