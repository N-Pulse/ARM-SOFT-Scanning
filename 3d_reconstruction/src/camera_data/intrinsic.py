# ================================================================
# 0. Section: Imports
# ================================================================
import json

import numpy as np

from functools import singledispatchmethod



# ================================================================
# 1. Section: Intrinsic Parameters Class
# ================================================================
class IntrinsicParameters:
    """
    A flexible container for a 3×3 camera intrinsic matrix.

    This class supports initialization from multiple data sources
    using `functools.singledispatchmethod`:
    - A NumPy array (`np.ndarray`)
    - A nested list (`list[list[float]]`)
    - A JSON file path (`str`) containing a key `"K"`

    Parameters
    ----------
    data : np.ndarray | list | str
        The source for the intrinsic matrix.
        - If `np.ndarray`: used directly (rounded to 3 decimals)
        - If `list`: converted to `np.ndarray`
        - If `str`: interpreted as a path to a JSON file with a key `"K"`

    Attributes
    ----------
    matrix : np.ndarray
        The intrinsic calibration matrix of shape (3, 3).

    Raises
    ------
    TypeError
        If the input type is not supported.
    ValueError
        If the matrix shape is not (3, 3).

    Examples
    --------
    >>> import numpy as np
    >>> K = np.array([[1000, 0, 640], [0, 1000, 360], [0, 0, 1]])
    >>> params = IntrinsicParameters(K)
    >>> params.matrix
    array([[1000.,    0.,  640.],
           [   0., 1000.,  360.],
           [   0.,    0.,    1.]])

    >>> params = IntrinsicParameters([[1000, 0, 640], [0, 1000, 360], [0, 0, 1]])
    >>> params.matrix.shape
    (3, 3)

    >>> # From JSON file
    >>> params = IntrinsicParameters("intrinsics.json")

    Notes
    -----
    - If initialized with a NumPy array, values are rounded to 3 decimals.
    - The JSON file must contain a key `"K"` whose value is a 3×3 list.
    - Accessing or modifying `matrix` enforces the correct shape.
    """

    @singledispatchmethod
    def __init__(self, data) -> None:
        raise TypeError(f"Unsupported init type: {type(data)!r}")
    
    @__init__.register
    def _(self, matrix: np.ndarray) -> None:
        self.matrix = np.round(matrix, 3)

    @__init__.register
    def _(self, matrix: list) -> None:
        self.matrix = np.array(matrix)

    @__init__.register
    def _(self, path: str) -> None:
        if path.endswith('.json'):
            with open(path, 'r') as f:
                data = json.load(f)
            self.matrix = np.array(data['K'])
        elif path.endswith('.txt'): 
            data = np.loadtxt(path)
            self.matrix = data
        else:
            raise ValueError("Unsupported file format. Use a .json or .txt file.")
            

    
    @property
    def matrix(self) -> np.ndarray:
        return self._matrix
    
    @matrix.setter
    def matrix(self, value: np.ndarray) -> None:
        if value.shape != (3, 3):
            raise ValueError("Intrinsic matrix must be of shape (3, 3)")
        self._matrix = value