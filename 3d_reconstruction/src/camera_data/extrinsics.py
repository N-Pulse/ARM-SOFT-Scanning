# ================================================================
# 0. Section: Imports
# ================================================================
import json

import numpy as np

from functools import singledispatchmethod
from ..algebra import invert_extrinsic_matrix



# ================================================================
# 1. Section: Extrinsic Parameters Class
# ================================================================
class ExtrinsicParameters:
    """
    A flexible container for camera extrinsic parameters (pose matrix and position).
    This class supports initialization from multiple data sources
    using `functools.singledispatchmethod`:
    - A NumPy array (`np.ndarray`) with optional position
    - A nested list (`list[list[float]]`) with optional position

    Parameters
    ----------
    data : np.ndarray | list
        The source for the extrinsic matrix.
        - If `np.ndarray`: used directly as 4×4 transformation matrix
        - If `list`: converted to `np.ndarray`
    position : int, optional
        The camera position identifier. If not provided, defaults to None.

    Attributes
    ----------
    matrix : np.ndarray
        The extrinsic transformation matrix of shape (4, 4) representing
        the camera pose in world coordinates.
    position : int | None
        The camera position identifier. None if no position was specified.

    Raises
    ------
    TypeError
        If the input type is not supported or position is not an integer.
    ValueError
        If the matrix shape is not (4, 4).

    Examples
    --------
    >>> import numpy as np
    >>> # 4x4 transformation matrix
    >>> T = np.eye(4)
    >>> params = ExtrinsicParameters(T, position=1)
    >>> params.matrix.shape
    (4, 4)
    >>> params.position
    1
    >>> # From list without position
    >>> T_list = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    >>> params = ExtrinsicParameters(T_list)
    >>> params.position is None
    True
    >>> # Matrix only (position defaults to None)
    >>> params = ExtrinsicParameters(np.eye(4))

    Notes
    -----
    - The extrinsic matrix represents the transformation from world coordinates
      to camera coordinates.
    - If no position is provided, a warning is printed and position is set to None.
    - The matrix must be exactly 4×4 in shape.
    """
    @singledispatchmethod
    def __init__(self, data) -> None:
        raise TypeError(f"Unsupported init type: {type(data)!r}")
    
    @__init__.register
    def _(self, matrix: np.ndarray, position: int | None = None, to_fix: bool = True) -> None:
        self.to_fix = to_fix
        self.original_matrix = matrix
        self.matrix = matrix
        self.position = position

    @__init__.register
    def _(self, matrix: list, position: int | None = None, to_fix: bool = True) -> None:
        self.to_fix = to_fix
        self.original_matrix = np.array(matrix)
        self.matrix = np.array(matrix)
        self.position = position


    @property
    def to_fix(self) -> bool:
        return self._to_fix
    
    @to_fix.setter
    def to_fix(self, value: bool) -> None:
        if not isinstance(value, bool):
            raise TypeError("to_fix must be a boolean")
        self._to_fix = value


    @property
    def matrix(self) -> np.ndarray:
        return self._matrix
    
    @matrix.setter
    def matrix(self, value: np.ndarray) -> None:
        if value.shape != (4, 4):
            raise ValueError("Extrinsic matrix must be of shape (4, 4)")

        # Sadly BlenderNerF uses the inverse of the extrinsic matrix convention
        if(self.to_fix): value = np.linalg.inv(value)

        self._matrix = value

    
    @property
    def original_matrix(self) -> np.ndarray:
        return self._original_matrix
    
    @original_matrix.setter
    def original_matrix(self, value: np.ndarray) -> None:
        if value.shape != (4, 4):
            raise ValueError("Extrinsic matrix must be of shape (4, 4)")

        self._original_matrix = value


    @property
    def position(self) -> int:
        return self._position
    
    @position.setter
    def position(self, value: int | None) -> None:
        if value is None:
            print("⚠️ Warning: No camera position was defined, position is set to None")
            self._position = value
            return
        elif not isinstance(value, int):
            raise TypeError("Position must be an integer")
        self._position = value


    @property
    def R(self) -> np.ndarray:
        """Get the rotation component of the extrinsic matrix."""
        return self._matrix[:3, :3]
    
    @property
    def t(self) -> np.ndarray:
        """Get the translation component of the extrinsic matrix."""
        return self._matrix[:3, 3]



# ================================================================
# 2. Section: Extrinsic Group Class
# ================================================================
class ExtrinsicGroup:
    """
    A flexible container for extrinsic camera parameters.
    This class supports initialization from multiple data sources
    using `functools.singledispatchmethod`:
    - A NumPy array (`np.ndarray`) of ExtrinsicParameters objects
    - A list (`list`) of ExtrinsicParameters objects  
    - A JSON file path (`str`) containing frame transformation matrices

    Parameters
    ----------
    data : np.ndarray | list | str
        The source for the extrinsic parameters.
        - If `np.ndarray`: used directly as array of ExtrinsicParameters
        - If `list`: converted to `np.ndarray` of ExtrinsicParameters
        - If `str`: interpreted as a path to a JSON file with frame data

    Attributes
    ----------
    extrinsics : np.ndarray
        A 1D array of ExtrinsicParameters objects representing camera poses.

    Raises
    ------
    TypeError
        If the input type is not supported or if array elements are not ExtrinsicParameters.
    ValueError
        If the extrinsics array is not 1-dimensional.

    Examples
    --------
    >>> import numpy as np
    >>> # From array of ExtrinsicParameters
    >>> ext_params = [ExtrinsicParameters(transform_matrix, 0)]
    >>> group = ExtrinsicGroup(np.array(ext_params))
    >>> group.extrinsics.shape
    (1,)
    >>> # From list
    >>> group = ExtrinsicGroup(ext_params)
    >>> len(group.extrinsics)
    1
    >>> # From JSON file
    >>> group = ExtrinsicGroup("transforms.json")

    Notes
    -----
    - The JSON file must contain a key `"frames"` with frame objects that have
      `"transform_matrix"` keys containing 4×4 transformation matrices.
    - Each ExtrinsicParameters object is created with its corresponding frame index.
    - The setter enforces that all elements are ExtrinsicParameters instances.
    """
    @singledispatchmethod
    def __init__(self, data) -> None:
        raise TypeError(f"Unsupported init type: {type(data)!r}")
    
    @__init__.register
    def _(self, extrinsics: np.ndarray) -> None:
        self.extrinsics = extrinsics

    @__init__.register
    def _(self, extrinsics: list) -> None:
        self.extrinsics = np.array(extrinsics)

    @__init__.register
    def _(self, extrinsics_path: str, to_fix: bool = True) -> None:
        with open(extrinsics_path, 'r') as f:
            extrinsics_data = json.load(f)
        extrinsics = []
        for idx, ext in enumerate(extrinsics_data['frames']):
            transform_matrix = ext['transform_matrix']
            extrinsics.append(ExtrinsicParameters(transform_matrix, idx, to_fix=to_fix))
        self.extrinsics = np.array(extrinsics)

    @property
    def extrinsics(self) -> np.ndarray:
        return self._extrinsics
    
    @extrinsics.setter
    def extrinsics(self, value: np.ndarray) -> None:
        if value.ndim != 1:
            raise ValueError("Extrinsics must be a 1D array of ExtrinsicParameters")
        for ext in value:
            if not isinstance(ext, ExtrinsicParameters):
                raise TypeError("All elements must be of type ExtrinsicParameters")
        self._extrinsics = value

    @property
    def nr_cameras(self) -> int:
        """Number of cameras (i.e., number of extrinsic parameter sets)."""
        return len(self._extrinsics)