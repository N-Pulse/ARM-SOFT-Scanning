# ================================================================
# 0. Section: Imports
# ================================================================
import numpy as np

from functools import singledispatchmethod

from .intrinsic import IntrinsicParameters
from .extrinsics import ExtrinsicGroup



# ================================================================
# 1. Section: Camera Data Class
# ================================================================
class CamerasData:
    """
    A container for camera calibration data including intrinsic and extrinsic parameters.
    This class supports initialization from multiple data sources
    using `functools.singledispatchmethod`:
    - Direct objects (`IntrinsicParameters` and `ExtrinsicGroup`)
    - File paths (`str`) to intrinsic and extrinsic parameter files

    Parameters
    ----------
    data : IntrinsicParameters, ExtrinsicGroup, or str
        The source for the camera data.
        - If `IntrinsicParameters` and `ExtrinsicGroup`: used directly
        - If two `str` arguments: interpreted as paths to intrinsic and extrinsic files

    Attributes
    ----------
    intrinisc : IntrinsicParameters
        The camera intrinsic calibration parameters.
    extrinsics : ExtrinsicGroup
        The camera extrinsic calibration parameters for multiple views.

    Raises
    ------
    TypeError
        If the input type combination is not supported.

    Examples
    --------
    >>> # From objects
    >>> intrinsic = IntrinsicParameters(K_matrix)
    >>> extrinsic = ExtrinsicGroup(poses)
    >>> cameras = CamerasData(intrinsic, extrinsic)
    >>> # From file paths
    >>> cameras = CamerasData("intrinsics.json", "extrinsics.json")
    >>> cameras.intrinisc.matrix.shape
    (3, 3)

    Notes
    -----
    - When initialized from file paths, the corresponding parameter objects
      are automatically created from the provided file paths.
    - The class ensures both intrinsic and extrinsic parameters are always
      available as attributes.
    """
    @singledispatchmethod
    def __init__(self, data) -> None:
        raise TypeError(f"Unsupported init type: {type(data)!r}")
    
    @__init__.register
    def _(self, intrinsic: IntrinsicParameters, extrinsics: ExtrinsicGroup) -> None:
        self.intrinsic = intrinsic
        self.extrinsics = extrinsics

    @__init__.register
    def _(self, intrinsic_path: str, extrinsics_path: str, to_fix: bool = True) -> None:
        self.intrinsic = IntrinsicParameters(intrinsic_path)
        self.extrinsics = ExtrinsicGroup(extrinsics_path, to_fix=to_fix)

    @property
    def K(self) -> IntrinsicParameters:
        return self.intrinsic.matrix
    
    @property
    def nr_cameras(self) -> int:
        return self.extrinsics.nr_cameras
    
    @property
    def extrinsics_array(self) -> ExtrinsicGroup:
        """Get the extrinsic parameters as a NumPy array of transformation matrices of shape (N, 4, 4)."""
        extrinsics_array = []
        for idx in range(self.nr_cameras):
            extrinsics_array.append(self.extrinsics.extrinsics[idx].matrix)
        return np.array(extrinsics_array)
    
    def P(self, camera_index: int) -> np.ndarray:
        P = self.K @ self.get_camera_extrinsics(camera_index)[:3, :]
        return P

    def get_camera_extrinsics(self, camera_index: int) -> np.ndarray:
        """Get the extrinsic parameters for a specific camera index."""
        if camera_index < 0 or camera_index >= self.nr_cameras:
            raise IndexError("Camera index out of range.")
        return self.extrinsics.extrinsics[camera_index].matrix