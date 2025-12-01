# ================================================================
# 0. Section: Imports
# ================================================================
import cv2
import os

import numpy as np

from dataclasses import dataclass
from functools import singledispatchmethod



# ================================================================
# 1. Section: Scan Container Classes
# ================================================================
@dataclass
class Scan:
    """
    A container for 2D scan data with associated position information.
    This class uses `functools.singledispatchmethod` to support flexible initialization
    from NumPy arrays with optional position parameters.

    Parameters
    ----------
    data : np.ndarray
        The scan data array. Must be 2-dimensional.
    position : int
        The position identifier associated with this scan.

    Attributes
    ----------
    scan : np.ndarray
        The 2D scan data array.
    position : int | None
        The position identifier. None if no position was specified.

    Raises
    ------
    TypeError
        If the input type is not supported or position is not an integer.
    ValueError
        If the scan array is not 2-dimensional.

    Examples
    --------
    >>> import numpy as np
    >>> # Create scan with 2D array and position
    >>> scan_data = np.random.rand(100, 200)
    >>> scan = Scan(scan_data, position=1)
    >>> scan.scan.shape
    (100, 200)
    >>> scan.position
    1

    Notes
    -----
    - The scan array must be exactly 2-dimensional (1 < ndim < 3).
    - If no position is provided, a warning is printed and position is set to None.
    - Position must be an integer value when specified.
    """
    @singledispatchmethod
    def __init__(self, data) -> None:
        raise TypeError(f"Unsupported init type: {type(data)!r}")
    
    @__init__.register
    def _(self, array: np.ndarray, position: int) -> None:
        self.scan = array
        self.position = position


    @property
    def scan(self) -> np.ndarray:
        return self._scan
    
    @scan.setter
    def scan(self, value: np.ndarray) -> None:
        if not(1 < len(value.shape) < 3):
            raise ValueError("Scans must be two dimensional")
        self._scan = value


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
    def shape(self) -> tuple[int, int]:
        return self.scan.shape



# ================================================================
# 2. Section: Scans Container Class
# ================================================================
@dataclass
class Scans:
    """
    A container class for managing multiple ultrasound scan images and their positions.
    This class supports initialization from multiple data sources using 
    `functools.singledispatchmethod`:
    - A NumPy array of Scan objects
    - A list of Scan objects  
    - Arrays/lists of images with corresponding position arrays
    - A directory path containing scan image files

    Parameters
    ----------
    data : np.ndarray | list | str
        The source for the scan data.
        - If `np.ndarray` of Scan objects: used directly
        - If `list` of Scan objects: converted to `np.ndarray`
        - If `str`: interpreted as directory path containing image files
    images : np.ndarray | list, optional
        Array or list of scan images when providing separate images and positions.
    positions : np.ndarray | list, optional
        Array or list of position values corresponding to each scan image.

    Attributes
    ----------
    scans : np.ndarray
        A 1D array of Scan objects containing the ultrasound scan data.
    positions : np.ndarray
        A 1D array of position values corresponding to each scan.

    Properties
    ----------
    nr_positions : int
        The number of scan positions in the collection.
    scan_shape : tuple[int, int]
        The shape (height, width) of the scan images.

    Methods
    -------
    scan(index: int) -> np.ndarray
        Returns the scan image at the specified index.
    position(index: int) -> int
        Returns the position value at the specified index.

    Raises
    ------
    TypeError
        If the input type is not supported or if array elements are not Scan objects.
    ValueError
        If scans have different shapes or if arrays are inconsistent.

    Examples
    --------
    >>> import numpy as np
    >>> # From directory of image files
    >>> scans = Scans("/path/to/scan/images/")
    >>> scans.nr_positions
    50
    >>> scans.scan_shape
    (512, 512)
    >>> # From list of Scan objects
    >>> scan_list = [Scan(image1, 0), Scan(image2, 1)]
    >>> scans = Scans(scan_list)
    >>> scans.scan(0).shape
    (512, 512)
    >>> # From separate images and positions
    >>> images = np.array([img1, img2, img3])
    >>> positions = np.array([0, 1, 2])
    >>> scans = Scans(images, positions)

    Notes
    -----
    - When loading from a directory, supported image formats include: 
      .png, .jpg, .jpeg, .tiff, .bmp, .gif
    - Images are automatically converted to binary (0 or 1) grayscale format
    - All scans must have the same dimensions
    - Files are sorted alphabetically when loading from directory
    """
    @singledispatchmethod
    def __init__(self, data) -> None:
        raise TypeError(f"Unsupported init type: {type(data)!r}")
    
    @__init__.register
    def _(self, images: np.ndarray) -> None:
        self.scans = images
        self.positions = np.array([scan.position for scan in images])

    @__init__.register
    def _(self, images: list) -> None:
        self.scans = np.array(images)
        self.positions = np.array([scan.position for scan in images])

    @__init__.register
    def _(self, images: np.ndarray | list, positions: np.ndarray | list) -> None:
        self.scans = np.array(images)
        self.positions = np.array(positions)

    @__init__.register
    def _(self, scans_path: str) -> None:
        files = [f for f in os.listdir(scans_path) if f.endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif'))]
        files.sort()

        scans = []
        for idx, file in enumerate(files):
            img = cv2.imread(os.path.join(scans_path, file), cv2.IMREAD_GRAYSCALE)
            img = np.where(img > 0, 1, 0).astype(np.uint8)
            scan = Scan(img, idx)
            scans.append(scan)
        
        self.scans = np.array(scans)
        self.positions = np.array([scan.position for scan in scans])


    @property
    def scans(self) -> np.ndarray:
        return self._scans
    
    @scans.setter
    def scans(self, value: np.ndarray) -> None:
        test_shape = None
        for ext in value:
            if test_shape is None: test_shape = ext.scan.shape
            if not isinstance(ext, Scan):
                raise TypeError("All elements must be of type Scan")
            if ext.scan.shape != test_shape:
                raise ValueError("All scans must have the same shape")
        self._scans = value


    @property
    def nr_positions(self) -> int:
        return len(self.positions)
    
    @property
    def scan_shape(self) -> tuple[int, int]:
        return self.scans[0].scan.shape
    

    def scan(self, index: int) -> np.ndarray:
        return self.scans[index].scan
    
    def full_scan(self, index: int) -> Scan:
        return self.scans[index]
    
    def position(self, index: int) -> int:
        return self.scans[index].position