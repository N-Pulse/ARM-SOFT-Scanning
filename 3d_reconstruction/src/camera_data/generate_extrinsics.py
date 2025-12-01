# Generates extrinsic matrices for all cameras and stores them in a json file
# ================================================================
# 0. Section: Imports
# ================================================================
import numpy as np
from .extrinsics import ExtrinsicParameters, ExtrinsicGroup



# ================================================================
# 1. Section: Geneate Extrinsics Functions
# ================================================================
def look_at(eye, target=np.array([0.0, 0.0, 0.0]), up=np.array([0.0, 0.0, 1.0])):
    eye = np.asarray(eye, dtype=float)
    target = np.asarray(target, dtype=float)
    up = np.asarray(up, dtype=float)

    # Forward vector (camera z-axis, pointing from camera to target)
    z_cam = target - eye
    z_cam /= np.linalg.norm(z_cam)

    # Right vector (camera x-axis)
    x_cam = np.cross(z_cam, up)
    x_cam /= np.linalg.norm(x_cam)

    # True up vector (camera y-axis)
    y_cam = np.cross(z_cam, x_cam)

    # Rotation matrix: columns are camera axes expressed in world coords
    R = np.stack([x_cam, y_cam, z_cam], axis=0)  # 3x3

    # Translation: t = -R * C, where C = eye
    t = -R @ eye

    # Assemble [R | t]
    extrinsic = np.hstack([R, t[:, None]])  # 3x4

    # Add the last row for homogeneous coordinates if needed
    extrinsic = np.vstack([extrinsic, np.array([0.0, 0.0, 0.0, 1.0])])  # 4x4
    return extrinsic


def circular_extrinsics(radius: float, num_positions: int, z: float = 0.0, center: np.ndarray = np.array([0.0, 0.0, 0.0])) -> ExtrinsicGroup:
    center = np.asarray(center, dtype=float)

    extrinsics = []
    for i in range(num_positions):
        angle = 2.0 * np.pi * i / num_positions  # evenly spaced
        # Camera position on circle (XY-plane around center)
        cam_x = center[0] + radius * np.cos(angle)
        cam_y = center[1] + radius * np.sin(angle)
        cam_z = center[2] + z  # usually z=0, same as center

        eye = np.array([cam_x, cam_y, cam_z], dtype=float)
        extrinsic = look_at(eye, target=center, up=np.array([0.0, 0.0, 1.0]))
        extrinsic = ExtrinsicParameters(extrinsic, int(i), to_fix=False)
        extrinsics.append(extrinsic)

    extrinsics = ExtrinsicGroup(extrinsics)

    return extrinsics