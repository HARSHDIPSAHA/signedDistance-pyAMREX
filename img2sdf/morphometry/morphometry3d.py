"""3D morphometric analysis from level-set fields.

Computes volumetric shape descriptors for 3D segmented objects:
- Volume
- Surface area (via marching cubes)
- Sphericity

Sign convention: pySdf uses φ < 0 inside, φ > 0 outside.
"""
from __future__ import annotations
import numpy as np


def compute_morphometry_3d(phi: np.ndarray, voxel_size: float = 1.0) -> dict:
    """Compute 3D morphometric features from a level-set field.

    Parameters
    ----------
    phi:
        Level-set field of shape (D, H, W).  pySdf convention: φ < 0 inside
        the object, φ > 0 outside.
    voxel_size:
        Physical size of each voxel edge (isotropic).

    Returns
    -------
    dict
        Keys: ``'volume'``, ``'surface_area'``, ``'sphericity'``.

    Raises
    ------
    ImportError
        If ``scikit-image`` is not installed (required for marching cubes).
    """
    try:
        from skimage.measure import marching_cubes
    except ImportError as exc:
        raise ImportError(
            "scikit-image is required for compute_morphometry_3d. "
            "Install it with: pip install scikit-image"
        ) from exc

    phi = np.asarray(phi, dtype=np.float64)

    # --- Volume ---
    binary_mask = phi < 0.0
    volume = float(np.count_nonzero(binary_mask)) * voxel_size ** 3

    # --- Surface area via marching cubes ---
    spacing = (voxel_size, voxel_size, voxel_size)
    if phi.min() >= 0.0 or phi.max() <= 0.0:
        # No zero crossing → degenerate geometry
        surface_area = 0.0
    else:
        verts, faces, _, _ = marching_cubes(phi, level=0.0, spacing=spacing)
        # Sum triangle areas: A = 0.5 * |cross(v1-v0, v2-v0)|
        v0 = verts[faces[:, 0]]
        v1 = verts[faces[:, 1]]
        v2 = verts[faces[:, 2]]
        cross = np.cross(v1 - v0, v2 - v0)
        surface_area = float(0.5 * np.sum(np.linalg.norm(cross, axis=1)))

    # --- Sphericity: ψ = π^(1/3) * (6V)^(2/3) / A ---
    if surface_area > 0.0 and volume > 0.0:
        sphericity = float(
            (np.pi ** (1.0 / 3.0)) * (6.0 * volume) ** (2.0 / 3.0) / surface_area
        )
    else:
        sphericity = 0.0

    return {
        "volume": volume,
        "surface_area": surface_area,
        "sphericity": sphericity,
    }
