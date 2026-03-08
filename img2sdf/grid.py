"""NumPy path: image file → level-set ndarray or ImageGeometry2D.
"""
from __future__ import annotations
import numpy as np
from .geometry import ImageGeometry2D
from ._pipeline import run_pipeline


def image_to_levelset_2d(
    image_path: str,
    params: dict,
    *,
    levelset_index: int = 0,
) -> np.ndarray:
    """Run the full uSCMAN pipeline on *image_path* and return the Φ ndarray.

    Parameters
    ----------
    image_path:
        Path to the input image.  Supported formats: PNG, JPG, BMP,
        TIFF (any depth), PGM, and HDF5 (``.h5`` with a ``phi`` dataset).
    params:
        Parameter dictionary in the same schema as uSCMAN's JSON config.
        Required top-level keys: ``"Image Properties"``,
        ``"Preprocessing Properties"``, ``"Segmentation"``.
    levelset_index:
        For multiphase segmentation, which Φ to return (0 or 1).
    """
    result = run_pipeline(image_path, params)
    phi_key = f"Phi{levelset_index + 1}"
    phi_flat = result["Segmentation"][phi_key]
    ny = result["Segmentation"]["J"]
    nx = result["Segmentation"]["I"]
    return -phi_flat.reshape(ny, nx) 


def image_to_geometry_2d(
    image_path: str,
    params: dict,
    bounds: tuple[tuple[float, float], tuple[float, float]] | None = None,
    *,
    levelset_index: int = 0,
) -> ImageGeometry2D:
    """Run the uSCMAN pipeline and return an ``ImageGeometry2D`` CSG leaf.

    Parameters
    ----------
    image_path, params, levelset_index:
        Same as :func:`image_to_levelset_2d`.
    bounds:
        Physical domain ``((x0, x1), (y0, y1))``.  When *None*, the image
        is mapped to pixel coordinates ``((0, nx), (0, ny))``.

    Returns
    -------
    ImageGeometry2D
        Ready to compose with other sdf2d shapes via ``.union()``,
        ``.intersect()``, ``.subtract()``, etc.
    """
    phi = image_to_levelset_2d(image_path, params, levelset_index=levelset_index)
    ny, nx = phi.shape
    if bounds is None:
        bounds = ((0.0, float(nx)), (0.0, float(ny)))
    return ImageGeometry2D(phi, bounds, image_path=image_path)


def save_npy(phi: np.ndarray, path: str) -> None:
    """Save a Φ ndarray to a ``.npy`` file (mirrors sdf2d/grid.py)."""
    np.save(path, phi)