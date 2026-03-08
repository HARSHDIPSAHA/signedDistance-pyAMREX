"""Grid sampling utilities for 2D signed distance functions."""

from __future__ import annotations

import os

import numpy as np
import numpy.typing as npt

from .geometry import SDF2D

_Array = npt.NDArray[np.floating]
_Bounds2D = tuple[tuple[float, float], tuple[float, float]]
_Resolution2D = tuple[int, int]


def sample_levelset_2d(
    geom: SDF2D,
    bounds: _Bounds2D,
    resolution: _Resolution2D,
) -> _Array:
    """Sample *geom* on a uniform 2-D cell-centred grid.

    Parameters
    ----------
    geom:
        A :class:`SDF2D` geometry whose ``sdf()`` method accepts ``(..., 2)`` arrays.
    bounds:
        ``((x0, x1), (y0, y1))`` physical extents of the domain.
    resolution:
        ``(nx, ny)`` number of cells along each axis.

    Returns
    -------
    numpy.ndarray
        Shape ``(ny, nx)`` array of signed distances, row-major (y first).
    """
    (x0, x1), (y0, y1) = bounds
    nx, ny = resolution

    xs = np.linspace(x0, x1, nx, endpoint=False) + (x1 - x0) / (2.0 * nx)
    ys = np.linspace(y0, y1, ny, endpoint=False) + (y1 - y0) / (2.0 * ny)

    Y, X = np.meshgrid(ys, xs, indexing="ij")
    p = np.stack([X, Y], axis=-1)
    return geom.sdf(p)


def save_npy(path: str, phi: _Array) -> None:
    """Save *phi* array to *path* (creates parent directories if needed)."""
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    np.save(path, phi)
