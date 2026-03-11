"""sdf3d.grid — Convenience wrapper for sampling a 3D SDF on a uniform grid.

Mirrors ``sdf2d.grid`` (if present) but for 3-D geometries.

The :class:`~sdf3d.geometry.SDF3D` base class already provides
``geom.to_numpy(bounds, resolution)`` — this module simply exports a
free-function form ``sample_levelset_3d(geom, bounds, resolution)`` that is
easier to use in pipelines and unit tests without holding a geometry object.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

_Array = npt.NDArray[np.floating]

_Bounds3D = tuple[
    tuple[float, float],
    tuple[float, float],
    tuple[float, float],
]
_Resolution3D = tuple[int, int, int]


def sample_levelset_3d(
    geom,
    bounds: _Bounds3D,
    resolution: _Resolution3D,
) -> _Array:
    """Sample *geom* on a uniform 3-D cell-centred grid.

    Parameters
    ----------
    geom:
        Any pySdf 3-D geometry — an ``SDF3D`` subclass or any callable
        ``(p: (...,3) ndarray) -> (...) ndarray``.
    bounds:
        ``((x0, x1), (y0, y1), (z0, z1))`` physical extents of the domain.
    resolution:
        ``(nx, ny, nz)`` number of grid cells along each axis.

    Returns
    -------
    numpy.ndarray
        Shape ``(nz, ny, nx)`` — signed distances at cell centres,
        z-first (array) indexing.

    Examples
    --------
    >>> from sdf3d import Sphere3D
    >>> from sdf3d.grid import sample_levelset_3d
    >>> phi = sample_levelset_3d(
    ...     Sphere3D(0.5),
    ...     bounds=((-1, 1), (-1, 1), (-1, 1)),
    ...     resolution=(32, 32, 32),
    ... )
    >>> phi.shape
    (32, 32, 32)
    """
    (x0, x1), (y0, y1), (z0, z1) = bounds
    nx, ny, nz = resolution

    xs = np.linspace(x0, x1, nx, endpoint=False) + (x1 - x0) / (2.0 * nx)
    ys = np.linspace(y0, y1, ny, endpoint=False) + (y1 - y0) / (2.0 * ny)
    zs = np.linspace(z0, z1, nz, endpoint=False) + (z1 - z0) / (2.0 * nz)

    Z, Y, X = np.meshgrid(zs, ys, xs, indexing="ij")
    p = np.stack([X, Y, Z], axis=-1)  # shape (nz, ny, nx, 3)

    # Support both SDF3D objects (have .sdf method) and bare callables
    if hasattr(geom, "sdf"):
        return geom.sdf(p)
    return geom(p)
