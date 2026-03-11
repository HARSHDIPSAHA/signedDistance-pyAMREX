"""3D center-point distributions for placing shapes in a bounding box.

Provides three placement strategies, all returning ``(N, 3)`` NumPy arrays
of center coordinates:

* :func:`generate_centers_random`   — rejection-sampling with min-separation
* :func:`generate_centers_inline`   — regular Cartesian grid
* :func:`generate_centers_staggered` — alternating-layer offset grid

A helper :func:`distribute_shape` places copies of any :class:`~sdf3d.SDF3D`
shape at the generated centers and returns their union as a new ``SDF3D``.

All functions are pure-NumPy (no CuPy / AMReX dependency required).
"""

from __future__ import annotations

import functools
import operator
from typing import Callable, Sequence

import numpy as np

# ---------------------------------------------------------------------------
# Type aliases (local — avoids a hard import of geometry at module level)
# ---------------------------------------------------------------------------
_Bounds3D = tuple[tuple[float, float], tuple[float, float], tuple[float, float]]


# ===========================================================================
# Random distribution
# ===========================================================================

def generate_centers_random(
    bounds: _Bounds3D,
    num_centers: int,
    min_separation: float,
    *,
    seed: int | None = None,
    max_attempts: int = 10_000,
) -> np.ndarray:
    """Generate random 3D center points within *bounds* with minimum separation.

    Uses rejection sampling (ported from PySCIMITAReX's
    ``Shapes.generateCentersRandom``, pure-NumPy version).

    Parameters
    ----------
    bounds:
        ``((x0, x1), (y0, y1), (z0, z1))`` physical extents.
    num_centers:
        Number of center points to place.
    min_separation:
        Minimum Euclidean distance between any two centers (threshold
        separation factor — TSF).
    seed:
        Optional random seed for reproducibility.
    max_attempts:
        Maximum number of candidate points to try per center before
        raising :exc:`RuntimeError`.

    Returns
    -------
    numpy.ndarray
        Shape ``(num_centers, 3)`` array of center coordinates.

    Raises
    ------
    RuntimeError
        If a center cannot be placed within *max_attempts* trials
        (domain may be too small for the requested separation).
    """
    (x0, x1), (y0, y1), (z0, z1) = bounds
    rng = np.random.default_rng(seed)
    centers: list[np.ndarray] = []

    for i in range(num_centers):
        for _ in range(max_attempts):
            candidate = rng.uniform(
                [x0, y0, z0],
                [x1, y1, z1],
            )
            if centers:
                existing = np.array(centers)
                dists = np.linalg.norm(existing - candidate, axis=1)
                if dists.min() < min_separation:
                    continue
            centers.append(candidate)
            break
        else:
            raise RuntimeError(
                f"Could not place center {i + 1}/{num_centers} after "
                f"{max_attempts} attempts.  Try reducing num_centers or "
                f"min_separation."
            )

    return np.array(centers)


# ===========================================================================
# Inline (Cartesian grid) distribution
# ===========================================================================

def generate_centers_inline(
    bounds: _Bounds3D,
    num_centers_per_axis: int | tuple[int, int, int],
) -> np.ndarray:
    """Generate evenly spaced 3D center points in a regular grid.

    Parameters
    ----------
    bounds:
        ``((x0, x1), (y0, y1), (z0, z1))`` physical extents.
    num_centers_per_axis:
        Either a single ``int`` (same count along every axis) or a
        ``(nx, ny, nz)`` tuple.

    Returns
    -------
    numpy.ndarray
        Shape ``(nx * ny * nz, 3)`` array of center coordinates.
    """
    (x0, x1), (y0, y1), (z0, z1) = bounds

    if isinstance(num_centers_per_axis, (int, np.integer)):
        nx = ny = nz = int(num_centers_per_axis)
    else:
        nx, ny, nz = (int(n) for n in num_centers_per_axis)

    dx = (x1 - x0) / nx
    dy = (y1 - y0) / ny
    dz = (z1 - z0) / nz

    xs = x0 + dx * (np.arange(nx) + 0.5)
    ys = y0 + dy * (np.arange(ny) + 0.5)
    zs = z0 + dz * (np.arange(nz) + 0.5)

    ZZ, YY, XX = np.meshgrid(zs, ys, xs, indexing="ij")
    centers = np.stack([XX.ravel(), YY.ravel(), ZZ.ravel()], axis=-1)
    return centers


# ===========================================================================
# Staggered distribution
# ===========================================================================

def generate_centers_staggered(
    bounds: _Bounds3D,
    num_centers_per_axis: int | tuple[int, int, int],
    *,
    offset_fraction: float = 0.5,
) -> np.ndarray:
    """Generate staggered 3D center points (alternating-layer offset grid).

    Odd-indexed layers (in Z) are shifted by *offset_fraction* of one cell
    width in X, and also in Y for a body-centred-like stagger.

    Parameters
    ----------
    bounds:
        ``((x0, x1), (y0, y1), (z0, z1))`` physical extents.
    num_centers_per_axis:
        Either a single ``int`` or ``(nx, ny, nz)`` tuple.
    offset_fraction:
        Fractional offset (0–1) applied to alternating layers.
        Default 0.5 gives half-cell offset.

    Returns
    -------
    numpy.ndarray
        Shape ``(nx * ny * nz, 3)`` array of center coordinates.
    """
    (x0, x1), (y0, y1), (z0, z1) = bounds

    if isinstance(num_centers_per_axis, (int, np.integer)):
        nx = ny = nz = int(num_centers_per_axis)
    else:
        nx, ny, nz = (int(n) for n in num_centers_per_axis)

    dx = (x1 - x0) / nx
    dy = (y1 - y0) / ny
    dz = (z1 - z0) / nz

    centers: list[np.ndarray] = []
    for k in range(nz):
        offset_x = (k % 2) * offset_fraction * dx
        offset_y = (k % 2) * offset_fraction * dy
        for j in range(ny):
            for i in range(nx):
                cx = x0 + dx * (i + 0.5) + offset_x
                cy = y0 + dy * (j + 0.5) + offset_y
                cz = z0 + dz * (k + 0.5)
                centers.append([cx, cy, cz])

    return np.array(centers)


# ===========================================================================
# distribute_shape
# ===========================================================================

def distribute_shape(
    shape_factory: Callable[..., object],
    centers: np.ndarray,
    **kwargs,
) -> object:
    """Place copies of a shape at each center and return their union.

    Parameters
    ----------
    shape_factory:
        A callable that returns an :class:`~sdf3d.geometry.SDF3D` instance,
        e.g. ``lambda: Sphere3D(0.1)``.  Keyword arguments in *kwargs* are
        forwarded to every call.
    centers:
        ``(N, 3)`` array of center coordinates produced by one of the
        ``generate_centers_*`` functions.
    **kwargs:
        Extra keyword arguments forwarded to *shape_factory*.

    Returns
    -------
    SDF3D
        Union of all placed shape instances.

    Raises
    ------
    ValueError
        If *centers* is empty.
    """
    centers = np.asarray(centers, dtype=float)
    if centers.ndim != 2 or centers.shape[1] != 3:
        raise ValueError(
            f"centers must be shape (N, 3), got {centers.shape}"
        )
    if len(centers) == 0:
        raise ValueError("centers array is empty")

    shapes = [
        shape_factory(**kwargs).translate(float(cx), float(cy), float(cz))
        for cx, cy, cz in centers
    ]
    return functools.reduce(operator.or_, shapes)
