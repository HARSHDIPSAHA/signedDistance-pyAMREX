"""2D center-point distributions for placing shapes in a bounding rectangle.

Provides three placement strategies, all returning ``(N, 2)`` NumPy arrays
of center coordinates:

* :func:`generate_centers_random`    — rejection-sampling with min-separation
* :func:`generate_centers_inline`    — regular Cartesian grid
* :func:`generate_centers_staggered` — alternating-row offset grid

A helper :func:`distribute_shape` places copies of any
:class:`~sdf2d.geometry.SDF2D` shape at the generated centers and returns
their union as a new ``SDF2D``.

All functions are pure-NumPy (no CuPy / AMReX dependency required).
"""

from __future__ import annotations

import functools
import operator
from typing import Callable, Sequence

import numpy as np

# ---------------------------------------------------------------------------
# Type alias
# ---------------------------------------------------------------------------
_Bounds2D = tuple[tuple[float, float], tuple[float, float]]


# ===========================================================================
# Random distribution
# ===========================================================================

def generate_centers_random(
    bounds: _Bounds2D,
    num_centers: int,
    min_separation: float,
    *,
    seed: int | None = None,
    max_attempts: int = 10_000,
) -> np.ndarray:
    """Generate random 2D center points within *bounds* with minimum separation.

    Uses rejection sampling (2D analogue of PySCIMITAReX's
    ``Shapes.generateCentersRandom``).

    Parameters
    ----------
    bounds:
        ``((x0, x1), (y0, y1))`` physical extents.
    num_centers:
        Number of center points to place.
    min_separation:
        Minimum Euclidean distance between any two centers.
    seed:
        Optional random seed for reproducibility.
    max_attempts:
        Maximum number of candidate points to try per center before
        raising :exc:`RuntimeError`.

    Returns
    -------
    numpy.ndarray
        Shape ``(num_centers, 2)`` array of center coordinates.

    Raises
    ------
    RuntimeError
        If a center cannot be placed within *max_attempts* trials.
    """
    (x0, x1), (y0, y1) = bounds
    rng = np.random.default_rng(seed)
    centers: list[np.ndarray] = []

    for i in range(num_centers):
        for _ in range(max_attempts):
            candidate = rng.uniform([x0, y0], [x1, y1])
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
    bounds: _Bounds2D,
    num_centers_per_axis: int | tuple[int, int],
) -> np.ndarray:
    """Generate evenly spaced 2D center points in a regular grid.

    Parameters
    ----------
    bounds:
        ``((x0, x1), (y0, y1))`` physical extents.
    num_centers_per_axis:
        Either a single ``int`` (same count along both axes) or a
        ``(nx, ny)`` tuple.

    Returns
    -------
    numpy.ndarray
        Shape ``(nx * ny, 2)`` array of center coordinates.
    """
    (x0, x1), (y0, y1) = bounds

    if isinstance(num_centers_per_axis, (int, np.integer)):
        nx = ny = int(num_centers_per_axis)
    else:
        nx, ny = (int(n) for n in num_centers_per_axis)

    dx = (x1 - x0) / nx
    dy = (y1 - y0) / ny

    xs = x0 + dx * (np.arange(nx) + 0.5)
    ys = y0 + dy * (np.arange(ny) + 0.5)

    YY, XX = np.meshgrid(ys, xs, indexing="ij")
    centers = np.stack([XX.ravel(), YY.ravel()], axis=-1)
    return centers


# ===========================================================================
# Staggered distribution
# ===========================================================================

def generate_centers_staggered(
    bounds: _Bounds2D,
    num_centers_per_axis: int | tuple[int, int],
    *,
    offset_fraction: float = 0.5,
) -> np.ndarray:
    """Generate staggered 2D center points (alternating-row offset grid).

    Odd rows (in Y) are shifted by *offset_fraction* of one cell width in X.

    Parameters
    ----------
    bounds:
        ``((x0, x1), (y0, y1))`` physical extents.
    num_centers_per_axis:
        Either a single ``int`` or ``(nx, ny)`` tuple.
    offset_fraction:
        Fractional offset (0–1) applied to alternating rows.

    Returns
    -------
    numpy.ndarray
        Shape ``(nx * ny, 2)`` array of center coordinates.
    """
    (x0, x1), (y0, y1) = bounds

    if isinstance(num_centers_per_axis, (int, np.integer)):
        nx = ny = int(num_centers_per_axis)
    else:
        nx, ny = (int(n) for n in num_centers_per_axis)

    dx = (x1 - x0) / nx
    dy = (y1 - y0) / ny

    centers: list[list[float]] = []
    for j in range(ny):
        offset_x = (j % 2) * offset_fraction * dx
        for i in range(nx):
            cx = x0 + dx * (i + 0.5) + offset_x
            cy = y0 + dy * (j + 0.5)
            centers.append([cx, cy])

    return np.array(centers)


# ===========================================================================
# distribute_shape
# ===========================================================================

def distribute_shape(
    shape_factory: Callable[..., object],
    centers: np.ndarray,
    **kwargs,
) -> object:
    """Place copies of a 2D shape at each center and return their union.

    Parameters
    ----------
    shape_factory:
        A callable that returns an :class:`~sdf2d.geometry.SDF2D` instance,
        e.g. ``lambda: Circle2D(0.08)``.  Keyword arguments in *kwargs* are
        forwarded to every call.
    centers:
        ``(N, 2)`` array of center coordinates produced by one of the
        ``generate_centers_*`` functions.
    **kwargs:
        Extra keyword arguments forwarded to *shape_factory*.

    Returns
    -------
    SDF2D
        Union of all placed shape instances.

    Raises
    ------
    ValueError
        If *centers* is empty or not shape ``(N, 2)``.
    """
    centers = np.asarray(centers, dtype=float)
    if centers.ndim != 2 or centers.shape[1] != 2:
        raise ValueError(
            f"centers must be shape (N, 2), got {centers.shape}"
        )
    if len(centers) == 0:
        raise ValueError("centers array is empty")

    shapes = [
        shape_factory(**kwargs).translate(cx, cy)
        for cx, cy in centers
    ]
    return functools.reduce(operator.or_, shapes)
