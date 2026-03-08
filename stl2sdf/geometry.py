"""Public API for stl2sdf: a single function, stl_to_geometry."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional, Union

import numpy as np

from sdf3d.geometry import SDF3D as _SDF3D
from ._math import _stl_to_triangles


def stl_to_geometry(
    path: Union[str, Path],
    *,
    ray_dir: Optional[np.ndarray] = None,
) -> _SDF3D:
    """Load an STL file and return a :class:`sdf3d.geometry.SDF3D`.

    The returned :class:`sdf3d.geometry.SDF3D` object has the same interface
    as analytic primitives (``Sphere3D``, ``Box3D``, etc.) and can be combined
    with them using ``.union()``, ``.subtract()``, ``.translate()``, etc.

    Sign convention: phi < 0 inside, phi = 0 on surface, phi > 0 outside.
    Requires a **watertight** mesh.  SDF evaluation is parallelised across
    CPU cores by `pysdf <https://github.com/sxyu/sdf>`_.

    Parameters
    ----------
    path:
        Path to the ``.stl`` file (binary or ASCII).
    ray_dir:
        Deprecated and ignored.  ``pysdf`` handles sign determination
        internally; pass ``None`` (the default).

    Examples
    --------
    >>> from stl2sdf import stl_to_geometry
    >>> from sdf3d import Sphere3D
    >>> from sdf3d.grid import sample_levelset_3d
    >>>
    >>> wheel = stl_to_geometry("mars_wheel.stl")
    >>> combined = wheel.union(Sphere3D(0.3).translate(0.5, 0, 0))
    >>> phi = sample_levelset_3d(combined, bounds=((-1,1),(-1,1),(-1,1)), resolution=(32,32,32))
    """
    from pysdf import SDF  # lazy import: clear ImportError if pysdf is missing

    if ray_dir is not None:
        warnings.warn(
            "ray_dir is ignored; pysdf handles sign determination internally.",
            UserWarning,
            stacklevel=2,
        )

    triangles = _stl_to_triangles(path)  # (F, 3, 3) float64
    # Deduplicate vertices so pysdf receives a proper indexed mesh.
    # np.unique on float32 STL coordinates is exact (no arithmetic, just sorting).
    verts_raw = triangles.reshape(-1, 3).astype(np.float32)
    verts, inverse = np.unique(verts_raw, axis=0, return_inverse=True)
    faces = inverse.reshape(-1, 3).astype(np.uint32)
    verts = np.ascontiguousarray(verts)
    faces = np.ascontiguousarray(faces)

    _sdf_obj = SDF(verts, faces)

    def _sdf(p: np.ndarray) -> np.ndarray:
        shape = p.shape[:-1]
        pts = np.ascontiguousarray(p.reshape(-1, 3), dtype=np.float32)
        # pysdf: positive inside, (N,1) float32 — negate and cast to match our convention
        return -_sdf_obj(pts).reshape(shape).astype(np.float64)

    return _SDF3D(_sdf)


def mesh_bounds(path, pad_frac: float = 0.10) -> tuple:
    """Return ``((x0,x1),(y0,y1),(z0,z1))`` bounding box of an STL mesh.

    Parameters
    ----------
    path:
        Path to the ``.stl`` file (``str`` or :class:`pathlib.Path`).
    pad_frac:
        Fractional padding added on each side; 0.10 → 10 % of each span.

    Returns
    -------
    tuple
        ``((x0,x1),(y0,y1),(z0,z1))`` as a 3-tuple of 2-tuples.
    """
    triangles = _stl_to_triangles(path)
    verts = triangles.reshape(-1, 3)
    lo, hi = verts.min(axis=0), verts.max(axis=0)
    pad = pad_frac * (hi - lo)
    return tuple(zip((lo - pad).tolist(), (hi + pad).tolist()))
