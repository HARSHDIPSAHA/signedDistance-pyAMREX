"""Internal SDF math for triangulated meshes.

All symbols here are private (underscore-prefixed).  Users should import
only from :mod:`stl2sdf.geometry`.

Unsigned distance uses Closest point on triangle using Voronoi region tests (Ericson’s algorithm).
Sign uses Möller–Trumbore ray casting.

How ``_triangles_to_sdf`` works: one pass over all F triangles accumulates
sq_min[i] (minimum squared distance, sqrt deferred to end) and hits[i] (forward
ray intersection parity count).  phi[i] = sign(hits[i]) * sqrt(sq_min[i]).
Complexity: O(F × N) where F = triangles, N = query points.
"""

from __future__ import annotations

import struct
from math import sqrt
from pathlib import Path
from typing import Optional, Union

import numpy as np

_DENOM_EPS:    float = 1e-30  # guards np.select branches that are always evaluated
_PARALLEL_EPS: float = 1e-10  # |det| below this → ray parallel to triangle plane
_ORIGIN_EPS:   float = 1e-10  # t ≤ this → intersection at or behind the ray origin

# Irrational components avoid axis-aligned degeneracies
_RAY_DIR: np.ndarray = np.array(
    [sqrt(2) - 1.0, sqrt(3) - 1.0, 1.0 / sqrt(3)], dtype=np.float64
)
_RAY_DIR = _RAY_DIR / np.linalg.norm(_RAY_DIR)


def _stl_to_triangles(path: Union[str, Path]) -> np.ndarray:
    """Read an STL file and return its triangles as a (F, 3, 3) float64 array.

    Supports binary and ASCII STL.  Normals are discarded.
    Detection uses the binary-size invariant (len == 84 + 50*F) rather than
    the "solid" keyword, which some CAD tools (e.g. SolidWorks) also write
    at the start of binary files.
    """
    path = Path(path)
    raw  = path.read_bytes()
    if len(raw) >= 84:
        count = struct.unpack_from("<I", raw, 80)[0]
        if len(raw) == 84 + 50 * count:
            return _binary_stl_to_triangles(raw)
    return _ascii_stl_to_triangles(raw.decode("ascii", errors="replace"))


def _binary_stl_to_triangles(raw: bytes) -> np.ndarray:
    count   = struct.unpack_from("<I", raw, 80)[0]
    dtype   = np.dtype([("normal", "<f4", (3,)), ("vertices", "<f4", (3, 3)), ("attr", "<u2")])
    records = np.frombuffer(raw, dtype=dtype, count=count, offset=84)
    return records["vertices"].astype(np.float64)  # (F, 3, 3)


def _ascii_stl_to_triangles(text: str) -> np.ndarray:
    verts: list[list[float]] = []
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("vertex"):
            parts = line.split()
            verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return np.array(verts, dtype=np.float64).reshape(-1, 3, 3)


# Closest point on triangle using Voronoi region tests (Ericson’s algorithm).
# taken from https://alexanderfabisch.github.io/distance3d/_modules/distance3d/distance/_triangle.html#point_to_triangle
# but vectorized for N query points on a single triangle
def _triangle_sq_dist(P: np.ndarray, tri: np.ndarray) -> np.ndarray:
    """Squared distance from each point in P (N, 3) to triangle tri (3, 3)."""
    A, B, C = tri[0], tri[1], tri[2]

    AB = B - A          # (3,)
    AC = C - A          # (3,)
    AP = P - A          # (N, 3)
    BP = P - B          # (N, 3)
    CP = P - C          # (N, 3)

    # d1–d6: projections of the vertex-to-point vectors onto AB and AC.
    # d1, d2 from A's perspective; d3, d4 from B's; d5, d6 from C's.
    d1 = AP @ AB    # (N,)
    d2 = AP @ AC    # (N,)
    d3 = BP @ AB    # (N,)
    d4 = BP @ AC    # (N,)
    d5 = CP @ AB    # (N,)
    d6 = CP @ AC    # (N,)

    # Cross-term determinants: proportional to the barycentric weight of the
    # opposite vertex. Negative vc → P outside edge AB; negative vb → outside
    # AC; negative va → outside BC.
    vc = d1 * d4 - d3 * d2
    vb = d5 * d2 - d1 * d6
    va = d3 * d6 - d5 * d4

    def _sq(cp):
        diff = P - cp
        return (diff * diff).sum(axis=-1)

    # Check if point in vertex region outside A
    cond_A = (d1 <= 0.0) & (d2 <= 0.0)

    # Check if point in vertex region outside B
    cond_B = (d3 >= 0.0) & (d4 <= d3)

    # Check if point in vertex region outside C
    cond_C = (d6 >= 0.0) & (d5 <= d6)

    # Check if point in edge region of AB
    # vc ≤ 0 (outside AB), d1 ≥ 0 (past A), d3 ≤ 0 (not yet past B).
    # t = d1 / (d1 - d3), clamped to [0, 1].
    cond_AB = (vc <= 0.0) & (d1 >= 0.0) & (d3 <= 0.0)
    t_AB    = d1 / np.maximum(d1 - d3, _DENOM_EPS)
    cp_AB   = A + np.clip(t_AB, 0.0, 1.0)[:, None] * AB

    # Check if point in edge region of AC
    # Symmetric to AB. t = d2 / (d2 - d6), clamped to [0, 1].
    cond_AC = (vb <= 0.0) & (d2 >= 0.0) & (d6 <= 0.0)
    t_AC    = d2 / np.maximum(d2 - d6, _DENOM_EPS)
    cp_AC   = A + np.clip(t_AC, 0.0, 1.0)[:, None] * AC

    # Check if point in edge region of BC
    # va ≤ 0, (d4-d3) ≥ 0 (past B), (d5-d6) ≥ 0 (not yet past C).
    # t = (d4-d3) / ((d4-d3) + (d5-d6)), clamped to [0, 1].
    cond_BC = (va <= 0.0) & ((d4 - d3) >= 0.0) & ((d5 - d6) >= 0.0)
    t_BC    = (d4 - d3) / np.maximum((d4 - d3) + (d5 - d6), _DENOM_EPS)
    cp_BC   = B + np.clip(t_BC, 0.0, 1.0)[:, None] * (C - B)

    # Point inside face region
    # Barycentric weights: u=va/denom (A), v=vb/denom (B), w=vc/denom (C).
    # Closest point = A + (vb/denom)*AB + (vc/denom)*AC.
    denom   = np.maximum(va + vb + vc, _DENOM_EPS)
    cp_face = A + (vb / denom)[:, None] * AB + (vc / denom)[:, None] * AC

    return np.select(
        [cond_A,  cond_B,  cond_C,  cond_AB,    cond_AC,    cond_BC],
        [_sq(A),  _sq(B),  _sq(C),  _sq(cp_AB), _sq(cp_AC), _sq(cp_BC)],
        default=_sq(cp_face),
    )

# Moller-Trumbore algorithm.
# taken from https://gist.github.com/V0XNIHILI/87c986441d8debc9cd0e9396580e85f4
# but vectorized for multiple rays on a single triangle
def _ray_triangle_hits(P: np.ndarray, ray_dir: np.ndarray, tri: np.ndarray) -> np.ndarray:
    """
    P: (N, 3) N ray origin points
    ray_dir: (3,) one common ray direction
    tri: (3, 3) triangle vertices

    Output: (N,) int32 array: 1 if the ray from that point hits the triangle, 0 otherwise.
    """

    # Triangle vertices and edges
    v0, v1, v2 = tri[0], tri[1], tri[2]
    e1 = v1 - v0
    e2 = v2 - v0

    # Determinant test: if near zero, ray is parallel to triangle plane
    h   = np.cross(ray_dir, e2)
    det = float(e1 @ h)
    if abs(det) < _PARALLEL_EPS:
        return np.zeros(len(P), dtype=np.int32)

    # The ray intersects the triangle
    # if some point along the ray
    # = some point inside the triangle

    # Point on triangle = v₀ + ue₁ + ve₂
    # ⇒ (u,v) = barycentric coordinates of the hit point
    # t = signed distance along the ray direction

    inv_det = 1.0 / det
    s = P - v0                  # (N, 3)
    u = inv_det * (s @ h)       # (N,)
    q = np.cross(s, e1)         # (N, 3)
    v = inv_det * (q @ ray_dir) # (N,)
    t = inv_det * (q @ e2)      # (N,)

    hit = (u >= 0.0) & (v >= 0.0) & ((u + v) <= 1.0) & (t > _ORIGIN_EPS)
    return hit.astype(np.int32)


def _triangles_to_sdf(
    points: np.ndarray,
    triangles: np.ndarray,
    *,
    ray_dir: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Signed distances from (N, 3) points to a triangle mesh (F, 3, 3).

    Returns (N,) phi: negative inside, positive outside.
    Requires a watertight mesh; O(F × N).
    """
    P    = np.asarray(points,    dtype=np.float64)
    tris = np.asarray(triangles, dtype=np.float64)
    if ray_dir is None:
        ray_dir = _RAY_DIR
    else:
        ray_dir = np.asarray(ray_dir, dtype=np.float64)
        ray_dir = ray_dir / np.linalg.norm(ray_dir)

    sq_min = np.full(len(P), np.inf)
    hits   = np.zeros(len(P), dtype=np.int32)
    for tri in tris:
        sq_min = np.minimum(sq_min, _triangle_sq_dist(P, tri))
        hits  += _ray_triangle_hits(P, ray_dir, tri)

    return np.where(hits % 2 == 1, -1.0, 1.0) * np.sqrt(np.maximum(sq_min, 0.0))
