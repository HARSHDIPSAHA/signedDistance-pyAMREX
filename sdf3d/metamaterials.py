"""Metamaterial / TPMS geometry classes for sdf3d.

Implements the most common Triply Periodic Minimal Surfaces (TPMS) and
beam-based lattices as :class:`~sdf3d.geometry.SDF3D` subclasses.

Available classes
-----------------
* :class:`Gyroid3D`      — sin(x)cos(y) + sin(y)cos(z) + sin(z)cos(x)
* :class:`SchwarzP3D`    — cos(x) + cos(y) + cos(z)          (Primitive)
* :class:`SchwarzD3D`    — Diamond surface
* :class:`Neovius3D`     — 3*(cos x+cos y+cos z) + 4*cos x*cos y*cos z
* :class:`BCCLattice3D`  — Body-centered-cubic beam lattice
* :class:`FCCLattice3D`  — Face-centered-cubic beam lattice

Each class accepts `cell_size` (default 1.0) and `thickness` (the half-width
of the solid shell around the zero level-set, default 0.1).  Optional `repeat`
controls how many unit cells are tiled in each direction.

Sign convention: ``phi < 0`` inside the solid, ``phi > 0`` outside (void).

All classes inherit from :class:`~sdf3d.geometry.SDF3D` and therefore
support all existing methods (translate, rotate, union/subtract/intersect,
to_numpy, save_png, save_plotly_html, …).
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .geometry import SDF3D
from . import primitives as _sdf


# ---------------------------------------------------------------------------
# Shared helper: TPMS shell SDF from a level-set function
# ---------------------------------------------------------------------------

def _tpms_sdf(p: np.ndarray, fn, cell_size: float, thickness: float) -> np.ndarray:
    """Evaluate a TPMS field and return a shell SDF.

    Maps the raw TPMS field value ``F(p)`` to a signed distance field by
    treating ``|F(p)| - thickness`` as a 1-D SDF (the shell around the
    iso-surface at ``F = 0``).  The result is further normalised by
    ``cell_size`` so that the SDF scale matches physical units.

    Parameters
    ----------
    p:
        ``(..., 3)`` point array in physical coordinates.
    fn:
        Callable ``fn(u, v, w) -> ndarray`` where ``u, v, w`` are the
        angular coordinates ``2π * xyz / cell_size``.
    cell_size:
        Physical size of one unit cell.
    thickness:
        Half-width of the solid shell (in the TPMS field's intrinsic
        scale, which is normalised to ≈ 1 unit range by the trig functions).
    """
    scale = 2.0 * np.pi / cell_size
    u = p[..., 0] * scale
    v = p[..., 1] * scale
    w = p[..., 2] * scale
    F = fn(u, v, w)
    # Shell: phi < 0 where |F| < thickness
    return np.abs(F) - thickness


# ===========================================================================
# TPMS classes
# ===========================================================================

class Gyroid3D(SDF3D):
    """Gyroid TPMS surface.

    The Gyroid implicit: ``sin(x)cos(y) + sin(y)cos(z) + sin(z)cos(x) = 0``

    Parameters
    ----------
    cell_size:
        Physical period length of one unit cell.
    thickness:
        Half-thickness of the solid shell around the iso-surface.
    repeat:
        Optional ``(rx, ry, rz)`` — tile the field over this many cells in
        each direction by scaling the input domain.  ``None`` means infinite
        repetition (the TPMS already repeats by definition).
    """

    def __init__(
        self,
        cell_size: float = 1.0,
        thickness: float = 0.2,
        repeat: Sequence[int] | None = None,
    ) -> None:
        self._cell_size = float(cell_size)
        self._thickness = float(thickness)
        self._repeat = repeat

        def _fn(u, v, w):
            return (
                np.sin(u) * np.cos(v)
                + np.sin(v) * np.cos(w)
                + np.sin(w) * np.cos(u)
            )

        def func(p: np.ndarray) -> np.ndarray:
            if self._repeat is not None:
                rx, ry, rz = self._repeat
                domain = np.array([rx * self._cell_size,
                                   ry * self._cell_size,
                                   rz * self._cell_size])
                p = np.mod(p, domain)
            return _tpms_sdf(p, _fn, self._cell_size, self._thickness)

        super().__init__(func)


class SchwarzP3D(SDF3D):
    """Schwarz Primitive (P) TPMS surface.

    Implicit: ``cos(x) + cos(y) + cos(z) = 0``

    Parameters
    ----------
    cell_size:
        Physical period length of one unit cell.
    thickness:
        Half-thickness of the solid shell around the iso-surface.
    repeat:
        Optional ``(rx, ry, rz)`` tiling counts.
    """

    def __init__(
        self,
        cell_size: float = 1.0,
        thickness: float = 0.2,
        repeat: Sequence[int] | None = None,
    ) -> None:
        self._cell_size = float(cell_size)
        self._thickness = float(thickness)
        self._repeat = repeat

        def _fn(u, v, w):
            return np.cos(u) + np.cos(v) + np.cos(w)

        def func(p: np.ndarray) -> np.ndarray:
            if self._repeat is not None:
                rx, ry, rz = self._repeat
                domain = np.array([rx * self._cell_size,
                                   ry * self._cell_size,
                                   rz * self._cell_size])
                p = np.mod(p, domain)
            return _tpms_sdf(p, _fn, self._cell_size, self._thickness)

        super().__init__(func)


class SchwarzD3D(SDF3D):
    """Schwarz Diamond (D) TPMS surface.

    Implicit:
    ``sin(x)sin(y)sin(z) + sin(x)cos(y)cos(z)
      + cos(x)sin(y)cos(z) + cos(x)cos(y)sin(z) = 0``

    Parameters
    ----------
    cell_size:
        Physical period length of one unit cell.
    thickness:
        Half-thickness of the solid shell around the iso-surface.
    repeat:
        Optional ``(rx, ry, rz)`` tiling counts.
    """

    def __init__(
        self,
        cell_size: float = 1.0,
        thickness: float = 0.2,
        repeat: Sequence[int] | None = None,
    ) -> None:
        self._cell_size = float(cell_size)
        self._thickness = float(thickness)
        self._repeat = repeat

        def _fn(u, v, w):
            return (
                np.sin(u) * np.sin(v) * np.sin(w)
                + np.sin(u) * np.cos(v) * np.cos(w)
                + np.cos(u) * np.sin(v) * np.cos(w)
                + np.cos(u) * np.cos(v) * np.sin(w)
            )

        def func(p: np.ndarray) -> np.ndarray:
            if self._repeat is not None:
                rx, ry, rz = self._repeat
                domain = np.array([rx * self._cell_size,
                                   ry * self._cell_size,
                                   rz * self._cell_size])
                p = np.mod(p, domain)
            return _tpms_sdf(p, _fn, self._cell_size, self._thickness)

        super().__init__(func)


class Neovius3D(SDF3D):
    """Neovius TPMS surface.

    Implicit: ``3*(cos(x)+cos(y)+cos(z)) + 4*cos(x)*cos(y)*cos(z) = 0``

    Parameters
    ----------
    cell_size:
        Physical period length of one unit cell.
    thickness:
        Half-thickness of the solid shell around the iso-surface.
    repeat:
        Optional ``(rx, ry, rz)`` tiling counts.
    """

    def __init__(
        self,
        cell_size: float = 1.0,
        thickness: float = 0.2,
        repeat: Sequence[int] | None = None,
    ) -> None:
        self._cell_size = float(cell_size)
        self._thickness = float(thickness)
        self._repeat = repeat

        def _fn(u, v, w):
            cx, cy, cz = np.cos(u), np.cos(v), np.cos(w)
            return 3.0 * (cx + cy + cz) + 4.0 * cx * cy * cz

        def func(p: np.ndarray) -> np.ndarray:
            if self._repeat is not None:
                rx, ry, rz = self._repeat
                domain = np.array([rx * self._cell_size,
                                   ry * self._cell_size,
                                   rz * self._cell_size])
                p = np.mod(p, domain)
            return _tpms_sdf(p, _fn, self._cell_size, self._thickness)

        super().__init__(func)


# ===========================================================================
# Lattice classes (beam-based)
# ===========================================================================

def _capsule_sdf(p: np.ndarray, a: np.ndarray, b: np.ndarray, r: float) -> np.ndarray:
    """Signed distance to a capsule (cylinder with hemispherical caps).

    Parameters
    ----------
    p:
        ``(..., 3)`` point array.
    a, b:
        Start and end points of the capsule axis.
    r:
        Capsule radius.
    """
    ab = b - a
    ap = p - a
    t = np.clip(
        np.sum(ap * ab, axis=-1) / np.dot(ab, ab),
        0.0, 1.0,
    )
    closest = a + t[..., None] * ab
    return np.linalg.norm(p - closest, axis=-1) - r


class BCCLattice3D(SDF3D):
    """Body-Centred Cubic (BCC) beam lattice.

    Generates a BCC unit cell from 8 corner-to-center beams (capsules).
    The unit cell is tiled over *repeat* periods.

    Parameters
    ----------
    cell_size:
        Physical side length of one cubic unit cell.
    beam_radius:
        Radius of each beam (capsule).
    repeat:
        ``(rx, ry, rz)`` — how many unit cells to tile in each direction.
        Defaults to ``(1, 1, 1)``.
    """

    def __init__(
        self,
        cell_size: float = 1.0,
        beam_radius: float = 0.08,
        repeat: Sequence[int] = (1, 1, 1),
    ) -> None:
        self._cell_size = float(cell_size)
        self._beam_radius = float(beam_radius)
        self._repeat = tuple(int(r) for r in repeat)

        # BCC: 8 corners + 1 body center
        # Corners are at (0,0,0) to (1,1,1) in unit-cell fractions
        _corners = np.array([
            [0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0],
            [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1],
        ], dtype=float)
        _center = np.array([0.5, 0.5, 0.5])

        def func(p: np.ndarray) -> np.ndarray:
            cs = self._cell_size
            rx, ry, rz = self._repeat
            # Map p to within the full tiled domain (0 → repeat*cell_size)
            # then fold into a single unit cell via modulo
            p_mod = np.mod(p, np.array([rx * cs, ry * cs, rz * cs]))
            p_cell = np.mod(p_mod, cs)  # within [0, cell_size)^3

            # Build union of corner-to-center capsules
            corners = _corners * cs
            center = _center * cs
            d = np.full(p_cell.shape[:-1], np.inf)
            for c in corners:
                d = np.minimum(d, _capsule_sdf(p_cell, c, center, self._beam_radius))
            return d

        super().__init__(func)


class FCCLattice3D(SDF3D):
    """Face-Centred Cubic (FCC) beam lattice.

    Generates an FCC unit cell from edge-midpoint-to-corner beams.
    The unit cell is tiled over *repeat* periods.

    Parameters
    ----------
    cell_size:
        Physical side length of one cubic unit cell.
    beam_radius:
        Radius of each beam (capsule).
    repeat:
        ``(rx, ry, rz)`` — how many unit cells to tile in each direction.
        Defaults to ``(1, 1, 1)``.
    """

    def __init__(
        self,
        cell_size: float = 1.0,
        beam_radius: float = 0.08,
        repeat: Sequence[int] = (1, 1, 1),
    ) -> None:
        self._cell_size = float(cell_size)
        self._beam_radius = float(beam_radius)
        self._repeat = tuple(int(r) for r in repeat)

        # FCC: 8 corners + 6 face centers
        _corners = np.array([
            [0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0],
            [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1],
        ], dtype=float)
        _face_centers = np.array([
            [0.5, 0.5, 0.0], [0.5, 0.5, 1.0],
            [0.5, 0.0, 0.5], [0.5, 1.0, 0.5],
            [0.0, 0.5, 0.5], [1.0, 0.5, 0.5],
        ], dtype=float)

        def func(p: np.ndarray) -> np.ndarray:
            cs = self._cell_size
            rx, ry, rz = self._repeat
            p_mod = np.mod(p, np.array([rx * cs, ry * cs, rz * cs]))
            p_cell = np.mod(p_mod, cs)

            corners = _corners * cs
            face_centers = _face_centers * cs

            d = np.full(p_cell.shape[:-1], np.inf)
            # Connect each face center to the 4 corners sharing that face.
            # A face center has exactly one coordinate at 0 or 1 (the "face
            # axis") and the other two at 0.5.  The 4 adjacent corners are
            # those whose coordinate on the face axis matches the face center.
            for fc in face_centers:
                fc_norm = fc / cs
                # Identify the face axis: the one coordinate that is 0 or 1
                face_axis = int(np.argmax(np.abs(fc_norm - 0.5)))
                for c in corners:
                    c_norm = c / cs
                    if abs(fc_norm[face_axis] - c_norm[face_axis]) < 1e-9:
                        d = np.minimum(
                            d, _capsule_sdf(p_cell, fc, c, self._beam_radius)
                        )
            return d

        super().__init__(func)
