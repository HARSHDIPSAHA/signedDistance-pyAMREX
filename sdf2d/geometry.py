"""2D geometry primitives and boolean operations for signed distance functions."""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
import numpy.typing as npt

from . import primitives as sdf

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
_Array = npt.NDArray[np.floating]
_SDFFunc = Callable[[_Array], _Array]


# ===========================================================================
# Base class
# ===========================================================================

class SDF2D:
    """Base class for 2D signed-distance-function geometries.

    A ``SDF2D`` wraps a callable ``func(p) -> distances`` where *p* is
    a ``(..., 2)`` array of 2D points and the return value is a ``(...)``
    array of signed distances.

    Subclasses override ``__init__`` to pass the appropriate primitive SDF to
    ``super().__init__(func)``.

    Implements:
    - Boolean operations: :meth:`union`, :meth:`subtract`, :meth:`intersect`
    - Modifiers:          :meth:`round`, :meth:`onion`
    - Transforms:         :meth:`translate`, :meth:`scale`, :meth:`rotate`
    """

    def __init__(self, func: _SDFFunc) -> None:
        self._func = func

    def sdf(self, p: _Array) -> _Array:
        """Evaluate signed distance at *p* (shape ``(..., 2)``)."""
        return self._func(p)

    def __call__(self, p: _Array) -> _Array:
        return self._func(p)

    # ------------------------------------------------------------------
    # Boolean operations
    # ------------------------------------------------------------------

    def union(self, other: SDF2D) -> SDF2D:
        """Return the union (min) of this shape and *other*."""
        return SDF2D(lambda p: sdf.opUnion(self.sdf(p), other.sdf(p)))

    def subtract(self, other: SDF2D) -> SDF2D:
        """Subtract *other* from this shape."""
        return SDF2D(lambda p: sdf.opSubtraction(other.sdf(p), self.sdf(p)))

    def intersect(self, other: SDF2D) -> SDF2D:
        """Return the intersection (max) of this shape and *other*."""
        return SDF2D(lambda p: sdf.opIntersection(self.sdf(p), other.sdf(p)))

    # ------------------------------------------------------------------
    # Modifiers
    # ------------------------------------------------------------------

    def round(self, rad: float) -> SDF2D:
        """Round the surface outward by *rad*."""
        return SDF2D(lambda p: sdf.opRound(p, self.sdf, rad))

    def onion(self, thickness: float) -> SDF2D:
        """Turn the solid into a hollow shell of *thickness*."""
        return SDF2D(lambda p: sdf.opOnion(self.sdf(p), thickness))

    # ------------------------------------------------------------------
    # Transforms
    # ------------------------------------------------------------------

    def translate(self, tx: float, ty: float) -> SDF2D:
        """Translate by ``(tx, ty)``."""
        t = np.array([tx, ty])
        return SDF2D(lambda p: self.sdf(p - t))

    def scale(self, s: float) -> SDF2D:
        """Uniformly scale by factor *s*."""
        return SDF2D(lambda p: sdf.opScale(p, s, self.sdf))

    def rotate(self, angle_rad: float) -> SDF2D:
        """Rotate by *angle_rad* radians (counter-clockwise)."""
        c = np.cos(angle_rad)
        s = np.sin(angle_rad)
        rot = np.array([[c, -s], [s, c]])
        return SDF2D(lambda p: sdf.opTx2D(p, rot, np.zeros(2), self.sdf))

    # ------------------------------------------------------------------
    # Visualisation helpers
    # ------------------------------------------------------------------

    def save_png(
        self,
        path,
        *,
        bounds=((-1.0, 1.0), (-1.0, 1.0)),
        resolution=(512, 512),
        title: str = "",
    ) -> None:
        """Sample the SDF on a 2-D grid and save a heatmap PNG.

        Parameters
        ----------
        path:
            Output ``str`` or :class:`pathlib.Path`; parent dir created
            automatically.
        bounds:
            ``((x0,x1),(y0,y1))`` domain extents.
        resolution:
            ``(nx,ny)`` grid resolution (default 512×512).
        title:
            Figure title.
        """
        from pathlib import Path
        import warnings

        path = Path(path)
        if path.parent == Path('.'):
            path = Path("output") / path
        path.parent.mkdir(parents=True, exist_ok=True)

        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:
            print(f"  save_png: {exc} — skipping")
            return

        from .grid import sample_levelset_2d

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            phi = sample_levelset_2d(self, bounds, resolution)

        (x0, x1), (y0, y1) = bounds
        extent = [x0, x1, y0, y1]
        lim = max(float(np.nanmax(np.abs(phi))), 1e-6)

        fig, ax = plt.subplots(figsize=(5, 5), facecolor="#111")
        ax.set_facecolor("#111")
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor("#444")
        ax.imshow(phi, origin="lower", extent=extent,
                  cmap="seismic", vmin=-lim, vmax=lim, interpolation="bilinear")
        try:
            ax.contour(phi, levels=[0.0], colors="white", linewidths=1.0,
                       extent=extent)
        except Exception:
            pass
        ax.set_title(title, color="white", fontsize=10)
        plt.tight_layout(pad=0.3)
        fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="#111")
        plt.close(fig)
        print(f"  Saved: {path}")


# ===========================================================================
# Primitive shapes
# ===========================================================================

class Circle2D(SDF2D):
    """Circle centred at origin with given *radius*."""

    def __init__(self, radius: float) -> None:
        super().__init__(lambda p: sdf.sdCircle(p, radius))


class Box2D(SDF2D):
    """Axis-aligned rectangle with *half_size* ``(hx, hy)`` centred at origin."""

    def __init__(self, half_size: Sequence[float]) -> None:
        b = np.array(half_size, dtype=float)
        super().__init__(lambda p: sdf.sdBox2D(p, b))


class RoundedBox2D(SDF2D):
    """Axis-aligned rectangle with corner *radius* and *half_size* ``(hx, hy)``."""

    def __init__(self, half_size: Sequence[float], radius: float) -> None:
        b = np.array(half_size, dtype=float)
        super().__init__(lambda p: sdf.sdRoundedBox2D(p, b, radius))


class OrientedBox2D(SDF2D):
    """An oriented (rotated) box from *corner_a* to *corner_b* with *thickness*."""

    def __init__(
        self,
        corner_a: Sequence[float],
        corner_b: Sequence[float],
        thickness: float,
    ) -> None:
        a = np.array(corner_a, dtype=float)
        b = np.array(corner_b, dtype=float)
        super().__init__(lambda p: sdf.sdOrientedBox2D(p, a, b, thickness))


class Segment2D(SDF2D):
    """Line segment from *point_a* to *point_b* (zero-width)."""

    def __init__(
        self, point_a: Sequence[float], point_b: Sequence[float]
    ) -> None:
        a = np.array(point_a, dtype=float)
        b = np.array(point_b, dtype=float)
        super().__init__(lambda p: sdf.sdSegment2D(p, a, b))


class Rhombus2D(SDF2D):
    """Rhombus (diamond) with *half_size* ``(hx, hy)``."""

    def __init__(self, half_size: Sequence[float]) -> None:
        b = np.array(half_size, dtype=float)
        super().__init__(lambda p: sdf.sdRhombus2D(p, b))


class Trapezoid2D(SDF2D):
    """Isosceles trapezoid with base radii *r1*, *r2* and *height*."""

    def __init__(self, r1: float, r2: float, height: float) -> None:
        super().__init__(lambda p: sdf.sdTrapezoid2D(p, r1, r2, height))


class Parallelogram2D(SDF2D):
    """Parallelogram with *width*, *height*, and horizontal *skew*."""

    def __init__(self, width: float, height: float, skew: float) -> None:
        super().__init__(lambda p: sdf.sdParallelogram2D(p, width, height, skew))


# ===========================================================================
# Triangles
# ===========================================================================

class EquilateralTriangle2D(SDF2D):
    """Equilateral triangle with circumradius *radius*."""

    def __init__(self, radius: float) -> None:
        super().__init__(lambda p: sdf.sdEquilateralTriangle2D(p, radius))


class TriangleIsosceles2D(SDF2D):
    """Isosceles triangle with half-base *width* and *height*."""

    def __init__(self, width: float, height: float) -> None:
        q = np.array([width, height], dtype=float)
        super().__init__(lambda p: sdf.sdTriangleIsosceles2D(p, q))


class Triangle2D(SDF2D):
    """Arbitrary triangle from three 2-D vertices."""

    def __init__(
        self,
        p0: Sequence[float],
        p1: Sequence[float],
        p2: Sequence[float],
    ) -> None:
        v0 = np.array(p0, dtype=float)
        v1 = np.array(p1, dtype=float)
        v2 = np.array(p2, dtype=float)
        super().__init__(lambda p: sdf.sdTriangle2D(p, v0, v1, v2))


# ===========================================================================
# Capsules
# ===========================================================================

class UnevenCapsule2D(SDF2D):
    """Capsule with different radii *r1* (bottom) and *r2* (top), and *height*."""

    def __init__(self, r1: float, r2: float, height: float) -> None:
        super().__init__(lambda p: sdf.sdUnevenCapsule2D(p, r1, r2, height))


# ===========================================================================
# Regular polygons
# ===========================================================================

class Pentagon2D(SDF2D):
    """Regular pentagon with circumradius *radius*."""

    def __init__(self, radius: float) -> None:
        super().__init__(lambda p: sdf.sdPentagon2D(p, radius))


class Hexagon2D(SDF2D):
    """Regular hexagon with circumradius *radius*."""

    def __init__(self, radius: float) -> None:
        super().__init__(lambda p: sdf.sdHexagon2D(p, radius))


class Octagon2D(SDF2D):
    """Regular octagon with circumradius *radius*."""

    def __init__(self, radius: float) -> None:
        super().__init__(lambda p: sdf.sdOctagon2D(p, radius))


class NGon2D(SDF2D):
    """Regular N-sided polygon with *n_sides* and circumradius *radius*."""

    def __init__(self, radius: float, n_sides: int) -> None:
        super().__init__(lambda p: sdf.sdNGon2D(p, radius, n_sides))


# ===========================================================================
# Stars
# ===========================================================================

class Hexagram2D(SDF2D):
    """6-pointed star (Star of David) with *radius*."""

    def __init__(self, radius: float) -> None:
        super().__init__(lambda p: sdf.sdHexagram2D(p, radius))


class Star2D(SDF2D):
    """N-pointed star with *radius* and *n_points* points.

    Parameters
    ----------
    factor:
        Controls pointiness; must satisfy ``2 ≤ factor ≤ n_points``.
        ``factor=2`` gives the sharpest star; ``factor=n_points`` gives a
        regular polygon.
    """

    def __init__(self, radius: float, n_points: int, factor: float) -> None:
        super().__init__(lambda p: sdf.sdStar(p, radius, n_points, factor))


# ===========================================================================
# Arcs and sectors
# ===========================================================================

class Pie2D(SDF2D):
    """Circular pie/sector.

    Parameters
    ----------
    angle_sc:
        ``(sin(half_angle), cos(half_angle))`` of the opening half-angle.
    radius:
        Outer radius.
    """

    def __init__(self, angle_sc: Sequence[float], radius: float) -> None:
        c = np.array(angle_sc, dtype=float)
        super().__init__(lambda p: sdf.sdPie2D(p, c, radius))


class CutDisk2D(SDF2D):
    """Circle with a straight cut at height *cut_height*."""

    def __init__(self, radius: float, cut_height: float) -> None:
        super().__init__(lambda p: sdf.sdCutDisk2D(p, radius, cut_height))


class Arc2D(SDF2D):
    """Arc (bent line) with given *angle_sc*, outer *radius*, and *thickness*.

    Parameters
    ----------
    angle_sc:
        ``(sin(half_angle), cos(half_angle))`` of the arc's opening half-angle.
    """

    def __init__(
        self, angle_sc: Sequence[float], radius: float, thickness: float
    ) -> None:
        sc = np.array(angle_sc, dtype=float)
        super().__init__(lambda p: sdf.sdArc2D(p, sc, radius, thickness))


class Ring2D(SDF2D):
    """Annulus (ring) between *inner_radius* and *outer_radius*."""

    def __init__(self, inner_radius: float, outer_radius: float) -> None:
        super().__init__(lambda p: sdf.sdRing2D(p, inner_radius, outer_radius))


class Horseshoe2D(SDF2D):
    """Horseshoe shape.

    Parameters
    ----------
    angle_sc:
        ``(sin(half_angle), cos(half_angle))`` of the gap half-angle.
    radius:
        Arc radius.
    thickness_vec2:
        ``(inner_thickness, outer_thickness)`` of the horseshoe arms.
    """

    def __init__(
        self,
        angle_sc: Sequence[float],
        radius: float,
        thickness_vec2: Sequence[float],
    ) -> None:
        c = np.array(angle_sc, dtype=float)
        w = np.array(thickness_vec2, dtype=float)
        super().__init__(lambda p: sdf.sdHorseshoe2D(p, c, radius, w))


# ===========================================================================
# Special shapes
# ===========================================================================

class Vesica2D(SDF2D):
    """Vesica piscis (lens/eye shape): intersection of two circles.

    Parameters
    ----------
    radius:
        Radius of both constituent circles.
    distance:
        Half-distance between the two circle centres.
    """

    def __init__(self, radius: float, distance: float) -> None:
        super().__init__(lambda p: sdf.sdVesica2D(p, radius, distance))


class Moon2D(SDF2D):
    """Crescent moon shape.

    Parameters
    ----------
    distance:
        Offset between the two overlapping circles.
    radius_a:
        Outer circle radius.
    radius_b:
        Inner (subtracted) circle radius.
    """

    def __init__(self, distance: float, radius_a: float, radius_b: float) -> None:
        super().__init__(lambda p: sdf.sdMoon2D(p, distance, radius_a, radius_b))


class RoundedCross2D(SDF2D):
    """Rounded cross of *size*."""

    def __init__(self, size: float) -> None:
        super().__init__(lambda p: sdf.sdRoundedCross2D(p, size))


class Egg2D(SDF2D):
    """Egg shape with two radii *radius_a* and *radius_b*."""

    def __init__(self, radius_a: float, radius_b: float) -> None:
        super().__init__(lambda p: sdf.sdEgg2D(p, radius_a, radius_b))


class Heart2D(SDF2D):
    """Heart shape (unit-scale, centred at origin)."""

    def __init__(self) -> None:
        super().__init__(lambda p: sdf.sdHeart2D(p))


class Cross2D(SDF2D):
    """Symmetric plus-sign cross.

    Parameters
    ----------
    size_vec2:
        ``(half_arm_length, half_arm_width)`` of the cross arms.
    rounding:
        Corner rounding radius.
    """

    def __init__(self, size_vec2: Sequence[float], rounding: float) -> None:
        b = np.array(size_vec2, dtype=float)
        super().__init__(lambda p: sdf.sdCross2D(p, b, rounding))


class RoundedX2D(SDF2D):
    """X-shape (cross at 45°) with *width* and corner *rounding*."""

    def __init__(self, width: float, rounding: float) -> None:
        super().__init__(lambda p: sdf.sdRoundedX2D(p, width, rounding))


# ===========================================================================
# Complex / parametric shapes
# ===========================================================================

class Polygon2D(SDF2D):
    """Arbitrary convex or concave polygon from N 2-D *vertices*."""

    def __init__(self, vertices: Sequence[Sequence[float]]) -> None:
        v = np.array(vertices, dtype=float)
        super().__init__(lambda p: sdf.sdPolygon2D(p, v))


class Ellipse2D(SDF2D):
    """Ellipse with semi-axes *semi_axes* ``(a, b)``."""

    def __init__(self, semi_axes: Sequence[float]) -> None:
        ab = np.array(semi_axes, dtype=float)
        super().__init__(lambda p: sdf.sdEllipse2D(p, ab))


class Parabola2D(SDF2D):
    """Parabola ``y = k·x²`` with *curvature* k."""

    def __init__(self, curvature: float) -> None:
        super().__init__(lambda p: sdf.sdParabola2D(p, curvature))


class ParabolaSegment2D(SDF2D):
    """Bounded parabolic segment with half-*width* and *height*."""

    def __init__(self, width: float, height: float) -> None:
        super().__init__(lambda p: sdf.sdParabolaSegment2D(p, width, height))


class Bezier2D(SDF2D):
    """Quadratic Bézier curve from control points *p0*, *p1* (control), *p2*."""

    def __init__(
        self,
        p0: Sequence[float],
        p1: Sequence[float],
        p2: Sequence[float],
    ) -> None:
        A = np.array(p0, dtype=float)
        B = np.array(p1, dtype=float)
        C = np.array(p2, dtype=float)
        super().__init__(lambda p: sdf.sdBezier2D(p, A, B, C))


class BlobbyCross2D(SDF2D):
    """Blobby cross shape of *size*."""

    def __init__(self, size: float) -> None:
        super().__init__(lambda p: sdf.sdBlobbyCross2D(p, size))


class Tunnel2D(SDF2D):
    """Tunnel/arch with *size_vec2* ``(half_width, height)``."""

    def __init__(self, size_vec2: Sequence[float]) -> None:
        wh = np.array(size_vec2, dtype=float)
        super().__init__(lambda p: sdf.sdTunnel2D(p, wh))


class Stairs2D(SDF2D):
    """Staircase with *step_size* ``(width, height)`` and *num_steps* steps."""

    def __init__(
        self, step_size: Sequence[float], num_steps: int
    ) -> None:
        wh = np.array(step_size, dtype=float)
        super().__init__(lambda p: sdf.sdStairs2D(p, wh, num_steps))


class QuadraticCircle2D(SDF2D):
    """Quadratic-circle approximation (unit-scale, centred at origin)."""

    def __init__(self) -> None:
        super().__init__(lambda p: sdf.sdQuadraticCircle2D(p))


class Hyperbola2D(SDF2D):
    """Hyperbola with *curvature* and half-*height*."""

    def __init__(self, curvature: float, height: float) -> None:
        super().__init__(lambda p: sdf.sdHyperbola2D(p, curvature, height))


# ===========================================================================
# Boolean operation classes
# ===========================================================================

class Union2D(SDF2D):
    """Union of two or more 2-D geometries (minimum SDF)."""

    def __init__(self, *geoms: SDF2D) -> None:
        def _sdf(p: _Array) -> _Array:
            d = geoms[0].sdf(p)
            for g in geoms[1:]:
                d = sdf.opUnion(d, g.sdf(p))
            return d

        super().__init__(_sdf)


class Intersection2D(SDF2D):
    """Intersection of two or more 2-D geometries (maximum SDF)."""

    def __init__(self, *geoms: SDF2D) -> None:
        def _sdf(p: _Array) -> _Array:
            d = geoms[0].sdf(p)
            for g in geoms[1:]:
                d = sdf.opIntersection(d, g.sdf(p))
            return d

        super().__init__(_sdf)


class Subtraction2D(SDF2D):
    """Subtract *cutter* from *base*."""

    def __init__(self, base: SDF2D, cutter: SDF2D) -> None:
        super().__init__(
            lambda p: sdf.opSubtraction(cutter.sdf(p), base.sdf(p))
        )
