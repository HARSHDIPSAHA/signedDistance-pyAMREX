"""3D geometry primitives and boolean operations for signed distance functions."""

from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from typing import TypeAlias

import numpy as np
import numpy.typing as npt

from . import primitives as sdf
from .primitives import Points3D, Distances

SDFFunc: TypeAlias = Callable[[Points3D], Distances]
_Array: TypeAlias = npt.NDArray[np.floating]
_Bounds3D = tuple[tuple[float, float], tuple[float, float], tuple[float, float]]
_Resolution3D = tuple[int, int, int]


def save_npy(path: str, phi: _Array) -> None:
    """Save *phi* array to *path* (creates parent directories if needed)."""
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    np.save(path, phi)


# ===========================================================================
# Base class
# ===========================================================================

class SDF3D:
    """Base class for 3D signed-distance-function geometries.

    A ``SDF3D`` wraps a callable ``func(p) -> distances`` where *p* is
    a ``(..., 3)`` array of 3D points and the return value is a ``(...)``
    array of signed distances.

    Implements:
    - Boolean operations: :meth:`union`, :meth:`subtract`, :meth:`intersect`
    - Modifiers:          :meth:`round`, :meth:`onion`
    - Transforms:         :meth:`translate`, :meth:`scale`, :meth:`elongate`
    - Rotations:          :meth:`rotate_x`, :meth:`rotate_y`, :meth:`rotate_z`
    """

    def __init__(self, func: SDFFunc) -> None:
        self.func = func

    def sdf(self, p: Points3D) -> Distances:
        """Evaluate signed distance at *p* (shape ``(..., 3)``)."""
        return self.func(p)

    def __call__(self, p: Points3D) -> Distances:
        return self.func(p)

    # ------------------------------------------------------------------
    # Boolean operations
    # ------------------------------------------------------------------

    def union(self, other: SDF3D) -> SDF3D:
        """Return the union (min) of this shape and *other*."""
        return SDF3D(lambda p: sdf.opUnion(self.sdf(p), other.sdf(p)))

    def subtract(self, other: SDF3D) -> SDF3D:
        """Subtract *other* from this shape."""
        return SDF3D(lambda p: sdf.opSubtraction(other.sdf(p), self.sdf(p)))

    def intersect(self, other: SDF3D) -> SDF3D:
        """Return the intersection (max) of this shape and *other*."""
        return SDF3D(lambda p: sdf.opIntersection(self.sdf(p), other.sdf(p)))

    # Operator shorthands: A | B → union, A - B → subtract, A / B → intersect
    def __or__(self, other: SDF3D) -> SDF3D:
        return self.union(other)

    def __sub__(self, other: SDF3D) -> SDF3D:
        return self.subtract(other)

    def __truediv__(self, other: SDF3D) -> SDF3D:
        return self.intersect(other)

    # ------------------------------------------------------------------
    # Modifiers
    # ------------------------------------------------------------------

    def round(self, rad: float) -> SDF3D:
        """Round the surface outward by *rad*."""
        return SDF3D(lambda p: sdf.opRound(p, self.sdf, rad))

    def onion(self, thickness: float) -> SDF3D:
        """Turn the solid into a hollow shell of *thickness*."""
        return SDF3D(lambda p: sdf.opOnion(self.sdf(p), thickness))

    # ------------------------------------------------------------------
    # Transforms
    # ------------------------------------------------------------------

    def translate(self, tx: float, ty: float, tz: float) -> SDF3D:
        """Translate by ``(tx, ty, tz)``."""
        t = np.array([tx, ty, tz])
        return SDF3D(lambda p: self.sdf(p - t))

    def scale(self, s: float) -> SDF3D:
        """Uniformly scale by factor *s*."""
        return SDF3D(lambda p: sdf.opScale(p, s, self.sdf))

    def elongate(self, hx: float, hy: float, hz: float) -> SDF3D:
        """Elongate along each axis by ``(hx, hy, hz)``."""
        h = np.array([hx, hy, hz])
        return SDF3D(lambda p: sdf.opElongate2(p, self.sdf, h))

    def rotate_x(self, angle_rad: float) -> SDF3D:
        """Rotate around the X axis by *angle_rad* radians."""
        c = np.cos(angle_rad)
        s = np.sin(angle_rad)
        rot = np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])
        return SDF3D(lambda p: sdf.opTx(p, rot, np.zeros(3), self.sdf))

    def rotate_y(self, angle_rad: float) -> SDF3D:
        """Rotate around the Y axis by *angle_rad* radians."""
        c = np.cos(angle_rad)
        s = np.sin(angle_rad)
        rot = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
        return SDF3D(lambda p: sdf.opTx(p, rot, np.zeros(3), self.sdf))

    def rotate_z(self, angle_rad: float) -> SDF3D:
        """Rotate around the Z axis by *angle_rad* radians."""
        c = np.cos(angle_rad)
        s = np.sin(angle_rad)
        rot = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        return SDF3D(lambda p: sdf.opTx(p, rot, np.zeros(3), self.sdf))

    # ------------------------------------------------------------------
    # Grid and AMReX output
    # ------------------------------------------------------------------

    def to_array(self, bounds: _Bounds3D, resolution: _Resolution3D) -> _Array:
        """Sample this SDF on a uniform 3-D cell-centred grid.

        Parameters
        ----------
        bounds:
            ``((x0, x1), (y0, y1), (z0, z1))`` physical extents of the domain.
        resolution:
            ``(nx, ny, nz)`` number of cells along each axis.

        Returns
        -------
        numpy.ndarray
            Shape ``(nz, ny, nx)`` array of signed distances, z-first indexing.
        """
        (x0, x1), (y0, y1), (z0, z1) = bounds
        nx, ny, nz = resolution
        xs = np.linspace(x0, x1, nx, endpoint=False) + (x1 - x0) / (2.0 * nx)
        ys = np.linspace(y0, y1, ny, endpoint=False) + (y1 - y0) / (2.0 * ny)
        zs = np.linspace(z0, z1, nz, endpoint=False) + (z1 - z0) / (2.0 * nz)
        Z, Y, X = np.meshgrid(zs, ys, xs, indexing="ij")
        p = np.stack([X, Y, Z], axis=-1)
        return self.sdf(p)

    def fill(self, grid) -> "amr.MultiFab":
        """Fill *grid* with this SDF and return a raw ``amrex.space3d.MultiFab``.

        Parameters
        ----------
        grid:
            A :class:`~sdf3d.amrex.MultiFabGrid3D` instance that defines the
            domain layout (geom, ba, dm).

        Returns
        -------
        amrex.space3d.MultiFab
            A single-component MultiFab filled with signed distance values.
        """
        mf = grid.create_multifab()
        grid.fill_multifab(mf, self.sdf)
        return mf

    def to_multifab(self, amrex_geom, ba, dm) -> "amr.MultiFab":
        """Convenience: create a :class:`~sdf3d.amrex.MultiFabGrid3D` and fill.

        Equivalent to::

            grid = MultiFabGrid3D(amrex_geom, ba, dm)
            mf   = shape.fill(grid)

        Returns
        -------
        amrex.space3d.MultiFab
        """
        from .amrex import MultiFabGrid3D
        return self.fill(MultiFabGrid3D(amrex_geom, ba, dm))

    # ------------------------------------------------------------------
    # Visualisation helpers
    # ------------------------------------------------------------------

    def save_png(
        self,
        path,
        *,
        bounds=((-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0)),
        resolution=(64, 64, 64),
        title: str = "",
        color=(0.9, 0.7, 0.2),
        elev: float = 25.0,
        azim: float = 35.0,
    ) -> None:
        """Sample, extract the zero isosurface, and save a shaded PNG.

        Parameters
        ----------
        path:
            Output file path (``str`` or :class:`pathlib.Path`); parent directory
            is created automatically.
        bounds:
            ``((x0,x1),(y0,y1),(z0,z1))`` domain extents.
        resolution:
            ``(nx,ny,nz)`` grid resolution (default 64³).
        title:
            Figure title.
        color:
            RGB tuple in [0,1] for the shading base colour.
        elev, azim:
            Matplotlib 3-D view angles in degrees.
        """
        from pathlib import Path
        path = Path(path)
        if path.parent == Path('.'):
            path = Path("output") / path
        path.parent.mkdir(parents=True, exist_ok=True)

        try:
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d.art3d import Poly3DCollection
            from skimage import measure
        except ImportError as exc:
            print(f"  save_png: {exc} — skipping")
            return

        phi = self.to_array(bounds, resolution)
        if phi.min() >= 0 or phi.max() <= 0:
            print(f"  save_png: no zero crossing — skipping {path.name}")
            return

        (x0, x1), (y0, y1), (z0, z1) = bounds
        nz, ny, nx = phi.shape
        spacing = ((z1 - z0) / nz, (y1 - y0) / ny, (x1 - x0) / nx)
        verts, faces, _, _ = measure.marching_cubes(phi, level=0, spacing=spacing)
        # phi is (nz,ny,nx) so marching_cubes verts[:,0]=z, [:,1]=y, [:,2]=x;
        # reorder to (x,y,z) for matplotlib, then shift to physical coordinates.
        verts = verts[:, [2, 1, 0]] + np.array([x0, y0, z0])

        tris  = verts[faces]
        norms = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
        nlen  = np.linalg.norm(norms, axis=1, keepdims=True)
        norms = norms / np.where(nlen > 0, nlen, 1.0)
        # abs-dot for winding-independent lighting (odd permutation above flips normals)
        shade = 0.3 + 0.7 * np.abs(norms @ np.array([0.577, 0.577, 0.577]))
        fc    = np.column_stack([
            shade * color[0], shade * color[1], shade * color[2], np.ones_like(shade)
        ])

        fig = plt.figure(figsize=(5, 5), facecolor="#111")
        ax  = fig.add_subplot(111, projection="3d")
        ax.set_facecolor("#111")
        ax.set_axis_off()
        ax.set_box_aspect([1, 1, 1])
        ax.add_collection3d(Poly3DCollection(verts[faces], facecolors=fc, edgecolors="none"))
        ax.set_xlim(x0, x1); ax.set_ylim(y0, y1); ax.set_zlim(z0, z1)
        ax.set_title(title, color="white", fontsize=10)
        ax.view_init(elev=elev, azim=azim)
        plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="#111")
        plt.close()
        print(f"  Saved: {path}")

    def save_plotly_html(
        self,
        path,
        *,
        bounds=((-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0)),
        resolution=(40, 40, 40),
        title: str = "",
        color: str = "#4a90d9",
    ) -> None:
        """Sample and save an interactive Plotly HTML isosurface.

        Parameters
        ----------
        path:
            Output ``str`` or :class:`pathlib.Path`; parent dir created
            automatically.
        bounds:
            ``((x0,x1),(y0,y1),(z0,z1))`` domain extents.
        resolution:
            ``(nx,ny,nz)`` grid resolution (default 40³).
        title:
            Figure title shown in the HTML page.
        color:
            Hex colour string for the isosurface.
        """
        save_plotly_html_grid(
            [(self, title, bounds)],
            path,
            resolution=resolution,
            title=title,
            color=color,
        )


# ===========================================================================
# Primitive shapes
# ===========================================================================

class Sphere3D(SDF3D):
    """Sphere centred at origin with given *radius*."""

    def __init__(self, radius: float) -> None:
        super().__init__(lambda p: sdf.sdSphere(p, radius))


class Box3D(SDF3D):
    """Axis-aligned box with *half_size* ``(hx, hy, hz)`` centred at origin."""

    def __init__(self, half_size: Sequence[float]) -> None:
        b = np.array(half_size, dtype=float)
        super().__init__(lambda p: sdf.sdBox(p, b))


class RoundBox3D(SDF3D):
    """Axis-aligned box with corner *radius* and *half_size* ``(hx, hy, hz)``."""

    def __init__(self, half_size: Sequence[float], radius: float) -> None:
        b = np.array(half_size, dtype=float)
        super().__init__(lambda p: sdf.sdRoundBox(p, b, radius))


class Cylinder3D(SDF3D):
    """Infinite cylinder.

    Parameters
    ----------
    axis_offset:
        ``(cx, cz)`` — offset of the axis in XZ-plane.
    radius:
        Cylinder radius.
    """

    def __init__(self, axis_offset: Sequence[float], radius: float) -> None:
        c = np.array([axis_offset[0], axis_offset[1], radius], dtype=float)
        super().__init__(lambda p: sdf.sdCylinder(p, c))


class ConeExact3D(SDF3D):
    """Exact (signed) cone.

    Parameters
    ----------
    sincos:
        ``(sin(half_angle), cos(half_angle))`` of the cone's apex angle.
    height:
        Height of the cone along Y.
    """

    def __init__(self, sincos: Sequence[float], height: float) -> None:
        c = np.array(sincos, dtype=float)
        super().__init__(lambda p: sdf.sdConeExact(p, c, height))


class Torus3D(SDF3D):
    """Torus in the XZ plane.

    Parameters
    ----------
    major_minor:
        ``(R, r)`` where *R* is the major (tube-centre) radius and
        *r* is the minor (tube-cross-section) radius.
    """

    def __init__(self, major_minor: Sequence[float]) -> None:
        t = np.array(major_minor, dtype=float)
        super().__init__(lambda p: sdf.sdTorus(p, t))




# ===========================================================================
# Module-level visualisation helpers
# ===========================================================================

def save_plotly_html_grid(
    panels,
    path,
    *,
    bounds=((-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0)),
    resolution=(40, 40, 40),
    title: str = "",
    color: str = "#4a90d9",
) -> None:
    """Render multiple geometries side-by-side in a single Plotly HTML file.

    Parameters
    ----------
    panels:
        A list of ``(geom, label)`` or ``(geom, label, per_panel_bounds)``
        tuples.  When a panel supplies its own bounds they override the
        *bounds* argument for that panel.
    path:
        Output ``str`` or :class:`pathlib.Path`; parent dir is created
        automatically.
    bounds:
        Default ``((x0,x1),(y0,y1),(z0,z1))`` used when a panel omits its
        own bounds.
    resolution:
        ``(nx,ny,nz)`` grid resolution applied to every panel.
    title:
        Overall figure title.
    color:
        Hex colour string for all isosurfaces.
    """
    from pathlib import Path

    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError as exc:
        print(f"  save_plotly_html_grid: {exc} — skipping")
        return

    path = Path(path)
    if path.parent == Path('.'):
        path = Path("output") / path
    path.parent.mkdir(parents=True, exist_ok=True)

    n = len(panels)
    fig = make_subplots(
        rows=1,
        cols=n,
        specs=[[{"type": "scene"}] * n],
        subplot_titles=[p[1] for p in panels],
        horizontal_spacing=0.06,
    )

    for col, panel in enumerate(panels, start=1):
        geom  = panel[0]
        panel_bounds = panel[2] if len(panel) >= 3 else bounds
        nx, ny, nz = resolution
        (x0, x1), (y0, y1), (z0, z1) = panel_bounds

        phi = geom.to_array(panel_bounds, resolution)

        xs = np.linspace(x0, x1, nx)
        ys = np.linspace(y0, y1, ny)
        zs = np.linspace(z0, z1, nz)
        Z3, Y3, X3 = np.meshgrid(zs, ys, xs, indexing="ij")

        fig.add_trace(
            go.Isosurface(
                x=X3.ravel(), y=Y3.ravel(), z=Z3.ravel(),
                value=phi.ravel(),
                isomin=0.0, isomax=0.0, surface_count=1,
                colorscale=[[0, color], [1, color]],
                showscale=False,
                caps=dict(x_show=False, y_show=False, z_show=False),
                lighting=dict(ambient=0.5, diffuse=0.8, specular=0.4, roughness=0.3),
                lightposition=dict(x=1000, y=1000, z=2000),
            ),
            row=1, col=col,
        )

    fig.update_scenes(aspectmode="data")
    fig.update_layout(
        title=dict(text=title, font=dict(size=18)),
        width=600 * n,
        height=600,
        paper_bgcolor="#1a1a2e",
        plot_bgcolor="#16213e",
        font=dict(color="#e0e0e0"),
    )

    fig.write_html(path, include_plotlyjs="cdn")
    print(f"  Saved: {path}")
