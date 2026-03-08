"""AMReX MultiFab integration for 2-D SDF fields.

This module requires ``amrex.space2d`` (pyAMReX built in 2-D mode).
It is intentionally kept separate so that the rest of ``sdf2d`` works
without AMReX installed.
"""

from typing import TYPE_CHECKING, Tuple

import numpy as np
import numpy.typing as npt

from . import primitives as sdf

if TYPE_CHECKING:
    import amrex.space2d as amr  # noqa: F401 — type-checker only
    from .geometry import SDF2D

_Array = npt.NDArray[np.floating]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_component_view(arr: _Array) -> _Array:
    """Return the scalar-component view of an AMReX numpy array."""
    if arr.ndim == 3:
        return arr[:, :, 0]
    return arr[:, :, 0, 0]


def _copy_mf_like(src: "amr.MultiFab", ba: "amr.BoxArray", dm: "amr.DistributionMapping") -> "amr.MultiFab":
    """Create a new :class:`amr.MultiFab` with the same layout as *src* and copy values."""
    import amrex.space2d as amr  # local import keeps the module importable without AMReX

    ncomp: int = src.n_comp() if callable(getattr(src, "n_comp", None)) else src.n_comp
    if callable(getattr(src, "n_grow", None)):
        ngrow = src.n_grow()
    elif hasattr(src, "nGrow"):
        ngrow = src.nGrow()
    else:
        ngrow = 0

    mf = amr.MultiFab(ba, dm, ncomp, ngrow)
    for mfi in src:
        arr_src = src.array(mfi).to_numpy()
        arr_dst = mf.array(mfi).to_numpy()
        _get_component_view(arr_dst)[...] = _get_component_view(arr_src)
    return mf


# ---------------------------------------------------------------------------
# Public class
# ---------------------------------------------------------------------------

class SDFMultiFab2D:
    """A bound factory that fills AMReX MultiFabs with 2-D SDF values.

    Construct it once with the AMReX grid layout (domain, box decomposition,
    MPI distribution); all subsequent calls reuse that layout automatically::

        lib = SDFMultiFab2D(geom, ba, dm)   # holds the grid layout
        mf  = lib.from_geometry(Circle2D(0.3))  # creates + fills a MultiFab

    Think of it like a database connection object: open it once, then call
    methods on it as many times as you need.  ``from_geometry`` is the main
    entry point; it is a thin wrapper over the two lower-level primitives::

        mf = lib.create_multifab()          # allocate empty MultiFab
        lib.fill_multifab(mf, geom.sdf)     # write SDF values into it

    Call :meth:`fill_multifab` directly when you want to reuse an existing
    MultiFab or supply a raw SDF callable instead of a geometry object.

    Parameters
    ----------
    geom:
        An ``amr.Geometry`` object that defines the physical domain.
    ba:
        ``amr.BoxArray`` describing the domain decomposition.
    dm:
        ``amr.DistributionMapping`` assigning boxes to MPI ranks.

    Implemented operations
    ----------------------
    :meth:`from_geometry`, :meth:`union`, :meth:`subtract`, :meth:`intersect`,
    :meth:`negate`
    """

    def __init__(
        self,
        geom: "amr.Geometry",
        ba: "amr.BoxArray",
        dm: "amr.DistributionMapping",
    ) -> None:
        self.geom = geom
        self.ba = ba
        self.dm = dm

    # ------------------------------------------------------------------
    # MultiFab factory
    # ------------------------------------------------------------------

    def create_multifab(self) -> "amr.MultiFab":
        """Return an uninitialised single-component MultiFab with no ghost cells."""
        import amrex.space2d as amr
        return amr.MultiFab(self.ba, self.dm, 1, 0)

    # ------------------------------------------------------------------
    # Geometry → MultiFab
    # ------------------------------------------------------------------

    def from_geometry(self, geometry: "SDF2D") -> "amr.MultiFab":
        """Evaluate *geometry* on the AMReX grid and return a MultiFab.

        This is a thin convenience wrapper — equivalent to:
            mf = lib.create_multifab()
            lib.fill_multifab(mf, geometry.sdf)

        Call :meth:`fill_multifab` directly when you need to reuse an
        existing MultiFab or supply a raw SDF callable instead of a geometry
        object.
        """
        mf = self.create_multifab()
        self.fill_multifab(mf, geometry.sdf)
        return mf

    # ------------------------------------------------------------------
    # Boolean operations on MultiFabs
    # ------------------------------------------------------------------

    def union(self, a: "amr.MultiFab", b: "amr.MultiFab") -> "amr.MultiFab":
        """Return ``min(a, b)`` element-wise."""
        out = _copy_mf_like(a, self.ba, self.dm)
        for mfi in out:
            arr_out = out.array(mfi).to_numpy()
            arr_b = b.array(mfi).to_numpy()
            _get_component_view(arr_out)[...] = sdf.opUnion(
                _get_component_view(arr_out), _get_component_view(arr_b)
            )
        return out

    def subtract(self, a: "amr.MultiFab", b: "amr.MultiFab") -> "amr.MultiFab":
        """Return ``max(-a, b)`` element-wise (subtract *b* from *a*)."""
        out = _copy_mf_like(a, self.ba, self.dm)
        for mfi in out:
            arr_out = out.array(mfi).to_numpy()
            arr_b = b.array(mfi).to_numpy()
            _get_component_view(arr_out)[...] = sdf.opSubtraction(
                _get_component_view(arr_out), _get_component_view(arr_b)
            )
        return out

    def intersect(self, a: "amr.MultiFab", b: "amr.MultiFab") -> "amr.MultiFab":
        """Return ``max(a, b)`` element-wise."""
        out = _copy_mf_like(a, self.ba, self.dm)
        for mfi in out:
            arr_out = out.array(mfi).to_numpy()
            arr_b = b.array(mfi).to_numpy()
            _get_component_view(arr_out)[...] = sdf.opIntersection(
                _get_component_view(arr_out), _get_component_view(arr_b)
            )
        return out

    def negate(self, a: "amr.MultiFab") -> "amr.MultiFab":
        """Negate *a* element-wise (flip inside/outside)."""
        out = _copy_mf_like(a, self.ba, self.dm)
        for mfi in out:
            arr_out = out.array(mfi).to_numpy()
            _get_component_view(arr_out)[...] *= -1.0
        return out

    # ------------------------------------------------------------------
    # Grid fill
    # ------------------------------------------------------------------

    def fill_multifab(
        self,
        mf: "amr.MultiFab",
        sdf_func: "_SDFFunc",  # type: ignore[name-defined]
    ) -> None:
        dx = self.geom.data().CellSize()
        if hasattr(self.geom, "ProbLoArray"):
            prob_lo = self.geom.ProbLoArray()
        elif hasattr(self.geom, "ProbLo"):
            prob_lo = np.array(self.geom.ProbLo())
        else:
            raise AttributeError("Geometry has no ProbLoArray/ProbLo accessor")

        for mfi in mf:
            arr = mf.array(mfi).to_numpy()
            bx = mfi.validbox()
            i_lo, j_lo = bx.lo_vect
            i_hi, j_hi = bx.hi_vect

            i = np.arange(i_lo, i_hi + 1)
            j = np.arange(j_lo, j_hi + 1)

            x = (i + 0.5) * dx[0] + prob_lo[0]
            y = (j + 0.5) * dx[1] + prob_lo[1]

            Y, X = np.meshgrid(y, x, indexing="ij")
            p = sdf.vec2(X, Y)

            _get_component_view(arr)[...] = sdf_func(p)
