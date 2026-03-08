"""AMReX MultiFab integration for 3-D SDF fields.

This module requires ``amrex.space3d`` (pyAMReX built in 3-D mode).
It is intentionally kept separate so that the rest of ``sdf3d`` works
without AMReX installed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from . import primitives as sdf

if TYPE_CHECKING:
    import amrex.space3d as amr  # noqa: F401 — type-checker only

_Array = npt.NDArray[np.floating]


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _get_component_view(arr: _Array) -> _Array:
    """Return the scalar-component view of an AMReX numpy array."""
    if arr.ndim == 4:
        return arr[:, :, :, 0]
    return arr[:, :, :, 0, 0]


def _copy_mf_like(
    src: "amr.MultiFab",
    ba: "amr.BoxArray",
    dm: "amr.DistributionMapping",
) -> "amr.MultiFab":
    """Allocate a new MultiFab matching *src*'s layout and copy its values."""
    import amrex.space3d as amr

    ncomp: int = src.n_comp() if callable(getattr(src, "n_comp", None)) else src.n_comp
    if callable(getattr(src, "n_grow", None)):
        ngrow = src.n_grow()
    elif hasattr(src, "nGrow"):
        ngrow = src.nGrow()
    else:
        ngrow = 0

    out = amr.MultiFab(ba, dm, ncomp, ngrow)
    for mfi in src:
        arr_src = src.array(mfi).to_numpy()
        arr_dst = out.array(mfi).to_numpy()
        _get_component_view(arr_dst)[...] = _get_component_view(arr_src)
    return out


# ---------------------------------------------------------------------------
# Public class
# ---------------------------------------------------------------------------

class MultiFabGrid3D:
    """Grid context for 3-D AMReX MultiFab operations.

    Create once per domain layout; reuse for every shape and boolean op::

        grid = MultiFabGrid3D(geom, ba, dm)

        mf_sphere = Sphere3D(0.3).fill(grid)
        mf_box    = Box3D((0.2, 0.2, 0.2)).fill(grid)
        mf_union  = grid.union(mf_sphere, mf_box)

        amr.write_single_level_plotfile(pf_dir, mf_union, ...)

    Parameters
    ----------
    geom:
        ``amr.Geometry`` defining the physical domain.
    ba:
        ``amr.BoxArray`` describing the grid decomposition.
    dm:
        ``amr.DistributionMapping`` assigning boxes to MPI ranks.
    """

    def __init__(
        self,
        geom: "amr.Geometry",
        ba: "amr.BoxArray",
        dm: "amr.DistributionMapping",
    ) -> None:
        self.geom = geom
        self.ba   = ba
        self.dm   = dm

    # ------------------------------------------------------------------
    # MultiFab allocation and fill
    # ------------------------------------------------------------------

    def create_multifab(self) -> "amr.MultiFab":
        """Return an uninitialised single-component MultiFab with no ghost cells."""
        import amrex.space3d as amr
        return amr.MultiFab(self.ba, self.dm, 1, 0)

    def fill_multifab(self, mf: "amr.MultiFab", sdf_func) -> None:
        """Write SDF values from *sdf_func* into the pre-allocated MultiFab *mf*."""
        dx = self.geom.data().CellSize()
        if hasattr(self.geom, "ProbLoArray"):
            prob_lo = self.geom.ProbLoArray()
        elif hasattr(self.geom, "ProbLo"):
            prob_lo = np.array(self.geom.ProbLo())
        else:
            raise AttributeError("Geometry has no ProbLoArray/ProbLo accessor")

        for mfi in mf:
            arr = mf.array(mfi).to_numpy()
            bx  = mfi.validbox()

            i_lo, j_lo, k_lo = bx.lo_vect
            i_hi, j_hi, k_hi = bx.hi_vect

            i = np.arange(i_lo, i_hi + 1)
            j = np.arange(j_lo, j_hi + 1)
            k = np.arange(k_lo, k_hi + 1)

            x = (i + 0.5) * dx[0] + prob_lo[0]
            y = (j + 0.5) * dx[1] + prob_lo[1]
            z = (k + 0.5) * dx[2] + prob_lo[2]

            Z, Y, X = np.meshgrid(z, y, x, indexing="ij")
            p = sdf.vec3(X, Y, Z)

            _get_component_view(arr)[...] = sdf_func(p)

    # ------------------------------------------------------------------
    # Boolean operations on raw MultiFabs
    # ------------------------------------------------------------------

    def union(self, a: "amr.MultiFab", b: "amr.MultiFab") -> "amr.MultiFab":
        """Return ``min(a, b)`` element-wise (SDF union)."""
        out = _copy_mf_like(a, self.ba, self.dm)
        for mfi in out:
            v_out = _get_component_view(out.array(mfi).to_numpy())
            v_b   = _get_component_view(b.array(mfi).to_numpy())
            v_out[...] = sdf.opUnion(v_out, v_b)
        return out

    def subtract(self, base: "amr.MultiFab", cutter: "amr.MultiFab") -> "amr.MultiFab":
        """Return *base* with *cutter* carved out: ``max(-cutter, base)``."""
        out = _copy_mf_like(base, self.ba, self.dm)
        for mfi in out:
            v_base   = _get_component_view(out.array(mfi).to_numpy())
            v_cutter = _get_component_view(cutter.array(mfi).to_numpy())
            # opSubtraction(d1=cutter, d2=base) = max(-cutter, base)
            v_base[...] = sdf.opSubtraction(v_cutter, v_base)
        return out

    def intersect(self, a: "amr.MultiFab", b: "amr.MultiFab") -> "amr.MultiFab":
        """Return ``max(a, b)`` element-wise (SDF intersection)."""
        out = _copy_mf_like(a, self.ba, self.dm)
        for mfi in out:
            v_out = _get_component_view(out.array(mfi).to_numpy())
            v_b   = _get_component_view(b.array(mfi).to_numpy())
            v_out[...] = sdf.opIntersection(v_out, v_b)
        return out

    def negate(self, a: "amr.MultiFab") -> "amr.MultiFab":
        """Negate *a* element-wise (flip inside/outside)."""
        out = _copy_mf_like(a, self.ba, self.dm)
        for mfi in out:
            _get_component_view(out.array(mfi).to_numpy())[...] *= -1.0
        return out
