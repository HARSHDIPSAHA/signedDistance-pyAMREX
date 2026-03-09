"""AMReX MultiFab integration for 2-D SDF fields.

This module requires ``amrex.space2d`` (pyAMReX built in 2-D mode).
It is intentionally kept separate so that the rest of ``sdf2d`` works
without AMReX installed.
"""

from typing import TYPE_CHECKING

import numpy as np

from . import primitives as sdf

if TYPE_CHECKING:
    import amrex.space2d as amr  # noqa: F401 — type-checker only


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _get_component_view(arr: np.ndarray) -> np.ndarray:
    """Return the scalar-component view of an AMReX numpy array."""
    if arr.ndim == 3:
        return arr[:, :, 0]
    return arr[:, :, 0, 0]


def _copy_mf_like(
    src: "amr.MultiFab",
    ba: "amr.BoxArray",
    dm: "amr.DistributionMapping",
) -> "amr.MultiFab":
    """Allocate a new MultiFab matching *src*'s layout and copy its values."""
    import amrex.space2d as amr

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

class MultiFabGrid2D:
    """Grid context for 2-D AMReX MultiFab operations.

    Create once per domain layout; reuse for every shape and boolean op::

        grid = MultiFabGrid2D(geom, ba, dm)

        mf_circle = Circle2D(0.3).to_multifab(grid)
        mf_box    = Box2D((0.2, 0.2)).to_multifab(grid)
        mf_union  = grid.union(mf_circle, mf_box)

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
