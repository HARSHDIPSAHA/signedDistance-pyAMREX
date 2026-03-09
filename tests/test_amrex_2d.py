"""AMReX 2-D integration tests for MultiFabGrid2D.

Requires pyAMReX (``amrex.space2d``) installed in the active Python environment.
Without it the entire module is skipped automatically.

Run in isolation (avoids the space2d/space3d pybind11 conflict)::

    python -m pytest tests/test_amrex_2d.py -v
"""
import sys

import numpy as np
import numpy.testing as npt
import pytest

if "amrex.space3d" in sys.modules:
    pytest.skip(
        "amrex.space3d already imported in this process; "
        "run tests/test_amrex_2d.py in isolation.",
        allow_module_level=True,
    )

amr2d = pytest.importorskip("amrex.space2d", exc_type=ImportError,
                             reason="pyAMReX 2D not installed")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_grid(n: int = 32, lo=(-1.0, -1.0), hi=(1.0, 1.0)):
    real_box = amr2d.RealBox(list(lo), list(hi))
    domain   = amr2d.Box(np.array([0, 0]), np.array([n - 1, n - 1]))
    geom     = amr2d.Geometry(domain, real_box, 0, [0, 0])
    ba       = amr2d.BoxArray(domain); ba.max_size(n // 2)
    dm       = amr2d.DistributionMapping(ba)
    return geom, ba, dm


def _collect(mf, n: int) -> np.ndarray:
    out = np.zeros((n, n))
    for mfi in mf:
        arr = mf.array(mfi).to_numpy()
        bx  = mfi.validbox()
        i0, j0 = bx.lo_vect; i1, j1 = bx.hi_vect
        out[j0:j1+1, i0:i1+1] = arr[:, :, 0, 0] if arr.ndim == 4 else arr[:, :, 0]
    return out


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestMultiFabGrid2D:
    @pytest.fixture(autouse=True)
    def init_amrex(self):
        amr2d.initialize([])
        yield
        amr2d.finalize()

    def test_fill_returns_multifab(self):
        from sdf2d import Circle2D, MultiFabGrid2D
        geom, ba, dm = _make_grid(n=16)
        grid = MultiFabGrid2D(geom, ba, dm)
        mf = Circle2D(0.5).to_multifab(grid)
        assert hasattr(mf, "array")

    def test_to_multifab_convenience(self):
        from sdf2d import Circle2D, MultiFabGrid2D
        geom, ba, dm = _make_grid(n=32)
        grid = MultiFabGrid2D(geom, ba, dm)
        mf = Circle2D(0.3).to_multifab(grid)
        phi = _collect(mf, 32)
        assert phi[16, 16] < 0   # origin is inside → negative

    def test_union_contains_both(self):
        from sdf2d import Circle2D, MultiFabGrid2D
        geom, ba, dm = _make_grid(n=32)
        grid = MultiFabGrid2D(geom, ba, dm)
        mf_a = Circle2D(0.25).translate(-0.4, 0.0).to_multifab(grid)
        mf_b = Circle2D(0.25).translate( 0.4, 0.0).to_multifab(grid)
        phi = _collect(grid.union(mf_a, mf_b), 32)
        assert phi[16,  8] < 0   # left circle centre
        assert phi[16, 24] < 0   # right circle centre

    def test_subtract_removes_cutter(self):
        from sdf2d import Circle2D, MultiFabGrid2D
        geom, ba, dm = _make_grid(n=32)
        grid   = MultiFabGrid2D(geom, ba, dm)
        base   = Circle2D(0.5).to_multifab(grid)
        cutter = Circle2D(0.2).to_multifab(grid)
        phi = _collect(grid.subtract(base, cutter), 32)
        assert phi[16, 16] > 0   # origin is in the hole → outside
        assert phi[16, 21] < 0   # ~(0.25, 0): inside base, outside cutter

    def test_intersect_requires_both(self):
        from sdf2d import Circle2D, MultiFabGrid2D
        geom, ba, dm = _make_grid(n=32)
        grid = MultiFabGrid2D(geom, ba, dm)
        mf_a = Circle2D(0.3).translate(-0.1, 0.0).to_multifab(grid)
        mf_b = Circle2D(0.3).translate( 0.1, 0.0).to_multifab(grid)
        phi = _collect(grid.intersect(mf_a, mf_b), 32)
        assert phi[16, 16] < 0   # origin is in the overlap → inside
        assert phi[16,  4] > 0   # far left (only inside a) → outside

    def test_negate_flips_sign(self):
        from sdf2d import Circle2D, MultiFabGrid2D
        geom, ba, dm = _make_grid(n=32)
        grid = MultiFabGrid2D(geom, ba, dm)
        mf   = Circle2D(0.3).to_multifab(grid)
        phi_orig = _collect(mf, 32)
        phi_neg  = _collect(grid.negate(mf), 32)
        npt.assert_allclose(phi_neg, -phi_orig)

    def test_sdf_operators_compose_before_fill(self):
        """SDF | - / operators compose in SDF space; fill once."""
        from sdf2d import Circle2D, Box2D, MultiFabGrid2D
        geom, ba, dm = _make_grid(n=32)
        grid = MultiFabGrid2D(geom, ba, dm)

        # Build composite SDF using operators, then fill once
        shape = Circle2D(0.25).translate(-0.3, 0.0) | Circle2D(0.25).translate(0.3, 0.0)
        phi = _collect(shape.to_multifab(grid), 32)
        assert phi[16,  8] < 0   # left circle centre
        assert phi[16, 24] < 0   # right circle centre
