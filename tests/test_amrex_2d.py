"""AMReX 2-D integration tests for SDFMultiFab2D.

Requires pyAMReX (``amrex.space2d``) installed in the active Python environment.
Without it the entire module is skipped automatically.

Run in isolation (avoids the space2d/space3d pybind11 conflict)::

    python -m pytest tests/test_amrex_2d.py -v

Or just run the full suite — 2-D tests run automatically, 3-D tests are skipped
when space2d was already imported first::

    uv run pytest tests/ -v          # uses uv's Python; skips if no pyAMReX
    python -m pytest tests/ -v       # uses system Python with pyAMReX installed
"""
import sys

import numpy as np
import numpy.testing as npt
import pytest

# amrex.space2d and amrex.space3d share a pybind11 type ("AMReX") and cannot
# coexist in the same process.  Skip cleanly if the 3D namespace is already loaded.
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

class TestSDFMultiFab2D:
    @pytest.fixture(autouse=True)
    def init_amrex(self):
        amr2d.initialize([])
        yield
        amr2d.finalize()

    def test_returns_multifab(self):
        from sdf2d import SDFMultiFab2D, Circle2D
        geom, ba, dm = _make_grid(n=16)
        lib = SDFMultiFab2D(geom, ba, dm)
        mf  = lib.from_geometry(Circle2D(0.5))
        assert hasattr(mf, "array")

    def test_union_contains_both(self):
        from sdf2d import SDFMultiFab2D, Circle2D
        geom, ba, dm = _make_grid(n=32)
        lib = SDFMultiFab2D(geom, ba, dm)
        a = lib.from_geometry(Circle2D(0.25).translate(-0.4, 0.0))
        b = lib.from_geometry(Circle2D(0.25).translate( 0.4, 0.0))
        u = lib.union(a, b)
        phi = _collect(u, 32)
        assert phi[16,  8] < 0   # left circle centre
        assert phi[16, 24] < 0   # right circle centre

    def test_subtract_removes_cutter(self):
        from sdf2d import SDFMultiFab2D, Circle2D
        geom, ba, dm = _make_grid(n=32)
        lib = SDFMultiFab2D(geom, ba, dm)
        cutter = lib.from_geometry(Circle2D(0.2))
        base   = lib.from_geometry(Circle2D(0.5))
        result = lib.subtract(cutter, base)
        phi = _collect(result, 32)
        assert phi[16, 16] > 0   # origin is in the hole → outside
        assert phi[16, 21] < 0   # ~(0.25, 0): inside base, outside cutter

    def test_intersect_requires_both(self):
        from sdf2d import SDFMultiFab2D, Circle2D
        geom, ba, dm = _make_grid(n=32)
        lib = SDFMultiFab2D(geom, ba, dm)
        a = lib.from_geometry(Circle2D(0.3).translate(-0.1, 0.0))
        b = lib.from_geometry(Circle2D(0.3).translate( 0.1, 0.0))
        inter = lib.intersect(a, b)
        phi = _collect(inter, 32)
        assert phi[16, 16] < 0   # origin is in the overlap → inside
        assert phi[16,  4] > 0   # far left (only inside a) → outside intersection

    def test_negate_flips_sign(self):
        from sdf2d import SDFMultiFab2D, Circle2D
        geom, ba, dm = _make_grid(n=32)
        lib = SDFMultiFab2D(geom, ba, dm)
        circ     = lib.from_geometry(Circle2D(0.3))
        neg      = lib.negate(circ)
        phi_orig = _collect(circ, 32)
        phi_neg  = _collect(neg,  32)
        npt.assert_allclose(phi_neg, -phi_orig)

    def test_from_geometry(self):
        from sdf2d import SDFMultiFab2D, Circle2D
        geom, ba, dm = _make_grid(n=32)
        lib = SDFMultiFab2D(geom, ba, dm)
        circle_geom = Circle2D(0.3)          # Circle2D takes radius only; centred at origin
        mf = lib.from_geometry(circle_geom)
        phi = _collect(mf, 32)
        assert phi[16, 16] < 0
