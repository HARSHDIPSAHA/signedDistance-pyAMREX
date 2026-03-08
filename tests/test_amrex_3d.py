"""AMReX 3-D integration tests for MultiFabGrid3D.

Requires pyAMReX (``amrex.space3d``) installed in the active Python environment.
Without it the entire module is skipped automatically.

IMPORTANT — run in isolation to avoid the space2d/space3d pybind11 conflict::

    python -m pytest tests/test_amrex_3d.py -v
"""
import sys

import numpy as np
import pytest

if "amrex.space2d" in sys.modules:
    pytest.skip(
        "amrex.space2d already imported in this process; "
        "run tests/test_amrex_3d.py in isolation.",
        allow_module_level=True,
    )

amr3d = pytest.importorskip("amrex.space3d", exc_type=ImportError,
                             reason="pyAMReX 3D not installed")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_grid(n: int = 16, lo=(-1.0, -1.0, -1.0), hi=(1.0, 1.0, 1.0)):
    real_box = amr3d.RealBox(list(lo), list(hi))
    domain   = amr3d.Box(np.array([0, 0, 0]), np.array([n - 1, n - 1, n - 1]))
    geom     = amr3d.Geometry(domain, real_box, 0, [0, 0, 0])
    ba       = amr3d.BoxArray(domain); ba.max_size(n // 2)
    dm       = amr3d.DistributionMapping(ba)
    return geom, ba, dm


def _collect(mf, n: int) -> np.ndarray:
    out = np.zeros((n, n, n))
    for mfi in mf:
        arr = mf.array(mfi).to_numpy()
        bx  = mfi.validbox()
        i0, j0, k0 = bx.lo_vect; i1, j1, k1 = bx.hi_vect
        vals = arr[:, :, :, 0] if arr.ndim == 4 else arr[:, :, :, 0, 0]
        out[k0:k1+1, j0:j1+1, i0:i1+1] = vals
    return out


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestMultiFabGrid3D:
    @pytest.fixture(autouse=True)
    def init_amrex(self):
        amr3d.initialize([])
        yield
        amr3d.finalize()

    def test_fill_returns_multifab(self):
        from sdf3d import Sphere3D, MultiFabGrid3D
        geom, ba, dm = _make_grid(n=16)
        grid = MultiFabGrid3D(geom, ba, dm)
        mf = Sphere3D(0.4).fill(grid)
        assert hasattr(mf, "array")

    def test_to_multifab_convenience(self):
        from sdf3d import Sphere3D
        geom, ba, dm = _make_grid(n=16)
        mf = Sphere3D(0.3).to_multifab(geom, ba, dm)
        phi = _collect(mf, 16)
        assert phi[8, 8, 8] < 0   # origin is inside → negative

    def test_union_contains_both(self):
        from sdf3d import Sphere3D, MultiFabGrid3D
        geom, ba, dm = _make_grid(n=16)
        grid = MultiFabGrid3D(geom, ba, dm)
        mf_a = Sphere3D(0.2).translate(-0.4, 0.0, 0.0).fill(grid)
        mf_b = Sphere3D(0.2).translate( 0.4, 0.0, 0.0).fill(grid)
        phi = _collect(grid.union(mf_a, mf_b), 16)
        assert phi[8, 8,  4] < 0   # left sphere centre
        assert phi[8, 8, 12] < 0   # right sphere centre

    def test_subtract_removes_cutter(self):
        from sdf3d import Sphere3D, MultiFabGrid3D
        geom, ba, dm = _make_grid(n=16)
        grid   = MultiFabGrid3D(geom, ba, dm)
        base   = Sphere3D(0.5).fill(grid)
        cutter = Sphere3D(0.2).fill(grid)
        phi = _collect(grid.subtract(base, cutter), 16)
        assert phi[8, 8, 8] > 0   # origin is in the hole → outside

    def test_intersect_requires_both(self):
        from sdf3d import Sphere3D, MultiFabGrid3D
        geom, ba, dm = _make_grid(n=16)
        grid = MultiFabGrid3D(geom, ba, dm)
        mf_a = Sphere3D(0.3).translate(-0.1, 0.0, 0.0).fill(grid)
        mf_b = Sphere3D(0.3).translate( 0.1, 0.0, 0.0).fill(grid)
        phi = _collect(grid.intersect(mf_a, mf_b), 16)
        assert phi[8, 8, 8] < 0   # overlap region → inside
        assert phi[8, 8, 2] > 0   # far left (only inside a) → outside

    def test_sdf_operators_compose_before_fill(self):
        """SDF | - / operators compose in SDF space; fill once."""
        from sdf3d import Sphere3D, MultiFabGrid3D
        geom, ba, dm = _make_grid(n=16)
        grid = MultiFabGrid3D(geom, ba, dm)

        shape = Sphere3D(0.2).translate(-0.4, 0.0, 0.0) | Sphere3D(0.2).translate(0.4, 0.0, 0.0)
        phi = _collect(shape.fill(grid), 16)
        assert phi[8, 8,  4] < 0   # left sphere centre
        assert phi[8, 8, 12] < 0   # right sphere centre
