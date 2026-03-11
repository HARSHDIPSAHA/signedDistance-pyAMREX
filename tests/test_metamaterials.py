"""Tests for sdf3d.metamaterials TPMS and lattice classes."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import numpy.testing as npt
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sdf3d import SDF3D
from sdf3d.metamaterials import (
    Gyroid3D,
    SchwarzP3D,
    SchwarzD3D,
    Neovius3D,
    BCCLattice3D,
    FCCLattice3D,
)

_BOUNDS = ((-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0))


def _grid(n: int = 8) -> np.ndarray:
    lin = np.linspace(-1.0, 1.0, n)
    Z, Y, X = np.meshgrid(lin, lin, lin, indexing="ij")
    return np.stack([X, Y, Z], axis=-1)


# ===========================================================================
# TPMS classes
# ===========================================================================

class TestGyroid3D:
    def test_is_sdf3d(self):
        assert isinstance(Gyroid3D(), SDF3D)

    def test_output_shape(self):
        phi = Gyroid3D().to_numpy(_BOUNDS, (8, 8, 8))
        assert phi.shape == (8, 8, 8)

    def test_has_zero_crossings(self):
        # With thickness=0.3, the gyroid should have both interior/exterior
        phi = Gyroid3D(cell_size=2.0, thickness=0.3).to_numpy(_BOUNDS, (20, 20, 20))
        assert phi.min() < 0, "Gyroid should have interior (phi < 0)"
        assert phi.max() > 0, "Gyroid should have exterior (phi > 0)"

    def test_sdf_callable(self):
        g = Gyroid3D()
        p = _grid(4)
        phi = g.sdf(p)
        assert phi.shape == (4, 4, 4)

    def test_supports_translation(self):
        g = Gyroid3D(cell_size=2.0, thickness=0.3)
        g_translated = g.translate(0.1, 0.2, 0.3)
        assert isinstance(g_translated, SDF3D)


class TestSchwarzP3D:
    def test_is_sdf3d(self):
        assert isinstance(SchwarzP3D(), SDF3D)

    def test_output_shape(self):
        phi = SchwarzP3D().to_numpy(_BOUNDS, (8, 8, 8))
        assert phi.shape == (8, 8, 8)

    def test_has_zero_crossings(self):
        phi = SchwarzP3D(cell_size=2.0, thickness=0.3).to_numpy(
            _BOUNDS, (20, 20, 20)
        )
        assert phi.min() < 0
        assert phi.max() > 0

    def test_symmetric(self):
        # Schwarz-P is symmetric under x→-x, y→-y, z→-z
        phi_pos = SchwarzP3D(cell_size=2.0, thickness=0.2)
        p = np.array([[0.3, 0.0, 0.0]])
        p_neg = np.array([[-0.3, 0.0, 0.0]])
        npt.assert_allclose(phi_pos.sdf(p), phi_pos.sdf(p_neg), atol=1e-10)


class TestSchwarzD3D:
    def test_is_sdf3d(self):
        assert isinstance(SchwarzD3D(), SDF3D)

    def test_output_shape(self):
        phi = SchwarzD3D().to_numpy(_BOUNDS, (8, 8, 8))
        assert phi.shape == (8, 8, 8)

    def test_has_zero_crossings(self):
        phi = SchwarzD3D(cell_size=2.0, thickness=0.3).to_numpy(
            _BOUNDS, (20, 20, 20)
        )
        assert phi.min() < 0
        assert phi.max() > 0


class TestNeovius3D:
    def test_is_sdf3d(self):
        assert isinstance(Neovius3D(), SDF3D)

    def test_output_shape(self):
        phi = Neovius3D().to_numpy(_BOUNDS, (8, 8, 8))
        assert phi.shape == (8, 8, 8)

    def test_has_zero_crossings(self):
        phi = Neovius3D(cell_size=2.0, thickness=0.3).to_numpy(
            _BOUNDS, (20, 20, 20)
        )
        assert phi.min() < 0
        assert phi.max() > 0


# ===========================================================================
# Lattice classes
# ===========================================================================

class TestBCCLattice3D:
    def test_is_sdf3d(self):
        assert isinstance(BCCLattice3D(), SDF3D)

    def test_output_shape(self):
        phi = BCCLattice3D(cell_size=0.5, beam_radius=0.05,
                           repeat=(2, 2, 2)).to_numpy(_BOUNDS, (8, 8, 8))
        assert phi.shape == (8, 8, 8)

    def test_has_solid(self):
        phi = BCCLattice3D(cell_size=0.5, beam_radius=0.1,
                           repeat=(2, 2, 2)).to_numpy(_BOUNDS, (16, 16, 16))
        assert phi.min() < 0, "BCC lattice should have solid regions"

    def test_center_point_inside(self):
        # The BCC body-centre at (0.25, 0.25, 0.25) inside a cell_size=0.5 cell
        # should be inside a beam (phi < 0) with a large enough beam_radius
        bcc = BCCLattice3D(cell_size=0.5, beam_radius=0.12, repeat=(2, 2, 2))
        p = np.array([[0.25, 0.25, 0.25]])  # body centre of first cell
        phi = bcc.sdf(p)
        assert phi[0] < 0, f"Body centre should be inside beam, got phi={phi[0]}"


class TestFCCLattice3D:
    def test_is_sdf3d(self):
        assert isinstance(FCCLattice3D(), SDF3D)

    def test_output_shape(self):
        phi = FCCLattice3D(cell_size=0.5, beam_radius=0.05,
                           repeat=(2, 2, 2)).to_numpy(_BOUNDS, (8, 8, 8))
        assert phi.shape == (8, 8, 8)

    def test_has_solid(self):
        phi = FCCLattice3D(cell_size=0.5, beam_radius=0.1,
                           repeat=(2, 2, 2)).to_numpy(_BOUNDS, (16, 16, 16))
        assert phi.min() < 0, "FCC lattice should have solid regions"


# ===========================================================================
# Boolean operations work on metamaterials
# ===========================================================================

class TestMetamaterialBooleans:
    def test_union_with_sphere(self):
        from sdf3d import Sphere3D
        g = Gyroid3D(cell_size=2.0, thickness=0.3)
        s = Sphere3D(0.5)
        u = g | s
        assert isinstance(u, SDF3D)
        phi = u.to_numpy(_BOUNDS, (8, 8, 8))
        assert phi.shape == (8, 8, 8)

    def test_intersect_clips_gyroid(self):
        from sdf3d import Sphere3D
        g = Gyroid3D(cell_size=2.0, thickness=0.3)
        s = Sphere3D(0.7)
        clipped = g / s
        phi = clipped.to_numpy(_BOUNDS, (16, 16, 16))
        # Outside sphere (r > 0.7) must be exterior (phi > 0)
        p = np.array([[0.9, 0.0, 0.0]])
        assert clipped.sdf(p)[0] > 0
