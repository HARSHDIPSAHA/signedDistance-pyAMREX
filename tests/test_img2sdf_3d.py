"""Tests for img2sdf 3D volume support.

Tests do NOT require cv2, AMReX — they test using synthetic binary volumes
and numpy arrays directly.  scikit-image is required only for the morphometry
tests and is skipped gracefully when absent.
"""
from __future__ import annotations
import sys
import os
import numpy as np
import numpy.testing as npt
import pytest

# Make sure the package root is on sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ---------------------------------------------------------------------------
# Shared synthetic data helpers
# ---------------------------------------------------------------------------

def _sphere_volume(shape=(32, 32, 32), radius=8, bg=20, fg=200):
    """Create a 3D numpy array with a spherical inclusion."""
    D, H, W = shape
    cd, ch, cw = D // 2, H // 2, W // 2
    d = np.arange(D)[:, None, None]
    h = np.arange(H)[None, :, None]
    w = np.arange(W)[None, None, :]
    dist = np.sqrt((d - cd) ** 2 + (h - ch) ** 2 + (w - cw) ** 2)
    vol = np.where(dist <= radius, fg, bg).astype(np.float64)
    return vol


def _sphere_phi(shape=(32, 32, 32), radius=8.0):
    """Create a pySdf-convention level-set for a sphere (phi < 0 inside)."""
    D, H, W = shape
    cd, ch, cw = D / 2.0, H / 2.0, W / 2.0
    d = np.arange(D)[:, None, None]
    h = np.arange(H)[None, :, None]
    w = np.arange(W)[None, None, :]
    dist = np.sqrt((d - cd) ** 2 + (h - ch) ** 2 + (w - cw) ** 2)
    return dist - radius  # negative inside, positive outside


# ===========================================================================
# 3D Chan-Vese
# ===========================================================================

class TestChanVese3D:
    """Tests for img2sdf.segmentation.cv_single_3d.chan_vese_3d."""

    def test_output_shape(self):
        pytest.importorskip("scipy")
        from img2sdf.segmentation.cv_single_3d import chan_vese_3d

        vol = _sphere_volume(shape=(16, 16, 16), radius=5)
        seg, phi_list = chan_vese_3d(vol, max_iter=5)

        assert len(phi_list) == 1
        assert phi_list[0].shape == vol.shape

    def test_output_has_both_signs(self):
        pytest.importorskip("scipy")
        from img2sdf.segmentation.cv_single_3d import chan_vese_3d

        vol = _sphere_volume(shape=(16, 16, 16), radius=5)
        _seg, phi_list = chan_vese_3d(vol, max_iter=10)
        phi = phi_list[0]

        assert phi.max() > 0, "phi should have positive values (outside)"
        assert phi.min() < 0, "phi should have negative values (inside)"

    def test_center_positive_uscman_convention(self):
        """Center of sphere should be positive (uSCMAN: phi > 0 inside)."""
        pytest.importorskip("scipy")
        from img2sdf.segmentation.cv_single_3d import chan_vese_3d

        vol = _sphere_volume(shape=(24, 24, 24), radius=8)
        _seg, phi_list = chan_vese_3d(vol, max_iter=30, sigma=2.0)
        phi = phi_list[0]

        cx, cy, cz = 12, 12, 12
        assert phi[cz, cy, cx] > 0, (
            "Center of sphere should be phi > 0 (uSCMAN inside convention)"
        )

    def test_segmentation_regions(self):
        pytest.importorskip("scipy")
        from img2sdf.segmentation.cv_single_3d import chan_vese_3d

        vol = _sphere_volume(shape=(16, 16, 16), radius=5)
        seg, _phi = chan_vese_3d(vol, max_iter=5)

        assert len(seg) == 2
        R1, R2 = seg
        assert R1.shape == vol.shape
        assert R2.shape == vol.shape
        # R1 + R2 should cover the whole volume
        npt.assert_array_equal(R1 + R2, np.ones_like(R1))

    def test_raises_on_2d_input(self):
        pytest.importorskip("scipy")
        from img2sdf.segmentation.cv_single_3d import chan_vese_3d

        with pytest.raises(ValueError, match="3D"):
            chan_vese_3d(np.zeros((8, 8)), max_iter=1)


# ===========================================================================
# ImageGeometry3D
# ===========================================================================

class TestImageGeometry3D:
    """Tests for img2sdf.geometry3d.ImageGeometry3D."""

    def test_import(self):
        pytest.importorskip("scipy")
        from img2sdf import ImageGeometry3D
        assert ImageGeometry3D is not None

    def test_sdf_center_negative(self):
        """phi < 0 at centre of sphere (pySdf convention)."""
        pytest.importorskip("scipy")
        from img2sdf import ImageGeometry3D

        phi = _sphere_phi(shape=(32, 32, 32), radius=8.0)
        bounds = ((0.0, 32.0), (0.0, 32.0), (0.0, 32.0))
        geom = ImageGeometry3D(phi, bounds)

        p_center = np.array([[16.0, 16.0, 16.0]])
        val = geom.sdf(p_center)
        assert val[0] < 0, f"Expected negative at centre, got {val[0]}"

    def test_sdf_outside_positive(self):
        """phi > 0 far outside sphere."""
        pytest.importorskip("scipy")
        from img2sdf import ImageGeometry3D

        phi = _sphere_phi(shape=(32, 32, 32), radius=8.0)
        bounds = ((0.0, 32.0), (0.0, 32.0), (0.0, 32.0))
        geom = ImageGeometry3D(phi, bounds)

        p_outside = np.array([[31.0, 31.0, 31.0]])
        val = geom.sdf(p_outside)
        assert val[0] > 0, f"Expected positive outside, got {val[0]}"

    def test_batch_query(self):
        """sdf() works on batches of points."""
        pytest.importorskip("scipy")
        from img2sdf import ImageGeometry3D

        phi = _sphere_phi(shape=(16, 16, 16), radius=5.0)
        bounds = ((0.0, 16.0), (0.0, 16.0), (0.0, 16.0))
        geom = ImageGeometry3D(phi, bounds)

        pts = np.random.default_rng(0).uniform(0, 16, (20, 3))
        vals = geom.sdf(pts)
        assert vals.shape == (20,)

    def test_csg_union_with_sphere3d(self):
        """ImageGeometry3D.union() works with a Sphere3D."""
        pytest.importorskip("scipy")
        from img2sdf import ImageGeometry3D
        from sdf3d import Sphere3D

        phi = _sphere_phi(shape=(32, 32, 32), radius=8.0)
        bounds = ((-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0))
        geom = ImageGeometry3D(phi, bounds)

        sphere = Sphere3D(0.3)
        combined = geom.union(sphere)

        p = np.array([[0.0, 0.0, 0.0]])
        val = combined.sdf(p)
        assert val.shape == (1,)

    def test_negate(self):
        pytest.importorskip("scipy")
        from img2sdf import ImageGeometry3D

        phi = _sphere_phi(shape=(16, 16, 16), radius=5.0)
        bounds = ((0.0, 16.0), (0.0, 16.0), (0.0, 16.0))
        geom = ImageGeometry3D(phi, bounds)
        negated = geom.negate()

        p_center = np.array([[8.0, 8.0, 8.0]])
        assert geom.sdf(p_center)[0] < 0
        assert negated.sdf(p_center)[0] > 0

    def test_repr(self):
        pytest.importorskip("scipy")
        from img2sdf import ImageGeometry3D

        phi = _sphere_phi(shape=(8, 8, 8), radius=3.0)
        bounds = ((0.0, 8.0), (0.0, 8.0), (0.0, 8.0))
        geom = ImageGeometry3D(phi, bounds)
        assert "ImageGeometry3D" in repr(geom)


# ===========================================================================
# volume_to_levelset_3d
# ===========================================================================

class TestVolumeToLevelset3D:
    """Tests for img2sdf.grid3d.volume_to_levelset_3d."""

    def test_output_shape(self):
        pytest.importorskip("scipy")
        from img2sdf import volume_to_levelset_3d

        vol = _sphere_volume(shape=(16, 16, 16), radius=5)
        params = {"Segmentation": {"max_iter": 5}}
        phi = volume_to_levelset_3d(vol, params)
        assert phi.shape == vol.shape

    def test_sign_convention(self):
        """phi < 0 at centre after negation (pySdf convention)."""
        pytest.importorskip("scipy")
        from img2sdf import volume_to_levelset_3d

        vol = _sphere_volume(shape=(24, 24, 24), radius=8)
        params = {"Segmentation": {"max_iter": 30, "sigma": 2.0}}
        phi = volume_to_levelset_3d(vol, params)

        cx, cy, cz = 12, 12, 12
        assert phi[cz, cy, cx] < 0, (
            "Centre of sphere should be phi < 0 (pySdf inside convention)"
        )

    def test_direct_array_input(self):
        """Pass a numpy array directly (no file I/O)."""
        pytest.importorskip("scipy")
        from img2sdf import volume_to_levelset_3d

        vol = np.random.default_rng(42).uniform(0, 1, (8, 8, 8))
        phi = volume_to_levelset_3d(vol, {"Segmentation": {"max_iter": 3}})
        assert phi.shape == (8, 8, 8)

    def test_npy_file_input(self, tmp_path):
        """Load from a .npy file."""
        pytest.importorskip("scipy")
        from img2sdf import volume_to_levelset_3d

        vol = _sphere_volume(shape=(8, 8, 8), radius=3)
        path = str(tmp_path / "vol.npy")
        np.save(path, vol)

        phi = volume_to_levelset_3d(path, {"Segmentation": {"max_iter": 3}})
        assert phi.shape == (8, 8, 8)


# ===========================================================================
# volume_to_geometry_3d
# ===========================================================================

class TestVolumeToGeometry3D:
    """Tests for img2sdf.grid3d.volume_to_geometry_3d."""

    def test_returns_image_geometry3d(self):
        pytest.importorskip("scipy")
        from img2sdf import volume_to_geometry_3d, ImageGeometry3D

        vol = _sphere_volume(shape=(16, 16, 16), radius=5)
        geom = volume_to_geometry_3d(vol, {"Segmentation": {"max_iter": 5}})
        assert isinstance(geom, ImageGeometry3D)

    def test_custom_bounds(self):
        pytest.importorskip("scipy")
        from img2sdf import volume_to_geometry_3d

        vol = _sphere_volume(shape=(16, 16, 16), radius=5)
        bounds = ((-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0))
        geom = volume_to_geometry_3d(
            vol, {"Segmentation": {"max_iter": 5}}, bounds=bounds
        )
        assert geom.bounds == bounds

    def test_works_with_sample_levelset_3d(self):
        """Integration: geometry is compatible with sdf3d.sample_levelset_3d."""
        pytest.importorskip("scipy")
        from img2sdf import volume_to_geometry_3d
        from sdf3d.grid import sample_levelset_3d

        vol = _sphere_volume(shape=(16, 16, 16), radius=5)
        geom = volume_to_geometry_3d(vol, {"Segmentation": {"max_iter": 5}})
        bounds = geom.bounds
        phi_out = sample_levelset_3d(geom, bounds, (8, 8, 8))
        assert phi_out.shape == (8, 8, 8)


# ===========================================================================
# compute_morphometry_3d
# ===========================================================================

class TestMorphometry3D:
    """Tests for img2sdf.morphometry.morphometry3d.compute_morphometry_3d."""

    def test_volume_sphere(self):
        """Volume of discretised sphere ≈ 4/3 π r³ within 5 % tolerance."""
        pytest.importorskip("scipy")
        pytest.importorskip("skimage")
        from img2sdf import compute_morphometry_3d

        r = 10.0
        phi = _sphere_phi(shape=(64, 64, 64), radius=r)
        result = compute_morphometry_3d(phi, voxel_size=1.0)

        V_exact = (4.0 / 3.0) * np.pi * r ** 3
        assert abs(result["volume"] - V_exact) / V_exact < 0.05, (
            f"Volume {result['volume']:.1f} deviates > 5% from {V_exact:.1f}"
        )

    def test_surface_area_sphere(self):
        """Surface area ≈ 4π r² within 10 % tolerance."""
        pytest.importorskip("scipy")
        pytest.importorskip("skimage")
        from img2sdf import compute_morphometry_3d

        r = 10.0
        phi = _sphere_phi(shape=(64, 64, 64), radius=r)
        result = compute_morphometry_3d(phi, voxel_size=1.0)

        A_exact = 4.0 * np.pi * r ** 2
        assert abs(result["surface_area"] - A_exact) / A_exact < 0.10, (
            f"Surface area {result['surface_area']:.1f} deviates > 10% "
            f"from {A_exact:.1f}"
        )

    def test_sphericity_sphere(self):
        """Sphericity of a sphere ≈ 1.0 within 0.1 tolerance."""
        pytest.importorskip("scipy")
        pytest.importorskip("skimage")
        from img2sdf import compute_morphometry_3d

        r = 10.0
        phi = _sphere_phi(shape=(64, 64, 64), radius=r)
        result = compute_morphometry_3d(phi, voxel_size=1.0)

        assert abs(result["sphericity"] - 1.0) < 0.1, (
            f"Sphericity {result['sphericity']:.3f} should be close to 1.0"
        )

    def test_sphericity_box_less_than_sphere(self):
        """Sphericity of an elongated box should be < 1.0."""
        pytest.importorskip("scipy")
        pytest.importorskip("skimage")
        from img2sdf import compute_morphometry_3d

        # Elongated box: box SDF via component-wise distance
        D, H, W = 64, 32, 16
        phi = np.ones((64, 64, 64))
        # Manually build a simple box SDF
        d = np.arange(64)[:, None, None]
        h = np.arange(64)[None, :, None]
        w = np.arange(64)[None, None, :]
        # Half-extents: D/2=32 along z, H/2=16 along y, W/2=8 along x
        qd = np.abs(d - 32) - 32
        qh = np.abs(h - 32) - 16
        qw = np.abs(w - 32) - 8
        # Box SDF
        phi = (
            np.sqrt(
                np.maximum(qd, 0) ** 2 +
                np.maximum(qh, 0) ** 2 +
                np.maximum(qw, 0) ** 2
            ) + np.minimum(np.maximum(qd, np.maximum(qh, qw)), 0)
        )
        result = compute_morphometry_3d(phi, voxel_size=1.0)
        assert result["sphericity"] < 1.0, (
            "Elongated box should have sphericity < 1.0"
        )

    def test_voxel_size_scaling(self):
        """Doubling voxel_size should scale volume by 8× and area by 4×."""
        pytest.importorskip("scipy")
        pytest.importorskip("skimage")
        from img2sdf import compute_morphometry_3d

        phi = _sphere_phi(shape=(32, 32, 32), radius=8.0)
        r1 = compute_morphometry_3d(phi, voxel_size=1.0)
        r2 = compute_morphometry_3d(phi, voxel_size=2.0)

        npt.assert_allclose(r2["volume"] / r1["volume"], 8.0, rtol=1e-9)
        npt.assert_allclose(r2["surface_area"] / r1["surface_area"], 4.0, rtol=1e-6)

    def test_empty_geometry(self):
        """All-positive phi (nothing inside) returns zero volume."""
        pytest.importorskip("scipy")
        pytest.importorskip("skimage")
        from img2sdf import compute_morphometry_3d

        phi = np.ones((16, 16, 16), dtype=np.float64)
        result = compute_morphometry_3d(phi)
        assert result["volume"] == 0.0
        assert result["surface_area"] == 0.0
        assert result["sphericity"] == 0.0

    def test_return_keys(self):
        pytest.importorskip("scipy")
        pytest.importorskip("skimage")
        from img2sdf import compute_morphometry_3d

        phi = _sphere_phi(shape=(16, 16, 16), radius=5.0)
        result = compute_morphometry_3d(phi)
        assert set(result.keys()) == {"volume", "surface_area", "sphericity"}


# ===========================================================================
# Integration
# ===========================================================================

class TestIntegration3D:
    """End-to-end: volume → segmentation → geometry → sample_levelset_3d."""

    def test_full_pipeline(self):
        pytest.importorskip("scipy")
        from img2sdf import volume_to_geometry_3d, ImageGeometry3D
        from sdf3d.grid import sample_levelset_3d

        vol = _sphere_volume(shape=(16, 16, 16), radius=5)
        geom = volume_to_geometry_3d(
            vol,
            {"Segmentation": {"max_iter": 10}},
        )
        assert isinstance(geom, ImageGeometry3D)

        bounds = ((0.0, 16.0), (0.0, 16.0), (0.0, 16.0))
        phi_grid = sample_levelset_3d(geom, bounds, (8, 8, 8))
        assert phi_grid.shape == (8, 8, 8)
        # The SDF should have both inside and outside regions
        assert phi_grid.min() < 0
        assert phi_grid.max() > 0
