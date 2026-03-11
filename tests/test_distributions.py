"""Tests for sdf3d.distributions and sdf2d.distributions."""

from __future__ import annotations

import numpy as np
import numpy.testing as npt
import pytest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sdf3d.distributions import (
    generate_centers_random,
    generate_centers_inline,
    generate_centers_staggered,
    distribute_shape,
)
from sdf2d.distributions import (
    generate_centers_random as generate_centers_random_2d,
    generate_centers_inline as generate_centers_inline_2d,
    generate_centers_staggered as generate_centers_staggered_2d,
    distribute_shape as distribute_shape_2d,
)
from sdf3d import Sphere3D
from sdf2d import Circle2D

_BOUNDS3D = ((-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0))
_BOUNDS2D = ((-1.0, 1.0), (-1.0, 1.0))


# ===========================================================================
# 3D distributions
# ===========================================================================

class TestGenerateCentersRandom3D:
    def test_output_shape(self):
        c = generate_centers_random(_BOUNDS3D, 5, 0.3, seed=42)
        assert c.shape == (5, 3)

    def test_within_bounds(self):
        c = generate_centers_random(_BOUNDS3D, 10, 0.15, seed=0)
        for i, (lo, hi) in enumerate(((-1.0, 1.0),) * 3):
            assert c[:, i].min() >= lo
            assert c[:, i].max() <= hi

    def test_min_separation_respected(self):
        min_sep = 0.4
        c = generate_centers_random(_BOUNDS3D, 5, min_sep, seed=7)
        for i in range(len(c)):
            for j in range(i + 1, len(c)):
                dist = np.linalg.norm(c[i] - c[j])
                assert dist >= min_sep - 1e-9, f"Separation {dist} < {min_sep}"

    def test_reproducible_with_seed(self):
        c1 = generate_centers_random(_BOUNDS3D, 5, 0.3, seed=99)
        c2 = generate_centers_random(_BOUNDS3D, 5, 0.3, seed=99)
        npt.assert_array_equal(c1, c2)

    def test_different_seeds_differ(self):
        c1 = generate_centers_random(_BOUNDS3D, 5, 0.3, seed=1)
        c2 = generate_centers_random(_BOUNDS3D, 5, 0.3, seed=2)
        assert not np.array_equal(c1, c2)

    def test_impossible_placement_raises(self):
        with pytest.raises(RuntimeError, match="Could not place center"):
            generate_centers_random(
                _BOUNDS3D, 100, 1.5, seed=0, max_attempts=50
            )


class TestGenerateCentersInline3D:
    def test_shape_scalar(self):
        c = generate_centers_inline(_BOUNDS3D, 3)
        assert c.shape == (27, 3)

    def test_shape_tuple(self):
        c = generate_centers_inline(_BOUNDS3D, (2, 3, 4))
        assert c.shape == (24, 3)

    def test_within_bounds(self):
        c = generate_centers_inline(_BOUNDS3D, 4)
        for i in range(3):
            assert c[:, i].min() >= -1.0
            assert c[:, i].max() <= 1.0

    def test_uniform_spacing(self):
        c = generate_centers_inline(_BOUNDS3D, 3)
        xs = np.sort(np.unique(np.round(c[:, 0], 10)))
        expected_xs = np.array([-1.0 + 2.0 / 3.0 * (i + 0.5) for i in range(3)])
        npt.assert_allclose(xs, expected_xs, atol=1e-10)


class TestGenerateCentersStaggered3D:
    def test_shape(self):
        c = generate_centers_staggered(_BOUNDS3D, 3)
        assert c.shape == (27, 3)

    def test_within_bounds(self):
        c = generate_centers_staggered(_BOUNDS3D, 4)
        for i in range(3):
            assert c[:, i].min() >= -1.0
            # Staggered centers can exceed bounds slightly by offset_fraction *
            # cell_width, but with default offset 0.5 and small cells the max
            # still stays < x1 + cell_width.  Just check they are reasonable.
            assert c[:, i].max() <= 1.5  # generous upper bound

    def test_odd_layer_offset(self):
        # For a 2x2x2 stagger with offset_fraction=0.5, even layers (k=0)
        # should have the same x-coords as inline, odd layers (k=1) should be
        # shifted by 0.5*dx.
        (x0, x1), _, _ = _BOUNDS3D
        c = generate_centers_staggered(_BOUNDS3D, 2)
        # k=0 layer: z in (-1 + dz/2,) → first 4 centers
        # k=1 layer: second 4 centers
        xs_k0 = np.sort(np.unique(np.round(c[:4, 0], 10)))
        xs_k1 = np.sort(np.unique(np.round(c[4:, 0], 10)))
        dx = (x1 - x0) / 2
        offset = 0.5 * dx
        npt.assert_allclose(xs_k1, xs_k0 + offset, atol=1e-10)


class TestDistributeShape3D:
    def test_returns_sdf3d(self):
        from sdf3d import SDF3D
        c = generate_centers_inline(_BOUNDS3D, 2)
        result = distribute_shape(lambda: Sphere3D(0.1), c)
        assert isinstance(result, SDF3D)

    def test_centers_are_inside(self):
        c = generate_centers_inline(_BOUNDS3D, 2)
        result = distribute_shape(lambda: Sphere3D(0.15), c)
        # All centers should be inside (phi < 0)
        p = c  # shape (8, 3)
        phi = result.sdf(p)
        assert (phi < 0).all(), "Not all centers are inside the distributed shapes"

    def test_empty_centers_raises(self):
        with pytest.raises(ValueError, match="empty"):
            distribute_shape(lambda: Sphere3D(0.1), np.zeros((0, 3)))

    def test_wrong_shape_raises(self):
        with pytest.raises(ValueError):
            distribute_shape(lambda: Sphere3D(0.1), np.zeros((5, 2)))


# ===========================================================================
# 2D distributions
# ===========================================================================

class TestGenerateCentersRandom2D:
    def test_output_shape(self):
        c = generate_centers_random_2d(_BOUNDS2D, 5, 0.3, seed=42)
        assert c.shape == (5, 2)

    def test_within_bounds(self):
        c = generate_centers_random_2d(_BOUNDS2D, 10, 0.15, seed=0)
        for i in range(2):
            assert c[:, i].min() >= -1.0
            assert c[:, i].max() <= 1.0

    def test_min_separation(self):
        min_sep = 0.5
        c = generate_centers_random_2d(_BOUNDS2D, 4, min_sep, seed=3)
        for i in range(len(c)):
            for j in range(i + 1, len(c)):
                dist = np.linalg.norm(c[i] - c[j])
                assert dist >= min_sep - 1e-9

    def test_impossible_raises(self):
        with pytest.raises(RuntimeError):
            generate_centers_random_2d(
                _BOUNDS2D, 50, 1.0, seed=0, max_attempts=20
            )


class TestGenerateCentersInline2D:
    def test_shape_scalar(self):
        c = generate_centers_inline_2d(_BOUNDS2D, 4)
        assert c.shape == (16, 2)

    def test_shape_tuple(self):
        c = generate_centers_inline_2d(_BOUNDS2D, (3, 5))
        assert c.shape == (15, 2)


class TestGenerateCentersStaggered2D:
    def test_shape(self):
        c = generate_centers_staggered_2d(_BOUNDS2D, 3)
        assert c.shape == (9, 2)

    def test_odd_row_offset(self):
        (x0, x1), _ = _BOUNDS2D
        c = generate_centers_staggered_2d(_BOUNDS2D, 2)
        xs_row0 = np.sort(c[:2, 0])   # first row (j=0)
        xs_row1 = np.sort(c[2:, 0])   # second row (j=1)
        dx = (x1 - x0) / 2
        npt.assert_allclose(xs_row1, xs_row0 + 0.5 * dx, atol=1e-10)


class TestDistributeShape2D:
    def test_returns_sdf2d(self):
        from sdf2d import SDF2D
        c = generate_centers_inline_2d(_BOUNDS2D, 2)
        result = distribute_shape_2d(lambda: Circle2D(0.1), c)
        assert isinstance(result, SDF2D)

    def test_centers_are_inside(self):
        c = generate_centers_inline_2d(_BOUNDS2D, 2)
        result = distribute_shape_2d(lambda: Circle2D(0.15), c)
        phi = result.sdf(c)
        assert (phi < 0).all()

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            distribute_shape_2d(lambda: Circle2D(0.1), np.zeros((0, 2)))

    def test_wrong_shape_raises(self):
        with pytest.raises(ValueError):
            distribute_shape_2d(lambda: Circle2D(0.1), np.zeros((5, 3)))
