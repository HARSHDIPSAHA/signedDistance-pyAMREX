"""3D volume → SDF example.

Creates a synthetic 64³ volume with a spherical inclusion, segments it
with 3D Chan-Vese, computes morphometrics, and generates an SDF field
compatible with pySdf's CSG tree.

Usage
-----
    python volume_3d_example.py

Output
------
    examples/img2sdf/output/sphere_sdf.npy   — saved level-set array
"""
from __future__ import annotations
import os
import numpy as np

# Ensure the package root is importable when run directly
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from img2sdf import (
    volume_to_levelset_3d,
    volume_to_geometry_3d,
    compute_morphometry_3d,
)

# ---------------------------------------------------------------------------
# 1. Generate a synthetic 64³ volume with a spherical inclusion
# ---------------------------------------------------------------------------

SIZE = 64
RADIUS = 20

print(f"Generating synthetic {SIZE}³ volume (sphere r={RADIUS}) …")
d = np.arange(SIZE)[:, None, None]
h = np.arange(SIZE)[None, :, None]
w = np.arange(SIZE)[None, None, :]
center = SIZE / 2.0
dist = np.sqrt((d - center) ** 2 + (h - center) ** 2 + (w - center) ** 2)
volume = np.where(dist <= RADIUS, 200.0, 20.0)  # bright sphere on dark bg

# ---------------------------------------------------------------------------
# 2. Segment with 3D Chan-Vese and get the level-set (pySdf convention)
# ---------------------------------------------------------------------------

params = {
    "Segmentation": {
        "max_iter": 100,
        "mu": 0.25,
        "sigma": 3.0,
        "dt": 0.5,
    }
}

print("Running 3D Chan-Vese segmentation …")
phi = volume_to_levelset_3d(volume, params)
print(f"  phi shape: {phi.shape},  min: {phi.min():.3f},  max: {phi.max():.3f}")

# ---------------------------------------------------------------------------
# 3. Compute 3D morphometric features
# ---------------------------------------------------------------------------

print("Computing 3D morphometrics …")
morph = compute_morphometry_3d(phi, voxel_size=1.0)
V_exact = (4.0 / 3.0) * np.pi * RADIUS ** 3
A_exact = 4.0 * np.pi * RADIUS ** 2
print(f"  Volume:       {morph['volume']:.1f}  (exact: {V_exact:.1f})")
print(f"  Surface area: {morph['surface_area']:.1f}  (exact: {A_exact:.1f})")
print(f"  Sphericity:   {morph['sphericity']:.4f}  (exact: 1.0)")

# ---------------------------------------------------------------------------
# 4. Build an ImageGeometry3D and compose with an analytic Sphere3D
# ---------------------------------------------------------------------------

from sdf3d import Sphere3D

bounds = ((0.0, float(SIZE)), (0.0, float(SIZE)), (0.0, float(SIZE)))
geom = volume_to_geometry_3d(volume, params, bounds=bounds)
print(f"  Geometry: {geom!r}")

# CSG: hollow out the image geometry by subtracting a smaller analytic sphere
inner_sphere = Sphere3D(12.0).translate(center, center, center)
hollow = geom.subtract(inner_sphere)
print("  CSG subtraction successful:", hollow)

# ---------------------------------------------------------------------------
# 5. Save the SDF
# ---------------------------------------------------------------------------

out_dir = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "sphere_sdf.npy")
np.save(out_path, phi)
print(f"\nSaved SDF to: {out_path}")
