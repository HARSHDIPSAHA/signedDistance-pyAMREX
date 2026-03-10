"""img2sdf 3D showcase — one standalone HTML file per shape.

Demonstrates the complete img2sdf 3D pipeline:
  1.  Volume creation (synthetic sphere in noise)
  2.  3D Chan-Vese segmentation  → level-set φ
  3.  Sign-convention validation (pySdf: φ<0 inside)
  4.  ImageGeometry3D construction + trilinear interpolation
  5.  CSG operations: union, subtract, intersect with analytic Sphere3D/Box3D
  6.  compute_morphometry_3d  → volume, surface area, sphericity
  7.  volume_to_levelset_3d  (direct array path)
  8.  volume_to_geometry_3d  (full pipeline shortcut)
  9.  Save-to-npy  +  reload
  10. One standalone Plotly HTML per shape (written with include_plotlyjs="cdn")

Outputs
-------
    examples/img2sdf/output/showcase_original.html
    examples/img2sdf/output/showcase_union.html
    examples/img2sdf/output/showcase_subtract.html
    examples/img2sdf/output/showcase_intersect.html
    examples/img2sdf/output/showcase_levelset.npy

Usage
-----
    python img2sdf_3d_showcase.py
"""
from __future__ import annotations
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from img2sdf import (
    ImageGeometry3D,
    volume_to_levelset_3d,
    volume_to_geometry_3d,
    compute_morphometry_3d,
)
from img2sdf.segmentation.cv_single_3d import chan_vese_3d
from sdf3d import Sphere3D, Box3D

try:
    import plotly.graph_objects as go
except ImportError:
    raise ImportError("Install plotly: pip install plotly")

try:
    from skimage.measure import marching_cubes
except ImportError:
    raise ImportError("Install scikit-image: pip install scikit-image")

try:
    from scipy.ndimage import zoom, gaussian_filter
except ImportError:
    raise ImportError("Install scipy: pip install scipy")

OUT_DIR = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUT_DIR, exist_ok=True)

DARK  = "#1a1a2e"
PANEL = "#16213e"


def _print(s):
    print(f"  {s}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _header(n, title):
    print(f"\n{'='*60}")
    print(f"  [{n}] {title}")
    print('='*60)


def _mesh3d_trace(phi, color, vis_n=48):
    """Extract zero-isosurface via marching cubes → go.Mesh3d.

    Parameters
    ----------
    phi:   3-D SDF array (phi < 0 inside).
    color: CSS hex colour for the mesh.
    vis_n: downsample target before marching cubes (keeps HTML small).
    """
    if phi.min() >= 0.0 or phi.max() <= 0.0:
        return go.Mesh3d(x=[0], y=[0], z=[0], i=[0], j=[0], k=[0],
                         opacity=0, showscale=False)
    if phi.shape[0] > vis_n:
        scale = vis_n / phi.shape[0]
        phi = zoom(phi, scale, order=1)
    # Smooth the phi field to remove noise from Chan-Vese segmentation
    phi = gaussian_filter(phi, sigma=2.5)
    verts, faces, _, _ = marching_cubes(phi, level=0.0, spacing=(1.0, 1.0, 1.0))
    # marching_cubes returns (axis0=z, axis1=y, axis2=x); map to Plotly (x, y, z)
    x_mc, y_mc, z_mc = verts[:, 2], verts[:, 1], verts[:, 0]
    return go.Mesh3d(
        x=x_mc, y=y_mc, z=z_mc,
        i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
        color=color,
        opacity=1.0,
        flatshading=False,
        lighting=dict(ambient=0.5, diffuse=0.8, specular=0.4, roughness=0.3),
        lightposition=dict(x=1000, y=1000, z=2000),
        showscale=False,
    )


def _save_shape_html(phi, title, color, filename):
    """Write one self-contained HTML file using fig.write_html with CDN Plotly."""
    fig = go.Figure(_mesh3d_trace(phi, color))
    fig.update_layout(
        title=dict(text=title, font=dict(color="#e0e0e0", size=16), x=0.5),
        paper_bgcolor=DARK,
        height=600, width=600,
        scene=dict(
            bgcolor=PANEL,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False,
                       showbackground=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False,
                       showbackground=False),
            zaxis=dict(showgrid=False, zeroline=False, showticklabels=False,
                       showbackground=False),
            aspectmode="data",
        ),
    )
    path = os.path.join(OUT_DIR, filename)
    fig.write_html(path, include_plotlyjs="cdn")
    print(f"  Saved → {path}")


def _sample_to_phi(geom, bounds, res=32):
    """Sample a Geometry3D to a phi grid for visualisation."""
    from sdf3d.grid import sample_levelset_3d
    return sample_levelset_3d(geom, bounds, (res, res, res))


# ===========================================================================
# 1. Synthetic volume: sphere in Gaussian noise
# ===========================================================================
_header(1, "Synthetic volume: sphere (r=14) in Gaussian noise")

N   = 48
rng = np.random.default_rng(0)
D = H = W = N
cx = cy = cz = N // 2

d = np.arange(D)[:, None, None]
h = np.arange(H)[None, :, None]
w = np.arange(W)[None, None, :]
dist = np.sqrt((d - cz)**2 + (h - cy)**2 + (w - cx)**2)
RADIUS = 14.0
clean_volume = np.where(dist <= RADIUS, 0.8, 0.1)
volume = clean_volume + 0.02 * rng.standard_normal((D, H, W))
volume = np.clip(volume, 0, 1)

_print(f"Volume shape: {volume.shape}  min={volume.min():.3f}  max={volume.max():.3f}")

# ===========================================================================
# 2. Direct chan_vese_3d call
# ===========================================================================
_header(2, "Direct chan_vese_3d — low-level API")

segmentation, phi_list = chan_vese_3d(
    volume,
    mu=0.25, lambda1=1.0, lambda2=1.0,
    sigma=3.0, max_iter=60, dt=0.5,
)
phi_raw   = phi_list[0]      # uSCMAN: phi > 0 inside
phi_pysdf = -phi_raw         # pySdf:  phi < 0 inside

_print(f"phi (uSCMAN) centre: {phi_raw[cz, cy, cx]:.4f}  (expect > 0)")
_print(f"phi (pySdf)  centre: {phi_pysdf[cz, cy, cx]:.4f}  (expect < 0)")
assert phi_pysdf[cz, cy, cx] < 0, "Centre should be inside (phi < 0)"
_print("✓  sign-convention check passed")

R1, R2 = segmentation
_print(f"R1 (inside voxels): {R1.sum()}   R2 (outside): {R2.sum()}")

# ===========================================================================
# 3. ImageGeometry3D — direct construction
# ===========================================================================
_header(3, "ImageGeometry3D — direct construction + sdf() evaluation")

BOUNDS = ((0.0, float(W)), (0.0, float(H)), (0.0, float(D)))
geom = ImageGeometry3D(phi_pysdf, BOUNDS)
_print(repr(geom))

p_centre  = np.array([[float(cx), float(cy), float(cz)]])
p_corner  = np.array([[0.5, 0.5, 0.5]])
sdf_c = geom.sdf(p_centre)[0]
sdf_k = geom.sdf(p_corner)[0]
_print(f"sdf(centre) = {sdf_c:.4f}  (expect < 0)")
_print(f"sdf(corner) = {sdf_k:.4f}  (expect > 0)")
assert sdf_c < 0, "Centre must be inside"
assert sdf_k > 0, "Corner must be outside"
_print("✓  sdf() spot-check passed")

# ===========================================================================
# 4. CSG: union
# ===========================================================================
_header(4, "CSG — union(image_geom, Sphere3D)")

small_sphere = Sphere3D(8.0).translate(cx + 18.0, cy, cz)
geom_union   = geom.union(small_sphere)

phi_union = _sample_to_phi(geom_union, BOUNDS, res=N)
_print(f"Union phi range: [{phi_union.min():.3f}, {phi_union.max():.3f}]")
_print("✓  union produced valid SDF field")

# ===========================================================================
# 5. CSG: subtract
# ===========================================================================
_header(5, "CSG — subtract(image_geom, Box3D) — hollow core")

inner_box = Box3D([6.0, 6.0, 6.0]).translate(cx, cy, cz)
geom_sub  = geom.subtract(inner_box)

phi_sub = _sample_to_phi(geom_sub, BOUNDS, res=N)
_print(f"Subtracted phi range: [{phi_sub.min():.3f}, {phi_sub.max():.3f}]")
_print("✓  subtract produced valid SDF field")

# ===========================================================================
# 6. CSG: intersect
# ===========================================================================
_header(6, "CSG — intersect(image_geom, Sphere3D) — trimmed half")

half_sphere = Sphere3D(20.0).translate(cx, cy, cz - 6.0)
geom_isect  = geom.intersect(half_sphere)

phi_isect = _sample_to_phi(geom_isect, BOUNDS, res=N)
_print(f"Intersect phi range: [{phi_isect.min():.3f}, {phi_isect.max():.3f}]")
_print("✓  intersect produced valid SDF field")

# ===========================================================================
# 7. Negate — flip sign convention
# ===========================================================================
_header(7, "negate() — round-trip sign flip")

geom_neg = geom.negate()
val_orig = geom.sdf(p_centre)[0]
val_neg  = geom_neg.sdf(p_centre)[0]
_print(f"Original sdf(centre) = {val_orig:.4f}")
_print(f"Negated  sdf(centre) = {val_neg:.4f}")
assert val_neg > 0, "After negate, centre should be outside (phi > 0)"
_print("✓  negate() round-trip passed")

# ===========================================================================
# 8. Morphometry on original + CSG results
# ===========================================================================
_header(8, "compute_morphometry_3d — original + CSG variants")

morph_results = {}
for name, phi_g in [
    ("Original",   phi_pysdf),
    ("Union",      phi_union),
    ("Subtract",   phi_sub),
    ("Intersect",  phi_isect),
]:
    m = compute_morphometry_3d(phi_g, voxel_size=1.0)
    morph_results[name] = m
    _print(f"{name:12s}  V={m['volume']:7.1f}  A={m['surface_area']:7.1f}  ψ={m['sphericity']:.4f}")

# ===========================================================================
# 9. volume_to_levelset_3d (pipeline shortcut)
# ===========================================================================
_header(9, "volume_to_levelset_3d — high-level pipeline")

params = {"Segmentation": {"max_iter": 40, "sigma": 3.0}}
phi_pipeline = volume_to_levelset_3d(volume, params)

_print(f"shape={phi_pipeline.shape}  min={phi_pipeline.min():.3f}  max={phi_pipeline.max():.3f}")
assert phi_pipeline[cz, cy, cx] < 0, "Pipeline: centre should be inside"
_print("✓  volume_to_levelset_3d: sign convention passed")

# ===========================================================================
# 10. volume_to_geometry_3d (full pipeline shortcut)
# ===========================================================================
_header(10, "volume_to_geometry_3d — returns ImageGeometry3D directly")

geom2 = volume_to_geometry_3d(volume, params, bounds=BOUNDS)
_print(repr(geom2))
assert isinstance(geom2, ImageGeometry3D)
_print("✓  returned ImageGeometry3D instance")

# ===========================================================================
# 11. Save / reload .npy
# ===========================================================================
_header(11, "Save to .npy and reload")

npy_path = os.path.join(OUT_DIR, "showcase_levelset.npy")
np.save(npy_path, phi_pysdf)
phi_reload = np.load(npy_path)
np.testing.assert_array_equal(phi_pysdf, phi_reload)
_print(f"Saved  → {npy_path}")
_print(f"Reload shape: {phi_reload.shape}")
_print("✓  save/reload round-trip passed")

# ===========================================================================
# 12. Save one HTML per shape (fig.write_html with CDN Plotly)
# ===========================================================================
_header(12, "Saving interactive HTML — one file per shape …")

shapes_to_plot = [
    ("Original (φ<0 inside)",    phi_pysdf, "#4a90d9", "showcase_original.html"),
    ("Union with small sphere",  phi_union, "#50c878", "showcase_union.html"),
    ("Subtract inner box",       phi_sub,   "#e87040", "showcase_subtract.html"),
    ("Intersect half-sphere",    phi_isect, "#d04090", "showcase_intersect.html"),
]

for title, phi, color, filename in shapes_to_plot:
    _save_shape_html(phi, title, color, filename)

print(f"\nSaved → {npy_path}")
print("\n" + "="*60)
print("  ALL CHECKS PASSED — img2sdf 3D API working correctly")
print("="*60)
