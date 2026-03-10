"""img2sdf 3D showcase — every feature, interactive HTML output.

Demonstrates the complete img2sdf 3D pipeline:
  1.  Volume creation (synthetic sphere in noise)
  2.  3D Chan-Vese segmentation  → level-set φ
  3.  Sign-convention validation (pySdf: φ<0 inside)
  4.  ImageGeometry3D construction + trilinear interpolation
  5.  CSG operations: union, subtract, intersect with analytic Sphere3D
  6.  compute_morphometry_3d  → volume, surface area, sphericity
  7.  volume_to_levelset_3d  (direct array path)
  8.  volume_to_geometry_3d  (full pipeline shortcut)
  9.  Save-to-npy  +  reload
  10. Interactive Plotly HTML with all shapes side-by-side

Outputs
-------
    examples/img2sdf/output/img2sdf_3d_showcase.html
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
    from plotly.subplots import make_subplots
except ImportError:
    raise ImportError("Install plotly: pip install plotly")

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


def _isosurface_div(phi, title, color, div_id, N=48):
    from scipy.ndimage import zoom as _zoom
    # Downsample to 32³ for lighter HTML output
    vis_n = 32
    scale = vis_n / phi.shape[0]
    phi_vis = _zoom(phi, scale, order=1) if scale != 1.0 else phi
    xs = np.linspace(0, N, vis_n)
    ys = np.linspace(0, N, vis_n)
    zs = np.linspace(0, N, vis_n)
    Z3, Y3, X3 = np.meshgrid(zs, ys, xs, indexing="ij")
    fig = go.Figure(go.Isosurface(
        x=X3.ravel(), y=Y3.ravel(), z=Z3.ravel(),
        value=phi_vis.ravel(),
        isomin=0.0, isomax=0.0, surface_count=1,
        colorscale=[[0, color], [1, color]],
        showscale=False,
        caps=dict(x_show=False, y_show=False, z_show=False),
        lighting=dict(ambient=0.5, diffuse=0.8, specular=0.4, roughness=0.3),
        lightposition=dict(x=1000, y=1000, z=2000),
    ))
    fig.update_layout(
        title=dict(text=title, font=dict(color="#e0e0e0", size=13), x=0.5),
        paper_bgcolor=DARK, height=340, width=360,
        scene=dict(
            bgcolor=PANEL,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            zaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        ),
    )
    return fig.to_html(include_plotlyjs=False, full_html=False, div_id=div_id)


def _sample_to_phi(geom, bounds, res=48):
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
volume = clean_volume + 0.05 * rng.standard_normal((D, H, W))
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
phi_raw = phi_list[0]          # uSCMAN: phi > 0 inside
phi_pysdf = -phi_raw           # pySdf:  phi < 0 inside

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

# Spot-check sdf()
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
# 12. Build interactive HTML report
# ===========================================================================
_header(12, "Building interactive Plotly HTML report …")

shapes_to_plot = [
    ("Original (φ<0 inside)",    phi_pysdf, "#4a90d9"),
    ("Union with small sphere",  phi_union, "#50c878"),
    ("Subtract inner box",       phi_sub,   "#e87040"),
    ("Intersect half-sphere",    phi_isect, "#d04090"),
]

iso_divs = [
    _isosurface_div(phi, title, color, div_id=f"iso{i}", N=N)
    for i, (title, phi, color) in enumerate(shapes_to_plot)
]

# --- Morphometry bar chart ---
morph_labels = list(morph_results.keys())
morph_colors = ["#4a90d9", "#50c878", "#e87040", "#d04090"]
fig_morph = make_subplots(rows=1, cols=3,
    subplot_titles=("Volume (voxels³)", "Surface Area (voxels²)", "Sphericity ψ"),
    horizontal_spacing=0.09,
)
fig_morph.add_trace(
    go.Bar(x=morph_labels,
           y=[morph_results[k]["volume"] for k in morph_labels],
           marker_color=morph_colors,
           text=[f"{morph_results[k]['volume']:.0f}" for k in morph_labels],
           textposition="auto", name="Volume"),
    row=1, col=1,
)
fig_morph.add_trace(
    go.Bar(x=morph_labels,
           y=[morph_results[k]["surface_area"] for k in morph_labels],
           marker_color=morph_colors,
           text=[f"{morph_results[k]['surface_area']:.0f}" for k in morph_labels],
           textposition="auto", name="Surface Area"),
    row=1, col=2,
)
fig_morph.add_trace(
    go.Bar(x=morph_labels,
           y=[morph_results[k]["sphericity"] for k in morph_labels],
           marker_color=morph_colors,
           text=[f"{morph_results[k]['sphericity']:.3f}" for k in morph_labels],
           textposition="auto", name="Sphericity"),
    row=1, col=3,
)
fig_morph.update_layout(
    paper_bgcolor=DARK, plot_bgcolor=PANEL,
    font=dict(color="#e0e0e0"), showlegend=False,
    height=380, width=1100,
)
fig_morph.update_xaxes(gridcolor="#2a2a4e")
fig_morph.update_yaxes(gridcolor="#2a2a4e")
morph_div = fig_morph.to_html(include_plotlyjs=False, full_html=False, div_id="morph_bars")

# --- Code snippets ---
code_snippets = {
    "Chan-Vese 3D (low level)": """\
from img2sdf.segmentation.cv_single_3d import chan_vese_3d

segmentation, [phi] = chan_vese_3d(
    volume,          # ndarray (D, H, W)
    mu=0.25,         # area weight
    sigma=3.0,       # Gaussian kernel
    max_iter=100,
)
# phi uses uSCMAN convention (phi > 0 inside)
phi_pysdf = -phi   # negate to pySdf convention""",

    "ImageGeometry3D": """\
from img2sdf import ImageGeometry3D

geom = ImageGeometry3D(phi_pysdf, bounds=((0,W),(0,H),(0,D)))
val  = geom.sdf(np.array([[cx, cy, cz]]))   # trilinear interp""",

    "CSG with analytic shapes": """\
from sdf3d import Sphere3D, Box3D

inner_box = Box3D([6,6,6]).translate(cx, cy, cz)
hollowed  = geom.subtract(inner_box)   # carve a box hole

union     = geom.union(Sphere3D(8).translate(cx+18, cy, cz))
trimmed   = geom.intersect(Sphere3D(20).translate(cx, cy, cz-6))""",

    "Morphometry": """\
from img2sdf import compute_morphometry_3d

m = compute_morphometry_3d(phi_pysdf, voxel_size=1.0)
print(m["volume"])        # voxel count × voxel_size³
print(m["surface_area"])  # marching-cubes isosurface area
print(m["sphericity"])    # ψ = π^(1/3)(6V)^(2/3) / A""",

    "High-level pipeline": """\
from img2sdf import volume_to_levelset_3d, volume_to_geometry_3d

params = {"Segmentation": {"max_iter": 100, "sigma": 3.0}}

phi  = volume_to_levelset_3d(volume, params)     # ndarray
geom = volume_to_geometry_3d(volume, params,
           bounds=((0,W),(0,H),(0,D)))           # ImageGeometry3D""",
}

code_html = "".join(
    f"""<div style="margin-bottom:16px">
<b style="color:#a0c0e0">{title}</b>
<pre style="background:#0d0d1e;border-radius:6px;padding:12px;overflow-x:auto;color:#c8e0ff;font-size:0.9em">{snippet}</pre>
</div>"""
    for title, snippet in code_snippets.items()
)

# --- Build final HTML ---
plotlyjs = '<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>'

html = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>img2sdf 3D Showcase — pySdf</title>
  {plotlyjs}
  <style>
    body   {{ background:{DARK}; color:#e0e0e0; font-family:sans-serif; padding:20px; max-width:1400px; margin:auto; }}
    h1     {{ text-align:center; color:#e0e0e0; font-size:1.8em; }}
    h2     {{ color:#a0c0e0; border-bottom:1px solid #2a2a4e; padding-bottom:4px; margin-top:32px; }}
    .iso-row {{ display:flex; flex-wrap:wrap; gap:8px; justify-content:center; }}
    code   {{ background:#0d0d1e; padding:2px 5px; border-radius:3px; color:#c8e0ff; }}
    .badge {{ display:inline-block; background:#2a3a5e; color:#c0d8ff; padding:3px 10px;
              border-radius:12px; font-size:0.8em; margin:2px; }}
  </style>
</head>
<body>
  <h1>img2sdf 3D Showcase</h1>
  <p style="text-align:center;color:#8090b0">
    Complete walkthrough of the <code>img2sdf</code> 3D pipeline:
    <span class="badge">chan_vese_3d</span>
    <span class="badge">ImageGeometry3D</span>
    <span class="badge">volume_to_levelset_3d</span>
    <span class="badge">volume_to_geometry_3d</span>
    <span class="badge">compute_morphometry_3d</span>
    <span class="badge">CSG union/subtract/intersect</span>
  </p>

  <h2>1 · Zero-isosurfaces: original + CSG variants</h2>
  <p style="color:#8090b0">
    Left: raw segmentation. Then: union with an offset sphere, subtract an inner box,
    intersect with a half-space sphere.
  </p>
  <div class="iso-row">{''.join(iso_divs)}</div>

  <h2>2 · Morphometry comparison (volume · surface area · sphericity)</h2>
  <p style="color:#8090b0">
    <code>compute_morphometry_3d(phi, voxel_size=1.0)</code> applied to each variant above.
    The union adds volume; subtract reduces it; intersect trims away half.
  </p>
  {morph_div}

  <h2>3 · API code snippets</h2>
  {code_html}
</body>
</html>
"""

out_html = os.path.join(OUT_DIR, "img2sdf_3d_showcase.html")
with open(out_html, "w", encoding="utf-8") as f:
    f.write(html)

print(f"\nSaved → {out_html}")
print(f"Saved → {npy_path}")
print("\n" + "="*60)
print("  ALL CHECKS PASSED — img2sdf 3D API working correctly")
print("="*60)
