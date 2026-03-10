"""3D Morphometry demo — one standalone HTML file per shape.

Demonstrates ``compute_morphometry_3d`` on four synthetic shapes:
  • Sphere      — ψ ≈ 1.0  (most compact)
  • Cube        — ψ ≈ 0.76
  • Elongated box — ψ < cube (less spherical)
  • Torus       — ψ ≪ 1 (most non-spherical)

Outputs (one HTML per shape + a bar-chart summary)
-------
    examples/img2sdf/output/morphometry_sphere.html
    examples/img2sdf/output/morphometry_cube.html
    examples/img2sdf/output/morphometry_elongated_box.html
    examples/img2sdf/output/morphometry_torus.html
    examples/img2sdf/output/morphometry_bars.html

Usage
-----
    python morphometry_3d_demo.py
"""
from __future__ import annotations
import os
import sys
import math
import numpy as np

# Make sure the repo root is on the path when run directly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from img2sdf import compute_morphometry_3d

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ImportError:
    raise ImportError("Install plotly: pip install plotly")

try:
    from skimage.measure import marching_cubes
except ImportError:
    raise ImportError("Install scikit-image: pip install scikit-image")

# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------
OUT_DIR = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUT_DIR, exist_ok=True)

DARK  = "#1a1a2e"
PANEL = "#16213e"
N     = 64      # grid size (N³)
VS    = 1.0     # voxel size


# ===========================================================================
# SDF factories — all centred in an N³ grid
# ===========================================================================

def _mgrid(N):
    c = N / 2.0
    d = np.arange(N)[:, None, None] - c
    h = np.arange(N)[None, :, None] - c
    w = np.arange(N)[None, None, :] - c
    return d, h, w


def make_sphere(N=64, r=20.0):
    d, h, w = _mgrid(N)
    return np.sqrt(d**2 + h**2 + w**2) - r


def make_cube(N=64, half=16.0):
    d, h, w = _mgrid(N)
    qd = np.abs(d) - half
    qh = np.abs(h) - half
    qw = np.abs(w) - half
    return (np.sqrt(np.maximum(qd, 0)**2 + np.maximum(qh, 0)**2 + np.maximum(qw, 0)**2)
            + np.minimum(np.maximum(qd, np.maximum(qh, qw)), 0))


def make_elongated_box(N=64, hd=24.0, hh=12.0, hw=8.0):
    d, h, w = _mgrid(N)
    qd = np.abs(d) - hd
    qh = np.abs(h) - hh
    qw = np.abs(w) - hw
    return (np.sqrt(np.maximum(qd, 0)**2 + np.maximum(qh, 0)**2 + np.maximum(qw, 0)**2)
            + np.minimum(np.maximum(qd, np.maximum(qh, qw)), 0))


def make_torus(N=64, R=18.0, r=6.0):
    d, h, w = _mgrid(N)
    ring = np.sqrt(w**2 + h**2) - R
    return np.sqrt(ring**2 + d**2) - r


# ===========================================================================
# Build shapes and compute morphometry
# ===========================================================================

shapes = [
    ("Sphere (r=20)",    "morphometry_sphere",       make_sphere(N, r=20.0),           "#4a90d9"),
    ("Cube (h=16)",      "morphometry_cube",          make_cube(N, half=16.0),           "#e87040"),
    ("Elongated Box",    "morphometry_elongated_box", make_elongated_box(N, 24, 12, 8),  "#50c878"),
    ("Torus (R=18,r=6)", "morphometry_torus",         make_torus(N, R=18.0, r=6.0),      "#d04090"),
]

r_sphere    = 20.0
V_sphere_ex = (4/3) * math.pi * r_sphere**3
A_sphere_ex = 4 * math.pi * r_sphere**2

results = []
for label, slug, phi, color in shapes:
    m = compute_morphometry_3d(phi, voxel_size=VS)
    results.append(dict(label=label, slug=slug, phi=phi, color=color, **m))
    print(f"{label:22s}  V={m['volume']:8.1f}  A={m['surface_area']:8.1f}  ψ={m['sphericity']:.4f}")
print(f"\nAnalytic sphere:       V={V_sphere_ex:8.1f}  A={A_sphere_ex:8.1f}  ψ=1.0000")


# ===========================================================================
# Helper: build and save one standalone HTML per shape
# ===========================================================================

def _mesh3d_trace(phi, color):
    """Extract zero-isosurface with marching cubes → go.Mesh3d."""
    if phi.min() >= 0.0 or phi.max() <= 0.0:
        return go.Mesh3d(x=[0], y=[0], z=[0], i=[0], j=[0], k=[0],
                         opacity=0, showscale=False)
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
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            zaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            aspectmode="data",
        ),
    )
    path = os.path.join(OUT_DIR, filename)
    fig.write_html(path, include_plotlyjs="cdn")
    print(f"  Saved → {path}")


# ===========================================================================
# Save one HTML per shape
# ===========================================================================

for r in results:
    _save_shape_html(r["phi"], r["label"], r["color"], f"{r['slug']}.html")


# ===========================================================================
# Bar chart: compare all metrics — also saved as its own HTML
# ===========================================================================

labels  = [r["label"]        for r in results]
volumes = [r["volume"]       for r in results]
areas   = [r["surface_area"] for r in results]
psis    = [r["sphericity"]   for r in results]
colors  = [r["color"]        for r in results]

fig_bars = make_subplots(
    rows=1, cols=3,
    subplot_titles=("Volume (voxels³)", "Surface Area (voxels²)", "Sphericity ψ"),
    horizontal_spacing=0.08,
)

fig_bars.add_trace(
    go.Bar(x=labels, y=volumes, marker_color=colors,
           text=[f"{v:.0f}" for v in volumes], textposition="auto", name="Volume"),
    row=1, col=1,
)
fig_bars.add_trace(
    go.Scatter(x=labels, y=[V_sphere_ex] * len(labels),
               mode="lines", line=dict(color="white", dash="dash"),
               name=f"sphere V={V_sphere_ex:.0f}"),
    row=1, col=1,
)
fig_bars.add_trace(
    go.Bar(x=labels, y=areas, marker_color=colors,
           text=[f"{a:.0f}" for a in areas], textposition="auto", name="Surface Area"),
    row=1, col=2,
)
fig_bars.add_trace(
    go.Scatter(x=labels, y=[A_sphere_ex] * len(labels),
               mode="lines", line=dict(color="white", dash="dash"),
               name=f"sphere A={A_sphere_ex:.0f}"),
    row=1, col=2,
)
fig_bars.add_trace(
    go.Bar(x=labels, y=psis, marker_color=colors,
           text=[f"{p:.3f}" for p in psis], textposition="auto", name="Sphericity"),
    row=1, col=3,
)
fig_bars.add_trace(
    go.Scatter(x=labels, y=[1.0] * len(labels),
               mode="lines", line=dict(color="white", dash="dash"),
               name="ψ=1 (sphere)"),
    row=1, col=3,
)

fig_bars.update_layout(
    paper_bgcolor=DARK, plot_bgcolor=PANEL,
    font=dict(color="#e0e0e0"),
    showlegend=False,
    height=500, width=1200,
)
fig_bars.update_xaxes(gridcolor="#2a2a4e")
fig_bars.update_yaxes(gridcolor="#2a2a4e")

bars_path = os.path.join(OUT_DIR, "morphometry_bars.html")
fig_bars.write_html(bars_path, include_plotlyjs="cdn")
print(f"  Saved → {bars_path}")
