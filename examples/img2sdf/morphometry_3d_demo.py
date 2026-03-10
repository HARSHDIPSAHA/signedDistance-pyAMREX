"""3D Morphometry showcase — all shapes, all features, interactive HTML.

Demonstrates ``compute_morphometry_3d`` on four synthetic shapes:
  • Sphere      — ψ ≈ 1.0  (most compact)
  • Cube        — ψ ≈ 0.76
  • Elongated box — ψ < cube (less spherical)
  • Torus       — ψ ≪ 1 (most non-spherical)

Outputs
-------
    examples/img2sdf/output/morphometry_3d_demo.html

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
    ("Sphere (r=20)",    make_sphere(N, r=20.0),           "#4a90d9"),
    ("Cube (h=16)",      make_cube(N, half=16.0),           "#e87040"),
    ("Elongated Box",    make_elongated_box(N, 24, 12, 8),  "#50c878"),
    ("Torus (R=18,r=6)", make_torus(N, R=18.0, r=6.0),     "#d04090"),
]

r_sphere    = 20.0
V_sphere_ex = (4/3) * math.pi * r_sphere**3
A_sphere_ex = 4 * math.pi * r_sphere**2

results = []
for label, phi, color in shapes:
    m = compute_morphometry_3d(phi, voxel_size=VS)
    results.append(dict(label=label, phi=phi, color=color, **m))
    print(f"{label:22s}  V={m['volume']:8.1f}  A={m['surface_area']:8.1f}  ψ={m['sphericity']:.4f}")
print(f"\nAnalytic sphere:       V={V_sphere_ex:8.1f}  A={A_sphere_ex:8.1f}  ψ=1.0000")


# ===========================================================================
# Helper: isosurface figure for one phi grid
# ===========================================================================

def _scene_cfg():
    return dict(
        bgcolor=PANEL,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        zaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    )


def _mesh3d_from_phi(phi, color):
    """Extract zero-isosurface with marching cubes and return a go.Mesh3d trace.

    Using go.Mesh3d (triangulated surface from marching cubes) instead of
    go.Isosurface avoids the silent rendering failure that Isosurface can
    exhibit when data is loaded via the Plotly CDN bundle.
    """
    if phi.min() >= 0.0 or phi.max() <= 0.0:
        # Degenerate case: no zero-crossing — return an invisible dummy trace
        return go.Mesh3d(x=[0], y=[0], z=[0], i=[0], j=[0], k=[0],
                         opacity=0, showscale=False)
    verts, faces, normals, _ = marching_cubes(phi, level=0.0, spacing=(1.0, 1.0, 1.0))
    # marching_cubes returns (axis0, axis1, axis2) = (z, y, x); assign to Plotly axes
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


_iso_div_counter = 0


def _iso_figure(phi, title, color):
    """Render a shape's zero-isosurface as an interactive Plotly HTML snippet."""
    global _iso_div_counter
    _iso_div_counter += 1
    fig = go.Figure(_mesh3d_from_phi(phi, color))
    fig.update_layout(
        title=dict(text=title, font=dict(color="#e0e0e0", size=14), x=0.5),
        paper_bgcolor=DARK, height=350, width=350,
        scene=_scene_cfg(),
    )
    return fig.to_html(include_plotlyjs=False, full_html=False,
                       div_id=f"iso_{_iso_div_counter}")


# ===========================================================================
# Bar chart figure: compare all metrics side by side
# ===========================================================================

labels  = [r["label"]        for r in results]
volumes = [r["volume"]       for r in results]
areas   = [r["surface_area"] for r in results]
psis    = [r["sphericity"]   for r in results]
colors  = [r["color"]        for r in results]

fig_bars = make_subplots(
    rows=1, cols=3,
    subplot_titles=("Volume  (voxels³)", "Surface Area  (voxels²)", "Sphericity  ψ"),
    horizontal_spacing=0.08,
)

fig_bars.add_trace(
    go.Bar(x=labels, y=volumes, marker_color=colors,
           text=[f"{v:.0f}" for v in volumes], textposition="auto", name="Volume"),
    row=1, col=1,
)
# Reference lines as scatter
fig_bars.add_trace(
    go.Scatter(x=labels, y=[V_sphere_ex]*len(labels),
               mode="lines", line=dict(color="white", dash="dash"),
               name=f"analytic sphere V={V_sphere_ex:.0f}"),
    row=1, col=1,
)

fig_bars.add_trace(
    go.Bar(x=labels, y=areas, marker_color=colors,
           text=[f"{a:.0f}" for a in areas], textposition="auto", name="Surface Area"),
    row=1, col=2,
)
fig_bars.add_trace(
    go.Scatter(x=labels, y=[A_sphere_ex]*len(labels),
               mode="lines", line=dict(color="white", dash="dash"),
               name=f"analytic sphere A={A_sphere_ex:.0f}"),
    row=1, col=2,
)

fig_bars.add_trace(
    go.Bar(x=labels, y=psis, marker_color=colors,
           text=[f"{p:.3f}" for p in psis], textposition="auto", name="Sphericity"),
    row=1, col=3,
)
fig_bars.add_trace(
    go.Scatter(x=labels, y=[1.0]*len(labels),
               mode="lines", line=dict(color="white", dash="dash"),
               name="ψ=1 (perfect sphere)"),
    row=1, col=3,
)

fig_bars.update_layout(
    paper_bgcolor=DARK, plot_bgcolor=PANEL,
    font=dict(color="#e0e0e0"),
    showlegend=False,
    height=420, width=1200,
)
fig_bars.update_xaxes(gridcolor="#2a2a4e")
fig_bars.update_yaxes(gridcolor="#2a2a4e")

bars_html = fig_bars.to_html(include_plotlyjs=False, full_html=False, div_id="bars")


# ===========================================================================
# Sphericity scatter: ψ vs relative volume (normalised to sphere)
# ===========================================================================

norm_vol = [v / V_sphere_ex for v in volumes]
fig_scatter = go.Figure(go.Scatter(
    x=norm_vol, y=psis,
    mode="markers+text",
    text=labels,
    textposition="top center",
    marker=dict(color=colors, size=18, symbol="circle",
                line=dict(color="white", width=1)),
    hovertemplate="<b>%{text}</b><br>Rel. Volume: %{x:.3f}<br>ψ: %{y:.4f}<extra></extra>",
))
fig_scatter.add_shape(type="line", x0=0, x1=1.05, y0=1, y1=1,
                      line=dict(color="white", dash="dash"))
fig_scatter.update_layout(
    title=dict(text="Sphericity vs Relative Volume", font=dict(color="#e0e0e0", size=14), x=0.5),
    xaxis=dict(title="Volume / V_sphere", gridcolor="#2a2a4e", color="#aaa"),
    yaxis=dict(title="Sphericity ψ", gridcolor="#2a2a4e", color="#aaa", range=[0, 1.1]),
    paper_bgcolor=DARK, plot_bgcolor=PANEL,
    font=dict(color="#e0e0e0"),
    height=420, width=600,
)
scatter_html = fig_scatter.to_html(include_plotlyjs=False, full_html=False, div_id="scatter")


# ===========================================================================
# Assemble final HTML
# ===========================================================================

iso_divs = [_iso_figure(r["phi"], r["label"], r["color"]) for r in results]

# Formula annotation
formula_html = """
<div style="background:#16213e;border-radius:8px;padding:16px 24px;margin:12px 0;font-family:monospace;color:#e0e0e0;font-size:1.05em">
  <b>Morphometric Formulas</b><br><br>
  <b>Volume</b> = count(φ &lt; 0) × voxel_size³<br>
  <b>Surface Area</b> = Σ triangle areas from marching-cubes isosurface at φ=0<br>
  <b>Sphericity</b> ψ = π<sup>1/3</sup> · (6V)<sup>2/3</sup> / A &nbsp;&nbsp;
  <em>(ψ = 1 for a perfect sphere)</em>
</div>
"""

# Results table
table_rows = "".join(
    f"<tr><td style='padding:4px 12px'>{r['label']}</td>"
    f"<td style='padding:4px 12px;text-align:right'>{r['volume']:.1f}</td>"
    f"<td style='padding:4px 12px;text-align:right'>{r['surface_area']:.1f}</td>"
    f"<td style='padding:4px 12px;text-align:right'>{r['sphericity']:.4f}</td></tr>"
    for r in results
)
table_html = f"""
<table style="border-collapse:collapse;color:#e0e0e0;font-family:monospace;background:#16213e;border-radius:8px;overflow:hidden">
  <tr style="background:#2a2a4e">
    <th style="padding:8px 12px">Shape</th>
    <th style="padding:8px 12px">Volume</th>
    <th style="padding:8px 12px">Surface Area</th>
    <th style="padding:8px 12px">Sphericity ψ</th>
  </tr>
  {table_rows}
  <tr style="border-top:1px solid #4a4a6e;font-style:italic;color:#9090b0">
    <td style="padding:4px 12px">Analytic sphere</td>
    <td style="padding:4px 12px;text-align:right">{V_sphere_ex:.1f}</td>
    <td style="padding:4px 12px;text-align:right">{A_sphere_ex:.1f}</td>
    <td style="padding:4px 12px;text-align:right">1.0000</td>
  </tr>
</table>
"""

plotlyjs = '<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>'

html = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>3D Morphometry Demo — pySdf img2sdf</title>
  {plotlyjs}
  <style>
    body {{ background:{DARK}; color:#e0e0e0; font-family:sans-serif; padding:20px; }}
    h1 {{ text-align:center; color:#e0e0e0; }}
    h2 {{ color:#a0c0e0; border-bottom:1px solid #2a2a4e; padding-bottom:4px; }}
    .iso-row {{ display:flex; flex-wrap:wrap; gap:8px; justify-content:center; }}
    .chart-row {{ display:flex; flex-wrap:wrap; gap:16px; justify-content:center; margin-top:16px; }}
  </style>
</head>
<body>
  <h1>3D Morphometry Demo — <code>compute_morphometry_3d</code></h1>

  <h2>1 · Shapes (zero-isosurfaces)</h2>
  <div class="iso-row">
    {''.join(iso_divs)}
  </div>

  <h2>2 · Metric comparison (bar charts)</h2>
  {bars_html}

  <h2>3 · Sphericity vs Relative Volume</h2>
  <div class="chart-row">
    {scatter_html}
    <div style="max-width:560px;margin-top:20px">{formula_html}{table_html}</div>
  </div>
</body>
</html>
"""

out_path = os.path.join(OUT_DIR, "morphometry_3d_demo.html")
with open(out_path, "w", encoding="utf-8") as f:
    f.write(html)
print(f"\nSaved → {out_path}")
