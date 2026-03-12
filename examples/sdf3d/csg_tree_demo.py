"""csg_tree_demo.py — Constructive Solid Geometry (CSG) tree demonstration.

Shows how Union (|), Intersection (/), and Subtraction (-) operators compose
analytic primitives into complex shapes step-by-step, rendering all panels as
interactive Plotly isosurfaces in a single HTML report.

Panel layout (4 rows × 4 cols)
-------------------------------
Row 1 — Primitive leaves:
    Sphere3D, Box3D, Torus3D, ConeExact3D

Row 2 — Binary CSG operations + start of deep tree:
    Union, Subtraction, Intersection, Level-1 base box

Row 3 — Deep CSG tree (levels 2–5):
    Level 2: base|sphere, Level 3: +intersect, Level 4: –cylinder, Level 5: |torus

Row 4 — Mixed modifiers & transforms:
    Onion shell, Elongated capsule, Union assembly, Final –box slice

Usage
-----
python examples/sdf3d/csg_tree_demo.py            # default res=32
python examples/sdf3d/csg_tree_demo.py --res 48   # higher quality
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Path setup — allow running from any working directory
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sdf3d import Sphere3D, Box3D, Torus3D, ConeExact3D, Cylinder3D

_OUTPUT_DIR = Path(__file__).parent / "output"
_BOUNDS = ((-0.8, 0.8), (-0.8, 0.8), (-0.8, 0.8))

# ---------------------------------------------------------------------------
# Text-based CSG tree printed to stdout
# ---------------------------------------------------------------------------

_CSG_TREE_TEXT = """\
CSG Tree — Deep multi-level composition (Rows 2–3):

  Level 5: with_torus
  └── Union
      ├── Level 4: with_hole
      │   └── Subtraction
      │       ├── Level 3: rounded
      │       │   └── Intersection
      │       │       ├── Level 2: with_sphere
      │       │       │   └── Union
      │       │       │       ├── Level 1: base = Box3D([0.30, 0.30, 0.30])
      │       │       │       └── Translate(Sphere3D r=0.25, dx=+0.30)
      │       │       └── Sphere3D r=0.55
      │       └── Cylinder3D(axis=[0,0], r=0.08)
      └── Translate(Torus3D [0.30, 0.05], dy=+0.35)

CSG Tree — Mixed modifiers & transforms (Row 4):

  Subtraction
  ├── Union
  │   ├── Onion(Sphere3D r=0.40, thickness=0.03)
  │   └── Translate(Elongate(Sphere3D r=0.15, hx=0.20), dz=+0.30)
  └── Translate(Box3D [0.50, 0.50, 0.10], dz=−0.15)
"""


# ---------------------------------------------------------------------------
# Build all panel geometries
# ---------------------------------------------------------------------------

def _make_panels() -> list[tuple]:
    """Return list of (geom, label, hex_color) for all 16 panels (4×4)."""

    # --- Row 1: Primitive leaves ---
    half_angle = math.radians(25)
    cone = ConeExact3D(
        [math.sin(half_angle), math.cos(half_angle)], height=0.55
    )
    row1 = [
        (Sphere3D(0.35),               "Sphere3D(r=0.35)",         "#4a90d9"),
        (Box3D([0.25, 0.25, 0.25]),    "Box3D(h=0.25)",            "#e94560"),
        (Torus3D([0.30, 0.10]),        "Torus3D(R=0.30, r=0.10)",  "#2ecc71"),
        (cone,                         "ConeExact3D(25°, h=0.55)", "#f39c12"),
    ]

    # --- Row 2: Binary CSG ops + Level-1 deep-tree seed ---
    s = Sphere3D(0.35)
    b = Box3D([0.25, 0.25, 0.25])
    row2 = [
        (s | b,                        "Union: sphere | box",      "#9b59b6"),
        (s - b,                        "Subtract: sphere − box",   "#e67e22"),
        (s / b,                        "Intersect: sphere / box",  "#1abc9c"),
        (Box3D([0.30, 0.30, 0.30]),    "Level 1: base Box3D",      "#3498db"),
    ]

    # --- Row 3: Deep CSG tree levels 2–5 ---
    base        = Box3D([0.30, 0.30, 0.30])
    with_sphere = base | Sphere3D(0.25).translate(0.3, 0.0, 0.0)
    rounded     = with_sphere / Sphere3D(0.55)
    with_hole   = rounded - Cylinder3D([0.0, 0.0], 0.08)
    with_torus  = with_hole | Torus3D([0.30, 0.05]).translate(0.0, 0.35, 0.0)
    row3 = [
        (with_sphere, "Level 2: base | sphere",       "#e74c3c"),
        (rounded,     "Level 3: + intersect sphere",  "#f1c40f"),
        (with_hole,   "Level 4: − cylinder",          "#2980b9"),
        (with_torus,  "Level 5: | torus (final)",     "#8e44ad"),
    ]

    # --- Row 4: Mixed modifiers & transforms ---
    shell    = Sphere3D(0.4).onion(0.03)
    capsule  = Sphere3D(0.15).elongate(0.2, 0.0, 0.0)
    assembly = shell | capsule.translate(0.0, 0.0, 0.3)
    final    = assembly - Box3D([0.5, 0.5, 0.1]).translate(0.0, 0.0, -0.15)
    row4 = [
        (shell,    "Onion shell (Sphere+onion)",   "#27ae60"),
        (capsule,  "Elongated capsule",            "#d35400"),
        (assembly, "Union: shell | capsule",       "#c0392b"),
        (final,    "Final: − box slice",           "#16a085"),
    ]

    return row1 + row2 + row3 + row4


# ---------------------------------------------------------------------------
# Build and save the Plotly HTML report
# ---------------------------------------------------------------------------

def _build_report(res: int, out_path: Path) -> None:
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        print("  plotly not installed — skipping HTML output")
        return

    panels = _make_panels()
    n_rows, n_cols = 4, 4
    assert len(panels) == n_rows * n_cols

    specs = [[{"type": "scene"}] * n_cols for _ in range(n_rows)]
    titles = [label for _, label, _ in panels]

    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        specs=specs,
        subplot_titles=titles,
        horizontal_spacing=0.04,
        vertical_spacing=0.06,
    )

    resolution = (res, res, res)
    (x0, x1), (y0, y1), (z0, z1) = _BOUNDS
    xs = np.linspace(x0, x1, res)
    ys = np.linspace(y0, y1, res)
    zs = np.linspace(z0, z1, res)
    Z3, Y3, X3 = np.meshgrid(zs, ys, xs, indexing="ij")

    total = len(panels)
    for idx, (geom, label, color) in enumerate(panels):
        row = idx // n_cols + 1
        col = idx % n_cols + 1

        print(f"  [{idx+1:02d}/{total}] {label}…", end=" ", flush=True)
        t0 = time.perf_counter()
        phi = geom.to_numpy(_BOUNDS, resolution)
        elapsed = time.perf_counter() - t0
        inside = (phi < 0).mean() * 100
        print(f"{elapsed:.1f}s  solid={inside:.1f}%")

        fig.add_trace(
            go.Isosurface(
                x=X3.ravel(), y=Y3.ravel(), z=Z3.ravel(),
                value=phi.ravel(),
                isomin=0.0, isomax=0.0, surface_count=1,
                colorscale=[[0, color], [1, color]],
                showscale=False,
                caps=dict(x_show=False, y_show=False, z_show=False),
                lighting=dict(ambient=0.4, diffuse=0.9, specular=0.5,
                              roughness=0.2),
                lightposition=dict(x=1000, y=1000, z=2000),
            ),
            row=row, col=col,
        )

    fig.update_scenes(
        aspectmode="data",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        zaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    )

    # Row section labels as annotations
    row_labels = [
        "Row 1 — Primitive Leaves",
        "Row 2 — Binary CSG Operations",
        "Row 3 — Deep CSG Tree (step-by-step)",
        "Row 4 — Mixed Modifiers & Transforms",
    ]
    row_y_positions = [0.97, 0.73, 0.49, 0.25]
    for label, y in zip(row_labels, row_y_positions):
        fig.add_annotation(
            text=f"<b>{label}</b>",
            xref="paper", yref="paper",
            x=0.5, y=y,
            showarrow=False,
            font=dict(size=13, color="#a0c4ff"),
            align="center",
        )

    fig.update_layout(
        title=dict(
            text="CSG Tree Demo — pySdf Constructive Solid Geometry",
            font=dict(size=20),
        ),
        width=1300,
        height=1500,
        paper_bgcolor="#1a1a2e",
        plot_bgcolor="#16213e",
        font=dict(color="#e0e0e0"),
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_path), include_plotlyjs="cdn")
    print(f"\n  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="CSG tree structure demo — renders all panels to one HTML"
    )
    parser.add_argument(
        "--res", type=int, default=32,
        help="Grid resolution per axis (default: 32)",
    )
    parser.add_argument(
        "--out", type=str, default=None,
        help="Output HTML path (default: examples/sdf3d/output/csg_tree_demo.html)",
    )
    args = parser.parse_args()

    out_html = Path(args.out) if args.out else _OUTPUT_DIR / "csg_tree_demo.html"

    print("CSG Tree Demo — pySdf\n")
    print(_CSG_TREE_TEXT)
    print(f"Rendering all panels (res={args.res}³)…\n")
    _build_report(args.res, out_html)
    print("\nDone.")


if __name__ == "__main__":
    main()
