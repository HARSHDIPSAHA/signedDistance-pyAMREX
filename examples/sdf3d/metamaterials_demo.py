"""metamaterials_demo.py — TPMS / lattice metamaterial gallery.

Generates and visualises Gyroid, Schwarz-P, Schwarz-D, Neovius,
BCC-lattice, and FCC-lattice unit cells using the classes from
:mod:`sdf3d.metamaterials`.

Each shape is rendered as an interactive Plotly isosurface and all panels
are written to a single HTML report.

Usage
-----
python examples/sdf3d/metamaterials_demo.py           # default res=32
python examples/sdf3d/metamaterials_demo.py --res 48  # higher quality
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sdf3d import (
    Gyroid3D,
    SchwarzP3D,
    SchwarzD3D,
    Neovius3D,
    BCCLattice3D,
    FCCLattice3D,
)

_OUTPUT_DIR = Path(__file__).parent / "output"
_BOUNDS = ((-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0))

# TPMS default thickness (chosen for good visibility at modest resolutions)
_TPMS_THICKNESS = 0.25

# Panels: (name, geom_factory)
_PANELS = [
    ("Gyroid",    lambda: Gyroid3D(cell_size=2.0,  thickness=_TPMS_THICKNESS)),
    ("Schwarz-P", lambda: SchwarzP3D(cell_size=2.0, thickness=_TPMS_THICKNESS)),
    ("Schwarz-D", lambda: SchwarzD3D(cell_size=2.0, thickness=_TPMS_THICKNESS)),
    ("Neovius",   lambda: Neovius3D(cell_size=2.0,  thickness=_TPMS_THICKNESS)),
    ("BCC Lattice", lambda: BCCLattice3D(cell_size=0.67, beam_radius=0.06, repeat=(3, 3, 3))),
    ("FCC Lattice", lambda: FCCLattice3D(cell_size=0.67, beam_radius=0.06, repeat=(3, 3, 3))),
]


# ---------------------------------------------------------------------------
# Build Plotly HTML report
# ---------------------------------------------------------------------------

def _build_report(res: int, out_path: Path) -> None:
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        print("  plotly not installed — skipping HTML output")
        return

    n = len(_PANELS)
    # Display in two rows
    rows, cols_per_row = 2, 3
    specs = [[{"type": "scene"}] * cols_per_row for _ in range(rows)]
    titles = [name for name, _ in _PANELS]

    fig = make_subplots(
        rows=rows,
        cols=cols_per_row,
        specs=specs,
        subplot_titles=titles,
        horizontal_spacing=0.05,
        vertical_spacing=0.08,
    )

    palette = [
        "#4a90d9", "#e94560", "#2ecc71",
        "#f39c12", "#9b59b6", "#1abc9c",
    ]

    resolution = (res, res, res)
    (x0, x1), (y0, y1), (z0, z1) = _BOUNDS
    xs = np.linspace(x0, x1, res)
    ys = np.linspace(y0, y1, res)
    zs = np.linspace(z0, z1, res)
    Z3, Y3, X3 = np.meshgrid(zs, ys, xs, indexing="ij")

    for idx, ((name, factory), color) in enumerate(zip(_PANELS, palette)):
        row = idx // cols_per_row + 1
        col = idx % cols_per_row + 1

        print(f"  [{idx+1}/{n}] {name}…", end=" ", flush=True)
        t0 = time.perf_counter()
        geom = factory()
        phi = geom.to_numpy(_BOUNDS, resolution)
        elapsed = time.perf_counter() - t0

        inside = (phi < 0).mean() * 100
        print(f"{elapsed:.1f}s  phi∈[{phi.min():.3f}, {phi.max():.3f}]  "
              f"solid={inside:.1f}%")

        if phi.min() >= 0:
            print(f"    WARNING: no solid interior — try larger thickness or "
                  f"smaller cell_size")

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
    fig.update_layout(
        title=dict(
            text="Metamaterial / TPMS Gallery",
            font=dict(size=22),
        ),
        width=900,
        height=750,
        paper_bgcolor="#1a1a2e",
        plot_bgcolor="#16213e",
        font=dict(color="#e0e0e0"),
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_path), include_plotlyjs="cdn")
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="TPMS / lattice gallery demo")
    parser.add_argument(
        "--res", type=int, default=32,
        help="Grid resolution per axis (default: 32)"
    )
    args = parser.parse_args()

    out_html = _OUTPUT_DIR / "metamaterials_report.html"
    print(f"Rendering metamaterials gallery (res={args.res})…\n")
    _build_report(args.res, out_html)
    print("\nDone.")


if __name__ == "__main__":
    main()
