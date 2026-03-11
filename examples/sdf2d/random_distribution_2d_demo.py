"""random_distribution_2d_demo.py — 2D random/inline/staggered circle placement.

Demonstrates the three center-distribution strategies from
:mod:`sdf2d.distributions` by placing circles in a 2-D bounding box,
evaluating the resulting SDF on a grid, and saving an interactive Plotly
HTML report (heatmap + contour of the union).

Usage
-----
python examples/sdf2d/random_distribution_2d_demo.py           # default res=256
python examples/sdf2d/random_distribution_2d_demo.py --res 512 # higher quality
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

from sdf2d import (
    Circle2D,
    Box2D,
    generate_centers_random,
    generate_centers_inline,
    generate_centers_staggered,
    distribute_shape,
)

_OUTPUT_DIR = Path(__file__).parent / "output"
_BOUNDS = ((-1.5, 1.5), (-1.5, 1.5))
_CIRCLE_RADIUS = 0.18
_MIN_SEP = 2.0 * _CIRCLE_RADIUS + 0.05


# ---------------------------------------------------------------------------
# Build demo geometries
# ---------------------------------------------------------------------------

def _make_distributions(seed: int = 0) -> list[dict]:
    scenarios = []

    # 1 ─ Random
    t0 = time.perf_counter()
    centers_r = generate_centers_random(
        _BOUNDS, num_centers=15, min_separation=_MIN_SEP, seed=seed
    )
    union_r = distribute_shape(lambda: Circle2D(_CIRCLE_RADIUS), centers_r)
    scenarios.append({"name": "Random (15 circles)", "geom": union_r,
                       "centers": centers_r})
    print(f"  Random: {len(centers_r)} centers in {time.perf_counter()-t0:.2f}s")

    # 2 ─ Inline (4×4 grid)
    t0 = time.perf_counter()
    centers_i = generate_centers_inline(_BOUNDS, num_centers_per_axis=4)
    union_i = distribute_shape(lambda: Circle2D(_CIRCLE_RADIUS), centers_i)
    scenarios.append({"name": "Inline (4×4 grid)", "geom": union_i,
                       "centers": centers_i})
    print(f"  Inline: {len(centers_i)} centers in {time.perf_counter()-t0:.2f}s")

    # 3 ─ Staggered (4×4)
    t0 = time.perf_counter()
    centers_s = generate_centers_staggered(_BOUNDS, num_centers_per_axis=4)
    union_s = distribute_shape(lambda: Circle2D(_CIRCLE_RADIUS), centers_s)
    scenarios.append({"name": "Staggered (4×4, offset)", "geom": union_s,
                       "centers": centers_s})
    print(f"  Staggered: {len(centers_s)} centers in {time.perf_counter()-t0:.2f}s")

    # 4 ─ Random boxes for variety
    t0 = time.perf_counter()
    centers_rb = generate_centers_random(
        _BOUNDS, num_centers=10, min_separation=_MIN_SEP + 0.1, seed=seed + 1
    )
    union_rb = distribute_shape(
        lambda: Box2D((_CIRCLE_RADIUS * 0.9, _CIRCLE_RADIUS * 0.9)), centers_rb
    )
    scenarios.append({"name": "Random boxes (10)", "geom": union_rb,
                       "centers": centers_rb})
    print(f"  Random boxes: {len(centers_rb)} centers in {time.perf_counter()-t0:.2f}s")

    return scenarios


# ---------------------------------------------------------------------------
# Build Plotly HTML report
# ---------------------------------------------------------------------------

def _build_report(scenarios: list[dict], res: int, out_path: Path) -> None:
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        print("  plotly not installed — skipping HTML output")
        return

    n = len(scenarios)
    rows, cols = 2, 2

    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=[s["name"] for s in scenarios],
        horizontal_spacing=0.08,
        vertical_spacing=0.12,
    )

    palettes = ["RdBu", "Viridis", "Cividis", "Plasma"]
    (x0, x1), (y0, y1) = _BOUNDS
    xs = np.linspace(x0, x1, res)
    ys = np.linspace(y0, y1, res)
    YY, XX = np.meshgrid(ys, xs, indexing="ij")
    resolution = (res, res)

    for idx, (scenario, palette) in enumerate(zip(scenarios, palettes)):
        row = idx // cols + 1
        col = idx % cols + 1

        phi = scenario["geom"].to_numpy(_BOUNDS, resolution)

        # Heatmap of phi
        fig.add_trace(
            go.Heatmap(
                x=xs, y=ys, z=phi,
                colorscale=palette,
                zmid=0.0,
                showscale=False,
            ),
            row=row, col=col,
        )

        # Zero contour
        fig.add_trace(
            go.Contour(
                x=xs, y=ys, z=phi,
                contours=dict(
                    coloring="none",
                    showlabels=False,
                    start=0, end=0, size=0,
                ),
                line=dict(color="white", width=2),
                showscale=False,
            ),
            row=row, col=col,
        )

        # Center markers
        fig.add_trace(
            go.Scatter(
                x=scenario["centers"][:, 0],
                y=scenario["centers"][:, 1],
                mode="markers",
                marker=dict(size=6, color="yellow", symbol="circle-open",
                            line=dict(width=2)),
                showlegend=False,
            ),
            row=row, col=col,
        )

    fig.update_layout(
        title=dict(
            text="2D Circle/Box Distributions: Random / Inline / Staggered",
            font=dict(size=18),
        ),
        width=900,
        height=800,
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
    parser = argparse.ArgumentParser(
        description="2D random/inline/staggered distribution demo"
    )
    parser.add_argument("--res", type=int, default=200,
                        help="Grid resolution per axis (default: 200)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    args = parser.parse_args()

    print("Building 2D distributions…")
    scenarios = _make_distributions(seed=args.seed)

    out_html = _OUTPUT_DIR / "random_distribution_2d_report.html"
    print(f"\nRendering HTML (res={args.res})…")
    _build_report(scenarios, args.res, out_html)

    print("\nDone.")


if __name__ == "__main__":
    main()
