"""random_distribution_demo.py — 3D random/inline/staggered sphere placement.

Demonstrates the three center-distribution strategies from
:mod:`sdf3d.distributions` by placing spheres in a box domain, evaluating
the resulting SDF on a 3-D grid, and saving an interactive Plotly HTML report.

Usage
-----
python examples/sdf3d/random_distribution_demo.py           # default res=32
python examples/sdf3d/random_distribution_demo.py --res 48  # higher quality
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Path setup — allow running from any working directory
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sdf3d import (
    Sphere3D,
    generate_centers_random,
    generate_centers_inline,
    generate_centers_staggered,
    distribute_shape,
)

_OUTPUT_DIR = Path(__file__).parent / "output"
_BOUNDS = ((-1.5, 1.5), (-1.5, 1.5), (-1.5, 1.5))
_SPHERE_RADIUS = 0.18
_MIN_SEP = 2.0 * _SPHERE_RADIUS + 0.05  # small gap between spheres


# ---------------------------------------------------------------------------
# Build demo geometries
# ---------------------------------------------------------------------------

def _make_distributions(seed: int = 0) -> list[dict]:
    """Return a list of dicts describing each distribution scenario."""
    scenarios = []

    # 1 ─ Random distribution
    t0 = time.perf_counter()
    centers_r = generate_centers_random(
        _BOUNDS, num_centers=12, min_separation=_MIN_SEP, seed=seed
    )
    union_r = distribute_shape(lambda: Sphere3D(_SPHERE_RADIUS), centers_r)
    scenarios.append(
        {"name": "Random (12 spheres)", "geom": union_r, "centers": centers_r}
    )
    print(f"  Random: {len(centers_r)} centers in {time.perf_counter()-t0:.2f}s")

    # 2 ─ Inline (3×3×3 grid)
    t0 = time.perf_counter()
    centers_i = generate_centers_inline(_BOUNDS, num_centers_per_axis=3)
    union_i = distribute_shape(lambda: Sphere3D(_SPHERE_RADIUS), centers_i)
    scenarios.append(
        {"name": "Inline (3³ grid)", "geom": union_i, "centers": centers_i}
    )
    print(f"  Inline: {len(centers_i)} centers in {time.perf_counter()-t0:.2f}s")

    # 3 ─ Staggered (3×3×3 layers, each odd layer offset)
    t0 = time.perf_counter()
    centers_s = generate_centers_staggered(_BOUNDS, num_centers_per_axis=3)
    union_s = distribute_shape(lambda: Sphere3D(_SPHERE_RADIUS), centers_s)
    scenarios.append(
        {"name": "Staggered (3³, offset)", "geom": union_s, "centers": centers_s}
    )
    print(f"  Staggered: {len(centers_s)} centers in {time.perf_counter()-t0:.2f}s")

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
    fig = make_subplots(
        rows=1,
        cols=n,
        specs=[[{"type": "scene"}] * n],
        subplot_titles=[s["name"] for s in scenarios],
        horizontal_spacing=0.05,
    )

    colors = ["#4a90d9", "#e94560", "#2ecc71"]
    resolution = (res, res, res)
    (x0, x1), (y0, y1), (z0, z1) = _BOUNDS

    for col, (scenario, color) in enumerate(zip(scenarios, colors), start=1):
        phi = scenario["geom"].to_numpy(_BOUNDS, resolution)
        xs = np.linspace(x0, x1, res)
        ys = np.linspace(y0, y1, res)
        zs = np.linspace(z0, z1, res)
        Z3, Y3, X3 = np.meshgrid(zs, ys, xs, indexing="ij")

        # Isosurface
        fig.add_trace(
            go.Isosurface(
                x=X3.ravel(), y=Y3.ravel(), z=Z3.ravel(),
                value=phi.ravel(),
                isomin=0.0, isomax=0.0, surface_count=1,
                colorscale=[[0, color], [1, color]],
                showscale=False,
                caps=dict(x_show=False, y_show=False, z_show=False),
                lighting=dict(ambient=0.4, diffuse=0.9, specular=0.3),
                lightposition=dict(x=1000, y=1000, z=2000),
            ),
            row=1, col=col,
        )

        # Center markers
        cx = scenario["centers"][:, 0]
        cy = scenario["centers"][:, 1]
        cz = scenario["centers"][:, 2]
        fig.add_trace(
            go.Scatter3d(
                x=cx, y=cy, z=cz,
                mode="markers",
                marker=dict(size=4, color="white", opacity=0.7),
                showlegend=False,
            ),
            row=1, col=col,
        )

    fig.update_scenes(
        aspectmode="data",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        zaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    )
    fig.update_layout(
        title=dict(
            text="3D Sphere Distributions: Random / Inline / Staggered",
            font=dict(size=20),
        ),
        width=600 * n,
        height=650,
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
        description="3D random/inline/staggered distribution demo"
    )
    parser.add_argument("--res", type=int, default=32,
                        help="Grid resolution per axis (default: 32)")
    parser.add_argument("--seed", type=int, default=7,
                        help="Random seed for reproducibility (default: 7)")
    args = parser.parse_args()

    print("Building distributions…")
    scenarios = _make_distributions(seed=args.seed)

    out_html = _OUTPUT_DIR / "random_distribution_report.html"
    print(f"\nRendering HTML (res={args.res})…")
    _build_report(scenarios, args.res, out_html)

    print("\nDone.")


if __name__ == "__main__":
    main()
