"""morphometry_stats_demo.py — Morphometric statistical analysis of synthetic shapes.

Generates a population of N random SDF shapes, computes morphometric properties
(volume, surface area, sphericity via ``compute_morphometry_3d``), plus additional
numpy-based metrics (aspect ratio, principal-axis orientation, equivalent diameters),
and visualises the results in a single interactive Plotly HTML report.

Metrics computed per shape
--------------------------
- **Volume**, **surface area**, **sphericity** — via ``compute_morphometry_3d``
- **Aspect ratio** — max_extent / min_extent of the bounding box of φ<0 voxels
- **Orientation angle** — angle (°) of the principal axis from the inertia tensor
  of the φ<0 voxels (projected onto the XY-plane)
- **d_V** — equivalent sphere diameter from volume: ``(6V/π)^(1/3)``
- **d_A** — equivalent sphere diameter from surface area: ``sqrt(A/π)``

HTML panels (2 rows × 4 cols, 7 active subplots)
-------------------------------------------------
1. Scatter: Aspect Ratio vs Sphericity (coloured by shape type)
2. Histogram: Orientation angle distribution
3. Histogram: Equivalent diameter from volume (d_V)
4. Histogram: Equivalent diameter from surface area (d_A)
5. Scatter: d_V vs d_A with y=x reference line
6. Bar chart: Mean sphericity by shape type
7. Box plot: Aspect ratio distribution by shape type

Usage
-----
python examples/sdf3d/morphometry_stats_demo.py               # N=50, res=48
python examples/sdf3d/morphometry_stats_demo.py --res 32      # faster draft
python examples/sdf3d/morphometry_stats_demo.py --n 100       # larger population
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

from img2sdf import compute_morphometry_3d
from sdf3d import Sphere3D, Box3D, Torus3D

_OUTPUT_DIR = Path(__file__).parent / "output"
_BOUNDS = ((-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0))

SHAPE_TYPES = ["sphere", "box", "elongated_box", "torus", "capsule"]
TYPE_COLORS = {
    "sphere":        "#4a90d9",
    "box":           "#e94560",
    "elongated_box": "#2ecc71",
    "torus":         "#f39c12",
    "capsule":       "#9b59b6",
}


# ---------------------------------------------------------------------------
# Shape generation with random dimensions and orientations
# ---------------------------------------------------------------------------

def _random_shape(rng: np.random.Generator):
    """Return ``(sdf_geom, shape_type)`` for a randomly configured shape."""
    stype = rng.choice(SHAPE_TYPES)

    if stype == "sphere":
        r = rng.uniform(0.15, 0.45)
        geom = Sphere3D(r)

    elif stype == "box":
        s = rng.uniform(0.12, 0.35)
        geom = Box3D([s, s, s])

    elif stype == "elongated_box":
        s = rng.uniform(0.10, 0.22)
        e = rng.uniform(0.08, 0.30)
        geom = Box3D([s, s, s]).elongate(e, 0.0, 0.0)

    elif stype == "torus":
        R = rng.uniform(0.22, 0.42)
        r = rng.uniform(0.05, min(0.14, R * 0.38))
        geom = Torus3D([R, r])

    else:  # capsule — elongated sphere
        r = rng.uniform(0.10, 0.24)
        e = rng.uniform(0.08, 0.30)
        geom = Sphere3D(r).elongate(e, 0.0, 0.0)

    # Apply random 3-D rotation so orientations are uniformly distributed
    rx = rng.uniform(0.0, 2.0 * math.pi)
    ry = rng.uniform(0.0, 2.0 * math.pi)
    rz = rng.uniform(0.0, 2.0 * math.pi)
    geom = geom.rotate_x(rx).rotate_y(ry).rotate_z(rz)

    return geom, stype


# ---------------------------------------------------------------------------
# Per-shape metric helpers
# ---------------------------------------------------------------------------

def _aspect_ratio(phi: np.ndarray) -> float:
    """Bounding-box aspect ratio (max_extent / min_extent) of the φ<0 region."""
    inside = np.argwhere(phi < 0.0)
    if inside.shape[0] < 2:
        return 1.0
    extents = (inside.max(axis=0) - inside.min(axis=0) + 1).astype(float)
    min_e = extents.min()
    if min_e <= 0.0:
        return 1.0
    return float(extents.max() / min_e)


def _orientation_angle(phi: np.ndarray) -> float:
    """Principal-axis angle (°) in the XY-plane from the inertia tensor of φ<0 voxels."""
    inside = np.argwhere(phi < 0.0).astype(float)
    if inside.shape[0] < 3:
        return 0.0
    delta = inside - inside.mean(axis=0)
    # 3×3 second-moment (covariance) matrix
    cov = (delta[:, :, None] * delta[:, None, :]).mean(axis=0)
    eigvals, eigvecs = np.linalg.eigh(cov)
    # Largest eigenvalue → principal axis
    principal = eigvecs[:, eigvals.argmax()]
    return math.degrees(math.atan2(float(principal[1]), float(principal[0])))


# ---------------------------------------------------------------------------
# Population generation
# ---------------------------------------------------------------------------

def _build_population(n: int, res: int, seed: int = 42) -> list[dict]:
    """Sample *n* random shapes and compute all morphometric metrics."""
    rng = np.random.default_rng(seed)
    resolution = (res, res, res)
    voxel_size = 2.0 / res  # physical edge length of each voxel (bounds span 2 units)

    results: list[dict] = []
    for i in range(n):
        geom, stype = _random_shape(rng)
        phi = geom.to_numpy(_BOUNDS, resolution)

        m = compute_morphometry_3d(phi, voxel_size=voxel_size)
        V = m["volume"]
        A = m["surface_area"]

        d_V = (6.0 * V / math.pi) ** (1.0 / 3.0) if V > 0.0 else 0.0
        d_A = math.sqrt(A / math.pi) if A > 0.0 else 0.0

        results.append({
            "shape_type":   stype,
            "volume":       V,
            "surface_area": A,
            "sphericity":   m["sphericity"],
            "aspect_ratio": _aspect_ratio(phi),
            "orientation":  _orientation_angle(phi),
            "d_V":          d_V,
            "d_A":          d_A,
        })

        print(
            f"  [{i+1:3d}/{n}] {stype:<14s}  "
            f"ψ={m['sphericity']:.3f}  "
            f"AR={results[-1]['aspect_ratio']:.2f}  "
            f"θ={results[-1]['orientation']:+.1f}°"
        )

    return results


# ---------------------------------------------------------------------------
# Summary table printed to stdout
# ---------------------------------------------------------------------------

def _print_summary(results: list[dict]) -> None:
    print("\n" + "=" * 74)
    print(
        f"  {'Shape Type':<16} {'N':>4}  "
        f"{'Mean ψ':>8}  {'Mean AR':>8}  "
        f"{'Mean d_V':>9}  {'Mean d_A':>9}"
    )
    print("-" * 74)
    for stype in SHAPE_TYPES:
        grp = [r for r in results if r["shape_type"] == stype]
        if not grp:
            continue
        print(
            f"  {stype:<16} {len(grp):>4d}  "
            f"{np.mean([r['sphericity']   for r in grp]):>8.3f}  "
            f"{np.mean([r['aspect_ratio'] for r in grp]):>8.2f}  "
            f"{np.mean([r['d_V']          for r in grp]):>9.4f}  "
            f"{np.mean([r['d_A']          for r in grp]):>9.4f}"
        )
    print("=" * 74)


# ---------------------------------------------------------------------------
# Build and save Plotly HTML report
# ---------------------------------------------------------------------------

def _build_report(results: list[dict], out_path: Path) -> None:
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        print("  plotly not installed — skipping HTML output")
        return

    DARK = "#1a1a2e"
    PANEL = "#16213e"
    GRID = "#2a2a4e"

    # 2 rows × 4 cols; last cell (row 2 col 4) is empty
    fig = make_subplots(
        rows=2, cols=4,
        specs=[
            [{"type": "xy"}, {"type": "xy"}, {"type": "xy"}, {"type": "xy"}],
            [{"type": "xy"}, {"type": "xy"}, {"type": "xy"}, None],
        ],
        subplot_titles=[
            "Aspect Ratio vs Sphericity",
            "Orientation Angle Distribution",
            "Equiv. Diameter from Volume (d_V)",
            "Equiv. Diameter from Surface Area (d_A)",
            "d_V vs d_A",
            "Mean Sphericity by Shape Type",
            "Aspect Ratio by Shape Type",
            "",  # placeholder for the None cell
        ],
        horizontal_spacing=0.10,
        vertical_spacing=0.18,
    )

    # ---- Panel 1: Aspect Ratio vs Sphericity ----
    for stype in SHAPE_TYPES:
        grp = [r for r in results if r["shape_type"] == stype]
        if not grp:
            continue
        fig.add_trace(
            go.Scatter(
                x=[r["aspect_ratio"] for r in grp],
                y=[r["sphericity"]   for r in grp],
                mode="markers",
                marker=dict(color=TYPE_COLORS[stype], size=9, opacity=0.85),
                name=stype,
                legendgroup=stype,
            ),
            row=1, col=1,
        )

    # ---- Panel 2: Orientation histogram ----
    fig.add_trace(
        go.Histogram(
            x=[r["orientation"] for r in results],
            nbinsx=18,
            marker_color="#4a90d9",
            opacity=0.85,
            name="Orientation",
            showlegend=False,
        ),
        row=1, col=2,
    )

    # ---- Panel 3: d_V histogram ----
    fig.add_trace(
        go.Histogram(
            x=[r["d_V"] for r in results],
            nbinsx=20,
            marker_color="#2ecc71",
            opacity=0.85,
            name="d_V",
            showlegend=False,
        ),
        row=1, col=3,
    )

    # ---- Panel 4: d_A histogram ----
    fig.add_trace(
        go.Histogram(
            x=[r["d_A"] for r in results],
            nbinsx=20,
            marker_color="#e94560",
            opacity=0.85,
            name="d_A",
            showlegend=False,
        ),
        row=1, col=4,
    )

    # ---- Panel 5: d_V vs d_A scatter ----
    for stype in SHAPE_TYPES:
        grp = [r for r in results if r["shape_type"] == stype]
        if not grp:
            continue
        fig.add_trace(
            go.Scatter(
                x=[r["d_V"] for r in grp],
                y=[r["d_A"] for r in grp],
                mode="markers",
                marker=dict(color=TYPE_COLORS[stype], size=9, opacity=0.85),
                name=stype,
                legendgroup=stype,
                showlegend=False,
            ),
            row=2, col=1,
        )
    # y = x reference line (perfect sphere)
    all_dV = [r["d_V"] for r in results]
    d_min, d_max = min(all_dV), max(all_dV)
    fig.add_trace(
        go.Scatter(
            x=[d_min, d_max], y=[d_min, d_max],
            mode="lines",
            line=dict(color="white", dash="dash", width=1),
            name="d_V = d_A  (sphere)",
            showlegend=True,
        ),
        row=2, col=1,
    )

    # ---- Panel 6: Mean sphericity bar chart ----
    bar_labels, bar_psi, bar_colors = [], [], []
    for stype in SHAPE_TYPES:
        grp = [r for r in results if r["shape_type"] == stype]
        if not grp:
            continue
        bar_labels.append(stype)
        bar_psi.append(float(np.mean([r["sphericity"] for r in grp])))
        bar_colors.append(TYPE_COLORS[stype])
    fig.add_trace(
        go.Bar(
            x=bar_labels, y=bar_psi,
            marker_color=bar_colors,
            text=[f"{v:.3f}" for v in bar_psi],
            textposition="auto",
            name="Mean ψ",
            showlegend=False,
        ),
        row=2, col=2,
    )

    # ---- Panel 7: Aspect ratio box plot ----
    for stype in SHAPE_TYPES:
        grp = [r for r in results if r["shape_type"] == stype]
        if not grp:
            continue
        fig.add_trace(
            go.Box(
                y=[r["aspect_ratio"] for r in grp],
                name=stype,
                marker_color=TYPE_COLORS[stype],
                showlegend=False,
            ),
            row=2, col=3,
        )

    # ---- Axis labels ----
    fig.update_xaxes(title_text="Aspect Ratio",          row=1, col=1)
    fig.update_yaxes(title_text="Sphericity ψ",          row=1, col=1)
    fig.update_xaxes(title_text="Angle (°)",             row=1, col=2)
    fig.update_yaxes(title_text="Count",                 row=1, col=2)
    fig.update_xaxes(title_text="d_V",                   row=1, col=3)
    fig.update_yaxes(title_text="Count",                 row=1, col=3)
    fig.update_xaxes(title_text="d_A",                   row=1, col=4)
    fig.update_yaxes(title_text="Count",                 row=1, col=4)
    fig.update_xaxes(title_text="d_V (vol. diameter)",   row=2, col=1)
    fig.update_yaxes(title_text="d_A (area diameter)",   row=2, col=1)
    fig.update_yaxes(title_text="Mean Sphericity ψ",     row=2, col=2)
    fig.update_yaxes(title_text="Aspect Ratio",          row=2, col=3)

    fig.update_layout(
        title=dict(
            text=(
                "Morphometric Statistical Analysis — "
                f"Synthetic SDF Population (N={len(results)})"
            ),
            font=dict(size=18),
        ),
        width=1500,
        height=850,
        paper_bgcolor=DARK,
        plot_bgcolor=PANEL,
        font=dict(color="#e0e0e0"),
        legend=dict(
            bgcolor="#1a1a2e",
            bordercolor="#3a3a5e",
            borderwidth=1,
        ),
    )
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=GRID)
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=GRID)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_path), include_plotlyjs="cdn")
    print(f"\n  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Morphometric statistical analysis of synthetic SDF shapes"
    )
    parser.add_argument(
        "--res", type=int, default=48,
        help="Grid resolution per axis (default: 48)",
    )
    parser.add_argument(
        "--n", type=int, default=50,
        help="Population size — number of random shapes (default: 50)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--out", type=str, default=None,
        help="Output HTML path (default: examples/sdf3d/output/morphometry_stats_demo.html)",
    )
    args = parser.parse_args()

    out_html = (
        Path(args.out) if args.out else
        _OUTPUT_DIR / "morphometry_stats_demo.html"
    )

    print(
        f"Morphometric Statistical Analysis Demo — pySdf\n"
        f"Population: N={args.n}, res={args.res}³, seed={args.seed}\n"
    )

    t0 = time.perf_counter()
    results = _build_population(args.n, args.res, seed=args.seed)
    elapsed = time.perf_counter() - t0
    print(f"\nComputed {args.n} shapes in {elapsed:.1f}s")

    _print_summary(results)
    _build_report(results, out_html)
    print("\nDone.")


if __name__ == "__main__":
    main()
