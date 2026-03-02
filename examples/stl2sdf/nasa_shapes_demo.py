"""nasa_shapes_demo.py — Multi-shape NASA STL → SDF demo.

Processes NASA STL meshes found in the same directory and computes their
signed distance fields.  Results are written as .npy files and combined
into a single Plotly HTML report.  Missing STL files are skipped silently.

Shapes (ordered by increasing triangle count)
--------------------------------------------
1. Orion Capsule plug    (~2 K triangles)   — small smooth capsule piece
2. Mars Rover Wheel      (~45 K triangles)  — Curiosity rover wheel tread
3. CubeSat bottom        (~5 K triangles)   — boxy satellite component
4. ISS Ratchet Wrench    (~7 K triangles)   — elongated tool with hex heads
5. Asteroid 433 Eros     (~200 K triangles) — irregular rocky body

Usage
-----
python examples/nasa_shapes_demo.py           # res=20 for all shapes
python examples/nasa_shapes_demo.py --res 30  # higher quality (much slower for Eros)
python examples/nasa_shapes_demo.py --skip-eros  # skip the slow 200K-tri mesh
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Shape catalogue
# ---------------------------------------------------------------------------
SHAPES = [
    {
        "name":  "Orion Capsule (plug)",
        "stem":  "orion_plug",
        "slow":  False,
    },
    {
        "name":  "Mars Rover Wheel",
        "stem":  "mars_wheel",
        "slow":  False,
    },
    {
        "name":  "CubeSat (bottom)",
        "stem":  "cubesat_bottom",
        "slow":  False,
    },
    {
        "name":  "ISS Ratchet Wrench",
        "stem":  "iss_wrench",
        "slow":  False,
    },
    {
        "name":  "Asteroid 433 Eros",
        "stem":  "eros",
        "slow":  True,   # ~200K triangles — flag so --skip-eros works
    },
]

_EXAMPLES_DIR = Path(__file__).parent
_OUTPUT_DIR   = _EXAMPLES_DIR / "output"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _auto_bounds(triangles: np.ndarray, pad_frac: float = 0.10):
    """Return ((x0,x1),(y0,y1),(z0,z1)) with fractional padding."""
    verts = triangles.reshape(-1, 3)
    lo, hi = verts.min(axis=0), verts.max(axis=0)
    pad    = pad_frac * (hi - lo)
    lo    -= pad
    hi    += pad
    return tuple(zip(lo.tolist(), hi.tolist()))


def _process_shape(shape: dict, res: int) -> dict | None:
    """Compute SDF for a shape whose STL is already on disk; skip if absent."""
    from stl2sdf import stl_to_geometry
    from stl2sdf._math import _stl_to_triangles
    from sdf3d.grid import sample_levelset_3d

    stl_path = _EXAMPLES_DIR / f"{shape['stem']}.stl"
    npy_path = _OUTPUT_DIR / f"{shape['stem']}_sdf.npy"

    print(f"\n{'='*60}", flush=True)
    print(f"  {shape['name']}", flush=True)
    print(f"{'='*60}", flush=True)

    if not stl_path.exists():
        print(f"  SKIP: {stl_path.name} not found in examples/stl2sdf/", flush=True)
        return None

    # --- inspect ---
    triangles = _stl_to_triangles(stl_path)
    n_tri  = len(triangles)
    bounds = _auto_bounds(triangles)
    print(f"  Triangles : {n_tri:>10,}", flush=True)
    print(f"  Bounds    : {bounds}", flush=True)
    print(f"  Grid      : {res}^3 = {res**3:,} points", flush=True)
    est_ops = n_tri * res**3
    print(f"  Est. ops  : ~{est_ops/1e9:.1f}B  (O(FxN))", flush=True)

    # --- SDF ---
    geom = stl_to_geometry(stl_path)
    t0   = time.perf_counter()
    phi  = sample_levelset_3d(geom, bounds, (res, res, res))
    elapsed = time.perf_counter() - t0

    inside_frac = (phi < 0).mean() * 100
    print(f"  Done in {elapsed:.1f} s", flush=True)
    print(f"  phi min={phi.min():.4f}  max={phi.max():.4f}", flush=True)
    print(f"  Inside fraction: {inside_frac:.1f}%", flush=True)

    np.save(npy_path, phi)
    print(f"  Saved: {npy_path}", flush=True)

    return {
        "name":   shape["name"],
        "phi":    phi,
        "bounds": bounds,
        "n_tri":  n_tri,
        "res":    res,
        "time_s": elapsed,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _build_report(results: list[dict], out_html: Path) -> None:
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        print("\nplotly not installed; skipping HTML report (uv sync --extra viz)",
              file=sys.stderr)
        return

    n = len(results)
    fig = make_subplots(
        rows=2, cols=n,
        specs=[[{"type": "xy"}] * n, [{"type": "scene"}] * n],
        subplot_titles=[
            *[f"{r['name']}<br>{r['n_tri']:,} tris | {r['res']}³ | {r['time_s']:.0f}s"
              for r in results],
            *["φ = 0 surface"] * n,
        ],
        vertical_spacing=0.08,
        horizontal_spacing=0.05,
    )

    for col, r in enumerate(results, start=1):
        phi    = r["phi"]
        bounds = r["bounds"]
        res    = r["res"]
        (x0, x1), (y0, y1), (z0, z1) = bounds

        xs = np.linspace(x0, x1, res, endpoint=False) + (x1 - x0) / (2.0 * res)
        ys = np.linspace(y0, y1, res, endpoint=False) + (y1 - y0) / (2.0 * res)
        zs = np.linspace(z0, z1, res, endpoint=False) + (z1 - z0) / (2.0 * res)
        Z3, Y3, X3 = np.meshgrid(zs, ys, xs, indexing="ij")

        # Row 1: mid-Z heatmap
        mid = phi[res // 2]
        clim = float(np.abs(mid).max())
        fig.add_trace(
            go.Heatmap(
                z=mid, x=xs, y=ys,
                colorscale="RdBu", reversescale=True,
                zmid=0.0, zmin=-clim, zmax=clim,
                showscale=(col == 1),
                colorbar=dict(title=dict(text="φ"), x=-0.02) if col == 1 else None,
            ),
            row=1, col=col,
        )
        fig.update_xaxes(scaleanchor=f"y{col if col > 1 else ''}", row=1, col=col)

        # Row 2: 3-D isosurface
        fig.add_trace(
            go.Isosurface(
                x=X3.ravel(), y=Y3.ravel(), z=Z3.ravel(),
                value=phi.ravel(),
                isomin=0.0, isomax=0.0, surface_count=1,
                colorscale=[[0, "#4a90d9"], [1, "#4a90d9"]],
                showscale=False,
                caps=dict(x_show=False, y_show=False, z_show=False),
                lighting=dict(ambient=0.6, diffuse=0.8, specular=0.3),
            ),
            row=2, col=col,
        )

    fig.update_layout(
        title=dict(text="NASA Shape Gallery — Signed Distance Fields", font=dict(size=18)),
        width=380 * n,
        height=900,
        paper_bgcolor="#1a1a2e",
        plot_bgcolor="#1a1a2e",
        font=dict(color="#e0e0e0"),
    )

    fig.write_html(out_html, include_plotlyjs="cdn")
    print(f"\nSaved interactive report: {out_html}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="NASA multi-shape STL → SDF demo")
    parser.add_argument("--res", type=int, default=20,
                        help="Grid resolution per axis (default 20; increase for quality)")
    parser.add_argument("--skip-eros", action="store_true",
                        help="Skip Asteroid 433 Eros (~200K triangles, slowest shape)")
    parser.add_argument("--out", type=Path,
                        default=_OUTPUT_DIR / "nasa_shapes_report.html",
                        help="Output HTML report path")
    args = parser.parse_args()

    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    shapes = [s for s in SHAPES if not (s["slow"] and args.skip_eros)]

    print(f"NASA SDF Demo — {len(shapes)} shape(s) at res={args.res}")
    print(f"  (use --skip-eros to skip the 200K-tri Eros mesh)")

    results = []
    for shape in shapes:
        result = _process_shape(shape, args.res)
        if result is not None:
            results.append(result)

    if results:
        _build_report(results, args.out)

    print(f"\nAll done.  {len(results)}/{len(shapes)} shape(s) succeeded.")


if __name__ == "__main__":
    main()
