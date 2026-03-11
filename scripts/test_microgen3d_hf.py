"""test_microgen3d_hf.py — Download & evaluate microgen3D STL meshes via pySdf.

Downloads STL files from the BGLab/microgen3D dataset on HuggingFace Hub,
converts each mesh to an SDF using :func:`stl2sdf.stl_to_geometry`, evaluates
it on a uniform grid, and saves a combined Plotly HTML report.

Prerequisites
-------------
Install the HuggingFace Hub client::

    pip install huggingface-hub>=0.20
    # or: pip install sdf-library[hf]

Usage
-----
python scripts/test_microgen3d_hf.py                     # auto-detect token
python scripts/test_microgen3d_hf.py --res 40            # grid resolution
python scripts/test_microgen3d_hf.py --max-shapes 5      # limit to 5 meshes
python scripts/test_microgen3d_hf.py --token YOUR_TOKEN  # explicit HF token

Environment variable ``HF_TOKEN`` is also honoured if set.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

_SCRIPT_DIR = Path(__file__).parent
_OUTPUT_DIR = _SCRIPT_DIR / "output"
_HF_DATASET = "BGLab/microgen3D"
_STL_PATTERN = "*.stl"

# ---------------------------------------------------------------------------
# Dependency check
# ---------------------------------------------------------------------------

def _require_huggingface_hub() -> object:
    try:
        from huggingface_hub import snapshot_download, list_repo_files
        return snapshot_download, list_repo_files
    except ImportError:
        print(
            "\nERROR: 'huggingface_hub' is not installed.\n"
            "Install it with one of:\n"
            "  pip install huggingface-hub>=0.20\n"
            "  pip install sdf-library[hf]\n",
            file=sys.stderr,
        )
        sys.exit(1)


# ---------------------------------------------------------------------------
# Token handling
# ---------------------------------------------------------------------------

def _get_token(cli_token: str | None) -> str | None:
    """Return the HF token from CLI arg, env var, or interactive prompt."""
    if cli_token:
        return cli_token
    env = os.environ.get("HF_TOKEN", "").strip()
    if env:
        return env
    # Interactive prompt
    try:
        tok = input(
            "Enter your HuggingFace token (or press Enter to skip): "
        ).strip()
        return tok if tok else None
    except (EOFError, KeyboardInterrupt):
        return None


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------

def _download_stl_files(
    token: str | None,
    max_shapes: int | None,
    cache_dir: Path,
) -> list[Path]:
    """Download STL files from the microgen3D dataset and return local paths."""
    snapshot_download, list_repo_files = _require_huggingface_hub()

    print(f"Listing files in {_HF_DATASET}…", flush=True)
    try:
        all_files = list(list_repo_files(
            _HF_DATASET,
            repo_type="dataset",
            token=token,
        ))
    except Exception as exc:
        print(f"ERROR listing repo files: {exc}", file=sys.stderr)
        sys.exit(1)

    stl_files = [f for f in all_files if f.lower().endswith(".stl")]
    if not stl_files:
        print("No STL files found in the dataset.", file=sys.stderr)
        sys.exit(1)

    if max_shapes is not None:
        stl_files = stl_files[:max_shapes]

    print(f"Found {len(stl_files)} STL file(s) to download.", flush=True)

    from huggingface_hub import hf_hub_download

    local_paths: list[Path] = []
    for fname in stl_files:
        print(f"  Downloading: {fname}", flush=True)
        try:
            local = hf_hub_download(
                repo_id=_HF_DATASET,
                filename=fname,
                repo_type="dataset",
                token=token,
                cache_dir=str(cache_dir),
            )
            local_paths.append(Path(local))
        except Exception as exc:
            print(f"  WARNING: could not download {fname}: {exc}")

    return local_paths


# ---------------------------------------------------------------------------
# Process a single STL
# ---------------------------------------------------------------------------

def _process_stl(stl_path: Path, res: int) -> dict | None:
    try:
        from stl2sdf import stl_to_geometry
        from stl2sdf._math import _stl_to_triangles
    except ImportError as exc:
        print(f"  ERROR importing stl2sdf: {exc}", file=sys.stderr)
        return None

    print(f"\n  Processing: {stl_path.name}", flush=True)

    try:
        triangles = _stl_to_triangles(stl_path)
    except Exception as exc:
        print(f"    ERROR loading STL: {exc}")
        return None

    n_tri = len(triangles)
    verts = triangles.reshape(-1, 3)
    lo, hi = verts.min(axis=0), verts.max(axis=0)
    pad = 0.10 * (hi - lo)
    lo -= pad
    hi += pad
    bounds = tuple(zip(lo.tolist(), hi.tolist()))

    print(f"    Triangles : {n_tri:>8,}", flush=True)
    print(f"    Bounds    : {bounds}", flush=True)

    try:
        geom = stl_to_geometry(stl_path)
    except Exception as exc:
        print(f"    ERROR building SDF: {exc}")
        return None

    t0 = time.perf_counter()
    try:
        phi = geom.to_numpy(bounds, (res, res, res))
    except Exception as exc:
        print(f"    ERROR evaluating SDF: {exc}")
        return None
    elapsed = time.perf_counter() - t0

    inside_frac = (phi < 0).mean() * 100
    print(f"    Time      : {elapsed:.1f}s", flush=True)
    print(f"    phi       : min={phi.min():.4f}  max={phi.max():.4f}", flush=True)
    print(f"    Inside    : {inside_frac:.1f}%", flush=True)

    return {
        "name":        stl_path.stem,
        "path":        stl_path,
        "n_tri":       n_tri,
        "bounds":      bounds,
        "phi":         phi,
        "res":         res,
        "time_s":      elapsed,
        "inside_frac": inside_frac,
    }


# ---------------------------------------------------------------------------
# Build HTML report
# ---------------------------------------------------------------------------

def _build_report(results: list[dict], out_html: Path) -> None:
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        print("  plotly not installed — skipping HTML output")
        return

    n = len(results)
    if n == 0:
        print("  No results to report.")
        return

    cols = min(n, 3)
    rows = (n + cols - 1) // cols
    specs = [[{"type": "scene"}] * cols for _ in range(rows)]
    titles = [r["name"] for r in results]
    # Pad titles to fill the grid
    while len(titles) < rows * cols:
        titles.append("")

    fig = make_subplots(
        rows=rows, cols=cols,
        specs=specs,
        subplot_titles=titles,
        horizontal_spacing=0.05,
        vertical_spacing=0.08,
    )

    colors = [
        "#4a90d9", "#e94560", "#2ecc71", "#f39c12",
        "#9b59b6", "#1abc9c", "#e67e22", "#3498db",
        "#e74c3c", "#2980b9",
    ]

    for idx, result in enumerate(results):
        row = idx // cols + 1
        col = idx % cols + 1
        color = colors[idx % len(colors)]

        phi = result["phi"]
        bounds = result["bounds"]
        res = result["res"]

        (x0, x1), (y0, y1), (z0, z1) = bounds
        xs = np.linspace(x0, x1, res)
        ys = np.linspace(y0, y1, res)
        zs = np.linspace(z0, z1, res)
        Z3, Y3, X3 = np.meshgrid(zs, ys, xs, indexing="ij")

        if phi.min() >= 0:
            print(f"  WARNING: {result['name']} has no interior (phi ≥ 0 everywhere)")

        fig.add_trace(
            go.Isosurface(
                x=X3.ravel(), y=Y3.ravel(), z=Z3.ravel(),
                value=phi.ravel(),
                isomin=0.0, isomax=0.0, surface_count=1,
                colorscale=[[0, color], [1, color]],
                showscale=False,
                caps=dict(x_show=False, y_show=False, z_show=False),
                lighting=dict(ambient=0.4, diffuse=0.9, specular=0.4),
                lightposition=dict(x=1000, y=1000, z=2000),
                name=result["name"],
            ),
            row=row, col=col,
        )

    # Statistics table placed in the last scene cell
    rows_data = [
        [r["name"] for r in results],
        [f"{r['n_tri']:,}" for r in results],
        [f"{r['inside_frac']:.1f}%" for r in results],
        [f"{r['time_s']:.1f}s" for r in results],
    ]
    fig.add_trace(
        go.Table(
            header=dict(
                values=["<b>Name</b>", "<b>Triangles</b>",
                        "<b>Inside %</b>", "<b>SDF Time</b>"],
                fill_color="#16213e",
                font=dict(color="#e0e0e0", size=12),
                align="left",
            ),
            cells=dict(
                values=rows_data,
                fill_color="#0f3460",
                font=dict(color="#e0e0e0", size=11),
                align="left",
            ),
        ),
        row=rows,
        col=cols,
    )

    fig.update_scenes(aspectmode="data")
    fig.update_layout(
        title=dict(
            text=f"microgen3D Dataset — STL → SDF Gallery ({n} shapes)",
            font=dict(size=20),
        ),
        width=max(900, 450 * cols),
        height=600 * rows + 150,
        paper_bgcolor="#1a1a2e",
        plot_bgcolor="#16213e",
        font=dict(color="#e0e0e0"),
    )

    out_html.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_html), include_plotlyjs="cdn")
    print(f"\n  Report saved: {out_html}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download microgen3D STL files and test pySdf stl2sdf"
    )
    parser.add_argument(
        "--res", type=int, default=40,
        help="SDF grid resolution per axis (default: 40)",
    )
    parser.add_argument(
        "--max-shapes", type=int, default=None,
        help="Maximum number of STL files to process (default: all)",
    )
    parser.add_argument(
        "--token", type=str, default=None,
        help="HuggingFace API token (overrides HF_TOKEN env var and prompt)",
    )
    parser.add_argument(
        "--cache-dir", type=str, default=str(_OUTPUT_DIR / "hf_cache"),
        help="Directory for HuggingFace cache (default: scripts/output/hf_cache)",
    )
    args = parser.parse_args()

    # Ensure huggingface_hub is available early
    _require_huggingface_hub()

    token = _get_token(args.token)
    cache_dir = Path(args.cache_dir)

    print(f"\n{'='*60}")
    print(f"  microgen3D STL → SDF Evaluation")
    print(f"  Dataset : {_HF_DATASET}")
    print(f"  Grid    : {args.res}^3 = {args.res**3:,} points")
    if args.max_shapes:
        print(f"  Limit   : {args.max_shapes} shapes")
    print(f"{'='*60}\n")

    # Download
    stl_paths = _download_stl_files(token, args.max_shapes, cache_dir)
    if not stl_paths:
        print("No STL files downloaded.", file=sys.stderr)
        sys.exit(1)

    # Process
    results = []
    for stl_path in stl_paths:
        result = _process_stl(stl_path, args.res)
        if result is not None:
            results.append(result)

    if not results:
        print("No shapes could be processed.", file=sys.stderr)
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"  Processed {len(results)}/{len(stl_paths)} shapes successfully")
    total_time = sum(r["time_s"] for r in results)
    print(f"  Total SDF time: {total_time:.1f}s")
    print(f"{'='*60}\n")

    # HTML report
    out_html = _OUTPUT_DIR / "microgen3d_report.html"
    _build_report(results, out_html)


if __name__ == "__main__":
    main()
