"""test_microgen3d_hf.py — Download & evaluate microgen3D voxel microstructures.

The BGLab/microgen3D dataset (https://huggingface.co/datasets/BGLab/microgen3D)
contains 3D **voxelized** microstructures stored in HDF5 files inside .tar.gz
archives.  The three dataset families are:

* ``data/sample_CH_two_phase.tar.gz``   (~9.6 MB)  — 2-phase Cahn–Hilliard sample
* ``data/sample_CH_three_phase.tar.gz`` (~12.2 MB) — 3-phase Cahn–Hilliard sample
* ``data/experimental.tar.gz``          (~843 MB)  — experimental microstructures

This script:
1. Downloads the two **sample** archives (small, suitable for testing).
2. Extracts them to a local cache directory.
3. Reads 3D voxel/phase arrays from the HDF5 files inside.
4. Converts each binary/phase mask to a Signed Distance Field (SDF) using
   ``scipy.ndimage.distance_transform_edt``.
5. Reports statistics (volume fraction, voxel shape, SDF min/max, timing).
6. Saves an interactive Plotly HTML gallery report.

Prerequisites
-------------
::

    pip install huggingface-hub>=0.20 h5py scipy
    # or: pip install sdf-library[hf]

Usage
-----
::

    python scripts/test_microgen3d_hf.py                     # default settings
    python scripts/test_microgen3d_hf.py --max-samples 3     # 3 microstructures per file
    python scripts/test_microgen3d_hf.py --token YOUR_TOKEN  # explicit HF token
    python scripts/test_microgen3d_hf.py --no-sample         # use full datasets (large!)

Environment variable ``HF_TOKEN`` is also honoured if set.
"""

from __future__ import annotations

import argparse
import io
import os
import sys
import tarfile
import time
from pathlib import Path
from typing import Iterator

import numpy as np

# ---------------------------------------------------------------------------
# Path setup — repo root on sys.path so pySdf modules are importable
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

_SCRIPT_DIR = Path(__file__).parent
_OUTPUT_DIR = _SCRIPT_DIR / "output"
_HF_DATASET = "BGLab/microgen3D"

# ---------------------------------------------------------------------------
# Sample archive catalogue
# (full archives are multi-GB; default to samples for quick testing)
# ---------------------------------------------------------------------------
_SAMPLE_ARCHIVES = [
    {
        "hf_path":   "data/sample_CH_two_phase.tar.gz",
        "label":     "CH 2-Phase (sample)",
        "n_phases":  2,
    },
    {
        "hf_path":   "data/sample_CH_three_phase.tar.gz",
        "label":     "CH 3-Phase (sample)",
        "n_phases":  3,
    },
]

_FULL_ARCHIVES = [
    {
        "hf_path":   "data/CH_two_phase.tar.gz",
        "label":     "CH 2-Phase (full)",
        "n_phases":  2,
    },
    {
        "hf_path":   "data/CH_three_phase.tar.gz",
        "label":     "CH 3-Phase (full)",
        "n_phases":  3,
    },
    {
        "hf_path":   "data/experimental.tar.gz",
        "label":     "Experimental (full)",
        "n_phases":  2,
    },
]


# ===========================================================================
# Dependency helpers
# ===========================================================================

def _require_huggingface_hub() -> None:
    try:
        import huggingface_hub  # noqa: F401
    except ImportError:
        print(
            "\nERROR: 'huggingface_hub' is not installed.\n"
            "Install it with one of:\n"
            "  pip install huggingface-hub>=0.20\n"
            "  pip install sdf-library[hf]\n",
            file=sys.stderr,
        )
        sys.exit(1)


def _require_h5py() -> None:
    try:
        import h5py  # noqa: F401
    except ImportError:
        print(
            "\nERROR: 'h5py' is not installed.\n"
            "Install it with: pip install h5py\n",
            file=sys.stderr,
        )
        sys.exit(1)


def _require_scipy() -> None:
    try:
        from scipy.ndimage import distance_transform_edt  # noqa: F401
    except ImportError:
        print(
            "\nERROR: 'scipy' is not installed.\n"
            "Install it with: pip install scipy\n",
            file=sys.stderr,
        )
        sys.exit(1)


# ===========================================================================
# Token handling
# ===========================================================================

def _get_token(cli_token: str | None) -> str | None:
    """Return HF token from CLI arg, env var, or interactive prompt."""
    if cli_token:
        return cli_token
    env = os.environ.get("HF_TOKEN", "").strip()
    if env:
        return env
    try:
        tok = input(
            "Enter your HuggingFace token (or press Enter to try anonymously): "
        ).strip()
        return tok if tok else None
    except (EOFError, KeyboardInterrupt):
        return None


# ===========================================================================
# Download and extraction
# ===========================================================================

def _download_archive(hf_path: str, token: str | None, cache_dir: Path) -> Path:
    """Download one archive from the HF dataset hub and return its local path."""
    from huggingface_hub import hf_hub_download

    local = hf_hub_download(
        repo_id=_HF_DATASET,
        filename=hf_path,
        repo_type="dataset",
        token=token,
        cache_dir=str(cache_dir),
    )
    return Path(local)


def _extract_archive(archive_path: Path, dest_dir: Path) -> list[Path]:
    """Extract *archive_path* (a .tar.gz) into *dest_dir*.

    Returns a sorted list of all extracted `.h5` / `.hdf5` files.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Extracting {archive_path.name} → {dest_dir} …", flush=True)
    with tarfile.open(archive_path, "r:gz") as tf:
        tf.extractall(dest_dir)

    h5_files = sorted(
        p for p in dest_dir.rglob("*.h5")
    ) + sorted(
        p for p in dest_dir.rglob("*.hdf5")
    )
    print(f"    Found {len(h5_files)} HDF5 file(s) after extraction.", flush=True)
    return h5_files


# ===========================================================================
# HDF5 reading
# ===========================================================================

def _iter_voxel_arrays(h5_path: Path, max_samples: int) -> Iterator[np.ndarray]:
    """Yield up to *max_samples* 3D voxel arrays from *h5_path*.

    The microgen3D HDF5 files contain microstructures stored as datasets.
    This function tries several common layout patterns:

    1. A single dataset ``"data"`` with shape ``(N, D, H, W)`` or ``(D, H, W)``
    2. Integer-keyed datasets ``"0"``, ``"1"``, … each with shape ``(D, H, W)``
    3. Any dataset whose shape is 3-D

    Parameters
    ----------
    h5_path:
        Path to a ``.h5`` file.
    max_samples:
        Maximum number of microstructure arrays to yield.
    """
    import h5py

    yielded = 0

    def _try_yield(arr: np.ndarray) -> bool:
        nonlocal yielded
        arr = np.asarray(arr)
        if arr.ndim == 3 and arr.size > 0:
            yield_arr = arr.astype(np.float32)
            return yield_arr
        return None

    with h5py.File(h5_path, "r") as f:
        all_keys = list(f.keys())

        # Pattern 1 — "data" key (batch)
        for key in ("data", "voxel", "microstructure", "images"):
            if key in f:
                arr = np.asarray(f[key])
                if arr.ndim == 4:
                    for i in range(min(max_samples, arr.shape[0])):
                        yield arr[i].astype(np.float32)
                        yielded += 1
                        if yielded >= max_samples:
                            return
                elif arr.ndim == 3:
                    yield arr.astype(np.float32)
                    yielded += 1
                    if yielded >= max_samples:
                        return

        if yielded >= max_samples:
            return

        # Pattern 2 — integer keys
        int_keys = sorted(
            k for k in all_keys if k.isdigit()
        )
        for key in int_keys:
            arr = np.asarray(f[key])
            if arr.ndim == 3:
                yield arr.astype(np.float32)
                yielded += 1
                if yielded >= max_samples:
                    return

        if yielded >= max_samples:
            return

        # Pattern 3 — any 3-D dataset (catch-all)
        for key in all_keys:
            if key in ("data", "voxel", "microstructure", "images"):
                continue
            arr = np.asarray(f[key])
            if arr.ndim == 4:
                for i in range(min(max_samples - yielded, arr.shape[0])):
                    yield arr[i].astype(np.float32)
                    yielded += 1
                    if yielded >= max_samples:
                        return
            elif arr.ndim == 3:
                yield arr.astype(np.float32)
                yielded += 1
                if yielded >= max_samples:
                    return

        if yielded == 0:
            print(f"    WARNING: no 3-D arrays found in {h5_path.name}. "
                  f"Keys: {all_keys}", file=sys.stderr)


# ===========================================================================
# Voxel → SDF conversion
# ===========================================================================

def _phase_to_sdf(voxels: np.ndarray, phase_id: int) -> np.ndarray:
    """Convert a phase array to a signed distance field.

    pySdf sign convention: ``phi < 0`` inside (where ``voxels == phase_id``),
    ``phi > 0`` outside.

    Parameters
    ----------
    voxels:
        3-D integer/float array of phase labels.
    phase_id:
        The phase label to treat as "inside" (solid).

    Returns
    -------
    numpy.ndarray
        Shape ``(D, H, W)``, dtype float64.  phi = dist_outside - dist_inside.
    """
    from scipy.ndimage import distance_transform_edt

    mask = (np.round(voxels).astype(np.int32) == int(phase_id))
    if not mask.any():
        return np.full(voxels.shape, np.inf, dtype=np.float64)
    if mask.all():
        return np.full(voxels.shape, -np.inf, dtype=np.float64)

    dist_inside  = distance_transform_edt(mask)        # distance from surface, inside mask
    dist_outside = distance_transform_edt(~mask)       # distance from surface, outside mask
    return (dist_outside - dist_inside).astype(np.float64)  # phi < 0 inside


# ===========================================================================
# Process one microstructure
# ===========================================================================

def _process_microstructure(
    voxels: np.ndarray,
    name: str,
    n_phases: int,
) -> list[dict]:
    """Convert a 3D voxel array to per-phase SDF results.

    Parameters
    ----------
    voxels:
        3-D phase array, shape (D, H, W).
    name:
        Display name for the microstructure.
    n_phases:
        Expected number of phases (1–3).

    Returns
    -------
    list of dict
        One dict per phase containing ``name``, ``phi``, ``shape``,
        ``volume_fraction``, ``phase_id``, ``time_s``.
    """
    unique_phases = np.unique(np.round(voxels).astype(np.int32))
    D, H, W = voxels.shape

    results = []
    for pid in unique_phases[:n_phases]:
        t0 = time.perf_counter()
        phi = _phase_to_sdf(voxels, int(pid))
        elapsed = time.perf_counter() - t0

        vol_frac = float((np.round(voxels).astype(np.int32) == pid).mean()) * 100
        results.append({
            "name":            f"{name} · phase {pid}",
            "phi":             phi,
            "shape":           (D, H, W),
            "volume_fraction": vol_frac,
            "phase_id":        int(pid),
            "time_s":          elapsed,
        })
        print(
            f"    phase {pid}: vol={vol_frac:.1f}%  "
            f"phi∈[{phi.min():.1f}, {phi.max():.1f}]  {elapsed:.2f}s",
            flush=True,
        )

    return results


# ===========================================================================
# HTML report
# ===========================================================================

def _build_report(all_results: list[dict], out_html: Path) -> None:
    """Build an interactive Plotly HTML gallery from processed microstructures."""
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        print("  plotly not installed — skipping HTML output", file=sys.stderr)
        return

    n = len(all_results)
    if n == 0:
        print("  No results to include in report.", file=sys.stderr)
        return

    cols = min(n, 3)
    rows_scene = (n + cols - 1) // cols

    # Each scene cell gets a "scene" spec; a final row holds the stats table
    specs_scenes = [[{"type": "scene"}] * cols for _ in range(rows_scene)]
    specs_table  = [[{"type": "table",  "colspan": cols}] + [None] * (cols - 1)]
    all_specs = specs_scenes + specs_table

    subplot_titles = [r["name"] for r in all_results]
    while len(subplot_titles) < rows_scene * cols:
        subplot_titles.append("")
    subplot_titles.append("")   # table row title

    fig = make_subplots(
        rows=rows_scene + 1,
        cols=cols,
        specs=all_specs,
        subplot_titles=subplot_titles,
        horizontal_spacing=0.04,
        vertical_spacing=0.06,
    )

    colors = [
        "#4a90d9", "#e94560", "#2ecc71", "#f39c12",
        "#9b59b6", "#1abc9c", "#e67e22", "#3498db",
        "#e74c3c", "#2980b9", "#16a085", "#8e44ad",
    ]

    for idx, result in enumerate(all_results):
        row = idx // cols + 1
        col = idx % cols + 1
        color = colors[idx % len(colors)]

        phi = result["phi"]
        D, H, W = result["shape"]

        xs = np.arange(W, dtype=float)
        ys = np.arange(H, dtype=float)
        zs = np.arange(D, dtype=float)
        Z3, Y3, X3 = np.meshgrid(zs, ys, xs, indexing="ij")

        # Guard: skip panels with no zero-crossing
        if phi.min() >= 0 or phi.max() <= 0:
            print(f"  SKIP isosurface for '{result['name']}': "
                  f"no zero-crossing (phi ∈ [{phi.min():.2f}, {phi.max():.2f}])")
            continue

        fig.add_trace(
            go.Isosurface(
                x=X3.ravel(), y=Y3.ravel(), z=Z3.ravel(),
                value=phi.ravel(),
                isomin=0.0, isomax=0.0, surface_count=1,
                colorscale=[[0, color], [1, color]],
                showscale=False,
                caps=dict(x_show=False, y_show=False, z_show=False),
                lighting=dict(ambient=0.4, diffuse=0.9, specular=0.3,
                              roughness=0.5),
                lightposition=dict(x=1000, y=1000, z=2000),
                name=result["name"],
            ),
            row=row, col=col,
        )

    # Statistics table
    fig.add_trace(
        go.Table(
            header=dict(
                values=[
                    "<b>Microstructure</b>", "<b>Shape (D×H×W)</b>",
                    "<b>Phase</b>", "<b>Vol %</b>", "<b>SDF time</b>",
                ],
                fill_color="#0f3460",
                font=dict(color="#e0e0e0", size=12),
                align="left",
            ),
            cells=dict(
                values=[
                    [r["name"]              for r in all_results],
                    ["×".join(str(s) for s in r["shape"]) for r in all_results],
                    [str(r["phase_id"])     for r in all_results],
                    [f"{r['volume_fraction']:.1f}%" for r in all_results],
                    [f"{r['time_s']:.2f}s"  for r in all_results],
                ],
                fill_color="#16213e",
                font=dict(color="#e0e0e0", size=11),
                align="left",
            ),
        ),
        row=rows_scene + 1, col=1,
    )

    fig.update_scenes(
        aspectmode="data",
        xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
        yaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
        zaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
    )
    fig.update_layout(
        title=dict(
            text=f"microgen3D — Voxel → SDF Gallery ({n} microstructures)",
            font=dict(size=20),
        ),
        width=max(900, 400 * cols),
        height=500 * rows_scene + 350,
        paper_bgcolor="#1a1a2e",
        plot_bgcolor="#16213e",
        font=dict(color="#e0e0e0"),
    )

    out_html.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_html), include_plotlyjs="cdn")
    print(f"\n  Report saved: {out_html}")


# ===========================================================================
# Main
# ===========================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Download BGLab/microgen3D sample archives from HuggingFace, "
            "convert 3D voxel microstructures to SDF, and save an HTML report."
        )
    )
    parser.add_argument(
        "--max-samples", type=int, default=2,
        help="Max number of microstructures to process per HDF5 file (default: 2)",
    )
    parser.add_argument(
        "--token", type=str, default=None,
        help="HuggingFace API token (overrides HF_TOKEN env var and prompt)",
    )
    parser.add_argument(
        "--cache-dir", type=str, default=str(_OUTPUT_DIR / "hf_cache"),
        help="Directory for HuggingFace downloads (default: scripts/output/hf_cache)",
    )
    parser.add_argument(
        "--no-sample", action="store_true",
        help="Download the full (multi-GB) archives instead of the sample subsets",
    )
    args = parser.parse_args()

    _require_huggingface_hub()
    _require_h5py()
    _require_scipy()

    token    = _get_token(args.token)
    cache_dir = Path(args.cache_dir)
    archives  = _FULL_ARCHIVES if args.no_sample else _SAMPLE_ARCHIVES

    print(f"\n{'='*60}")
    print(f"  microgen3D — Voxel Microstructure → SDF")
    print(f"  Dataset    : {_HF_DATASET}")
    print(f"  Archives   : {[a['hf_path'] for a in archives]}")
    print(f"  Max samples: {args.max_samples} per HDF5 file")
    print(f"{'='*60}\n")

    all_results: list[dict] = []

    for archive_info in archives:
        hf_path   = archive_info["hf_path"]
        label     = archive_info["label"]
        n_phases  = archive_info["n_phases"]

        print(f"\n{'─'*60}")
        print(f"  Archive: {hf_path}  [{label}]")
        print(f"{'─'*60}")

        # 1. Download
        print(f"  Downloading from HuggingFace…", flush=True)
        try:
            local_archive = _download_archive(hf_path, token, cache_dir)
        except Exception as exc:
            print(f"  ERROR downloading {hf_path}: {exc}", file=sys.stderr)
            continue
        print(f"  Downloaded: {local_archive}")

        # 2. Extract
        extract_dir = cache_dir / Path(hf_path).stem
        try:
            h5_files = _extract_archive(local_archive, extract_dir)
        except Exception as exc:
            print(f"  ERROR extracting {local_archive.name}: {exc}", file=sys.stderr)
            continue

        if not h5_files:
            print(f"  WARNING: no HDF5 files found after extracting {local_archive.name}")
            continue

        # 3. Process each HDF5 file
        for h5_path in h5_files:
            print(f"\n  HDF5 file: {h5_path.name}", flush=True)
            sample_idx = 0
            for voxels in _iter_voxel_arrays(h5_path, max_samples=args.max_samples):
                sample_idx += 1
                D, H, W = voxels.shape
                unique = np.unique(np.round(voxels).astype(np.int32))
                print(
                    f"    Sample {sample_idx}: shape=({D},{H},{W})  "
                    f"phases={unique.tolist()}",
                    flush=True,
                )
                name = f"{label} · {h5_path.stem}[{sample_idx-1}]"
                results = _process_microstructure(voxels, name, n_phases)
                all_results.extend(results)

    if not all_results:
        print(
            "\nNo microstructures could be processed.\n"
            "Check your HF token and network connection.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"  Processed {len(all_results)} phase SDF(s) total")
    total_time = sum(r["time_s"] for r in all_results)
    print(f"  Total SDF time: {total_time:.2f}s")
    print(f"{'='*60}")

    out_html = _OUTPUT_DIR / "microgen3d_report.html"
    _build_report(all_results, out_html)


if __name__ == "__main__":
    main()
