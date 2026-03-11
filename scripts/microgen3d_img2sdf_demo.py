"""microgen3d_img2sdf_demo.py — img2sdf pipeline on the microgen3D HF dataset.

Downloads voxelized 3-D microstructures from the BGLab/microgen3D HuggingFace
dataset, runs them through pySdf's **img2sdf** library, and saves a Plotly
HTML report demonstrating:

1.  ``img2sdf.ImageGeometry3D``  — wraps a pre-computed SDF as a pySdf CSG node
2.  CSG operators  ``|`` / ``-`` / ``/``  — union, subtract, intersect geometries
3.  ``img2sdf.compute_morphometry_3d``  — volume, surface area, sphericity per phase
4.  ``img2sdf.volume_to_geometry_3d`` — optional Chan-Vese path on continuous data

Dataset layout (from https://huggingface.co/datasets/BGLab/microgen3D)
-----------------------------------------------------------------------
* ``data/sample_CH_two_phase.tar.gz``   (~9.6 MB, 2-phase Cahn–Hilliard samples)
* ``data/sample_CH_three_phase.tar.gz`` (~12.2 MB, 3-phase Cahn–Hilliard samples)

After extraction each archive contains HDF5 shards under ``train/`` and ``val/``
directories.  Each HDF5 file holds multiple 3-D voxel arrays stored under
integer keys (``"0"``, ``"1"``, …) or a batch ``"data"`` key.

img2sdf pipeline
----------------
For *pre-labeled* (phase-ID) voxels the recommended pipeline is::

    mask = (voxels == phase_id)
    phi  = distance_transform_edt(mask) - distance_transform_edt(~mask)
    geom = ImageGeometry3D(phi, bounds)   # full pySdf CSG node

For *raw continuous-intensity* volumes (e.g. X-ray tomography) use::

    geom = volume_to_geometry_3d(volume, params={"Segmentation": {}})

Prerequisites
-------------
::

    pip install huggingface-hub>=0.20 h5py scipy scikit-image plotly

Usage
-----
::

    python scripts/microgen3d_img2sdf_demo.py
    python scripts/microgen3d_img2sdf_demo.py --max-samples 2 --token YOUR_TOKEN
    python scripts/microgen3d_img2sdf_demo.py --chan-vese          # Chan-Vese path

Environment variable ``HF_TOKEN`` is also honoured if set.
"""

from __future__ import annotations

import argparse
import os
import sys
import tarfile
import time
from pathlib import Path
from typing import Iterator

import numpy as np

# ---------------------------------------------------------------------------
# Repo root on sys.path so pySdf modules are importable
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

_SCRIPT_DIR = Path(__file__).parent
_OUTPUT_DIR  = _SCRIPT_DIR / "output"
_HF_DATASET  = "BGLab/microgen3D"

_SAMPLE_ARCHIVES = [
    {
        "hf_path":  "data/sample_CH_two_phase.tar.gz",
        "label":    "CH 2-Phase",
        "n_phases": 2,
    },
    {
        "hf_path":  "data/sample_CH_three_phase.tar.gz",
        "label":    "CH 3-Phase",
        "n_phases": 3,
    },
]


# ===========================================================================
# Dependency checks
# ===========================================================================

def _require(pkg: str, install_hint: str | None = None) -> None:
    """Exit with a helpful message if *pkg* is not importable."""
    try:
        __import__(pkg)
    except ImportError:
        hint = install_hint or f"pip install {pkg}"
        print(f"\nERROR: '{pkg}' is not installed.\n  {hint}\n", file=sys.stderr)
        sys.exit(1)


# ===========================================================================
# Token handling
# ===========================================================================

def _get_token(cli_token: str | None) -> str | None:
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
# Download + extraction
# ===========================================================================

def _download_archive(hf_path: str, token: str | None, cache_dir: Path) -> Path:
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
    """Extract *archive_path* (.tar.gz) and return sorted list of .h5 files."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Extracting {archive_path.name} → {dest_dir} …", flush=True)
    with tarfile.open(archive_path, "r:gz") as tf:
        tf.extractall(dest_dir)
    h5_files = sorted(dest_dir.rglob("*.h5")) + sorted(dest_dir.rglob("*.hdf5"))
    print(f"    Found {len(h5_files)} HDF5 file(s).", flush=True)
    return h5_files


# ===========================================================================
# HDF5 reading helpers
# ===========================================================================

def _iter_voxel_arrays(h5_path: Path, max_samples: int) -> Iterator[np.ndarray]:
    """Yield up to *max_samples* 3-D voxel arrays from *h5_path*.

    Handles multiple common HDF5 layouts used in microgen3D:
    - Batch key ``"data"`` with shape ``(N, D, H, W)``
    - Integer-named datasets ``"0"``, ``"1"``, … each ``(D, H, W)``
    - Any remaining 3-D dataset (catch-all)
    """
    import h5py

    yielded = 0

    with h5py.File(h5_path, "r") as f:
        all_keys = list(f.keys())

        # Pattern 1 — named batch keys
        for key in ("data", "voxel", "microstructure", "images"):
            if key not in f:
                continue
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

        # Pattern 2 — integer-named keys
        int_keys = sorted(k for k in all_keys if k.isdigit())
        for key in int_keys:
            arr = np.asarray(f[key])
            if arr.ndim == 3:
                yield arr.astype(np.float32)
                yielded += 1
                if yielded >= max_samples:
                    return

        if yielded >= max_samples:
            return

        # Pattern 3 — any remaining 3-D or 4-D dataset
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
# Core img2sdf helpers
# ===========================================================================

def _phase_to_phi(voxels: np.ndarray, phase_id: int) -> np.ndarray:
    """Binary mask → signed distance field (pySdf convention: phi < 0 inside).

    Parameters
    ----------
    voxels:
        3-D integer/float phase array.
    phase_id:
        The phase label to treat as *inside* (solid).

    Returns
    -------
    numpy.ndarray
        Shape (D, H, W), dtype float64.  ``phi < 0`` inside, ``phi > 0`` outside.
    """
    from scipy.ndimage import distance_transform_edt

    mask = (np.round(voxels).astype(np.int32) == int(phase_id))
    if not mask.any():
        return np.full(voxels.shape, np.inf, dtype=np.float64)
    if mask.all():
        return np.full(voxels.shape, -np.inf, dtype=np.float64)

    dist_inside  = distance_transform_edt(mask)   # distance from surface, inside
    dist_outside = distance_transform_edt(~mask)  # distance from surface, outside
    return (dist_outside - dist_inside).astype(np.float64)   # phi < 0 inside


def _make_image_geometry(
    phi: np.ndarray,
    voxel_size: float = 1.0,
) -> "img2sdf.ImageGeometry3D":
    """Wrap a phi array in an ``ImageGeometry3D`` pySdf CSG node.

    Parameters
    ----------
    phi:
        Level-set array (D, H, W), pySdf convention (phi < 0 inside).
    voxel_size:
        Physical size of each voxel (isotropic, used to set ``bounds``).

    Returns
    -------
    ImageGeometry3D
        Full pySdf CSG node — supports ``|``, ``-``, ``/``, ``.translate()``, etc.
    """
    import img2sdf
    D, H, W = phi.shape
    bounds = (
        (0.0, W * voxel_size),
        (0.0, H * voxel_size),
        (0.0, D * voxel_size),
    )
    return img2sdf.ImageGeometry3D(phi, bounds)


# ===========================================================================
# Per-microstructure processing
# ===========================================================================

def _process_microstructure(
    voxels: np.ndarray,
    label: str,
    n_phases: int,
    *,
    voxel_size: float = 1.0,
    chan_vese: bool = False,
    chan_vese_params: dict | None = None,
) -> list[dict]:
    """Run the img2sdf pipeline on a single 3-D voxel microstructure.

    Steps
    -----
    1.  For each detected phase build a per-phase ``ImageGeometry3D``
        (EDT-based phi, pySdf sign convention).
    2.  Compute morphometrics via ``img2sdf.compute_morphometry_3d``.
    3.  Demonstrate CSG: union of all phases, clip phase-1 to a spherical ROI.
    4.  Optionally run ``img2sdf.volume_to_geometry_3d`` (Chan-Vese) on the
        raw voxel array for a qualitative comparison.

    Returns
    -------
    list of dict
        One dict per visualisation panel with keys:
        ``name``, ``geom``, ``phi``, ``bounds``, ``morpho``, ``panel_type``.
    """
    import img2sdf
    from sdf3d import Sphere3D

    D, H, W = voxels.shape
    unique_phases = np.unique(np.round(voxels).astype(np.int32))[:n_phases]
    voxel_bounds  = (
        (0.0, W * voxel_size),
        (0.0, H * voxel_size),
        (0.0, D * voxel_size),
    )

    results: list[dict] = []

    # ------------------------------------------------------------------ #
    # 1.  Per-phase ImageGeometry3D + morphometrics
    # ------------------------------------------------------------------ #
    phase_geoms: dict[int, img2sdf.ImageGeometry3D] = {}
    for pid in unique_phases:
        t0 = time.perf_counter()
        phi = _phase_to_phi(voxels, int(pid))
        geom = _make_image_geometry(phi, voxel_size=voxel_size)
        phase_geoms[int(pid)] = geom
        elapsed = time.perf_counter() - t0

        vol_frac = float((np.round(voxels).astype(np.int32) == pid).mean()) * 100

        # Morphometrics (requires scikit-image)
        morpho: dict | None = None
        try:
            morpho = img2sdf.compute_morphometry_3d(phi, voxel_size=voxel_size)
        except ImportError:
            pass  # scikit-image not installed — skip morphometry

        print(
            f"    [img2sdf] Phase {pid}: vol={vol_frac:.1f}%  "
            f"phi∈[{phi.min():.1f},{phi.max():.1f}]  {elapsed:.2f}s"
            + (
                f"  morpho: V={morpho['volume']:.1f} "
                f"A={morpho['surface_area']:.1f} ψ={morpho['sphericity']:.3f}"
                if morpho else ""
            ),
            flush=True,
        )

        results.append({
            "name":       f"{label} · phase {pid}",
            "geom":       geom,
            "phi":        phi,
            "bounds":     voxel_bounds,
            "morpho":     morpho,
            "vol_frac":   vol_frac,
            "phase_id":   int(pid),
            "panel_type": "phase",
            "time_s":     elapsed,
        })

    # ------------------------------------------------------------------ #
    # 2.  CSG: union of all detected phases → "total solid"
    # ------------------------------------------------------------------ #
    if len(phase_geoms) >= 2:
        t0 = time.perf_counter()
        geoms_list = list(phase_geoms.values())
        union_geom = geoms_list[0]
        for g in geoms_list[1:]:
            union_geom = union_geom | g          # pySdf | operator (union)
        phi_union = union_geom.to_numpy(
            voxel_bounds, (W, H, D)
        )
        elapsed = time.perf_counter() - t0
        print(
            f"    [img2sdf CSG] Union all phases: "
            f"phi∈[{phi_union.min():.1f},{phi_union.max():.1f}]  {elapsed:.2f}s",
            flush=True,
        )
        results.append({
            "name":       f"{label} · union (all phases)",
            "geom":       union_geom,
            "phi":        phi_union,
            "bounds":     voxel_bounds,
            "morpho":     None,
            "vol_frac":   float((phi_union < 0).mean()) * 100,
            "phase_id":   -1,
            "panel_type": "csg_union",
            "time_s":     elapsed,
        })

    # ------------------------------------------------------------------ #
    # 3.  CSG: clip first non-background phase to a spherical ROI
    # ------------------------------------------------------------------ #
    # Pick the first phase that is not 0 (or phase 1 for 2-phase data)
    clip_pid = unique_phases[-1]   # last phase is usually "material" phase
    if clip_pid in phase_geoms:
        t0 = time.perf_counter()
        # Sphere centred in voxel space, radius = 40 % of shortest side
        cx, cy, cz = W / 2.0, H / 2.0, D / 2.0
        r = 0.40 * min(W, H, D) * voxel_size
        sphere_clip = Sphere3D(r).translate(cx, cy, cz)
        # Intersect microstructure phase with bounding sphere
        clipped = phase_geoms[clip_pid] / sphere_clip   # pySdf / operator (intersect)
        phi_clip = clipped.to_numpy(voxel_bounds, (W, H, D))
        elapsed = time.perf_counter() - t0
        print(
            f"    [img2sdf CSG] Phase {clip_pid} ∩ sphere(r={r:.1f}): "
            f"phi∈[{phi_clip.min():.1f},{phi_clip.max():.1f}]  {elapsed:.2f}s",
            flush=True,
        )
        results.append({
            "name":       f"{label} · phase {clip_pid} ∩ sphere",
            "geom":       clipped,
            "phi":        phi_clip,
            "bounds":     voxel_bounds,
            "morpho":     None,
            "vol_frac":   float((phi_clip < 0).mean()) * 100,
            "phase_id":   clip_pid,
            "panel_type": "csg_clip",
            "time_s":     elapsed,
        })

    # ------------------------------------------------------------------ #
    # 4.  Optional Chan-Vese path via volume_to_geometry_3d
    # ------------------------------------------------------------------ #
    if chan_vese:
        t0 = time.perf_counter()
        seg_overrides = dict(chan_vese_params or {})
        seg_overrides.setdefault("max_iter", 30)
        cv_params = {"Segmentation": seg_overrides}
        try:
            cv_geom = img2sdf.volume_to_geometry_3d(
                voxels.astype(np.float64), cv_params
            )
            phi_cv  = cv_geom.phi
            elapsed = time.perf_counter() - t0
            print(
                f"    [img2sdf Chan-Vese] volume_to_geometry_3d: "
                f"phi∈[{phi_cv.min():.1f},{phi_cv.max():.1f}]  {elapsed:.2f}s",
                flush=True,
            )
            results.append({
                "name":       f"{label} · Chan-Vese segmentation",
                "geom":       cv_geom,
                "phi":        phi_cv,
                "bounds":     cv_geom.bounds,
                "morpho":     None,
                "vol_frac":   float((phi_cv < 0).mean()) * 100,
                "phase_id":   -2,
                "panel_type": "chan_vese",
                "time_s":     elapsed,
            })
        except Exception as exc:  # pragma: no cover
            print(f"    WARNING: Chan-Vese failed: {exc}", file=sys.stderr)

    return results


# ===========================================================================
# Plotly HTML report
# ===========================================================================

# Panel-type → colour mapping
_PANEL_COLORS = {
    "phase":     ["#4a90d9", "#e94560", "#2ecc71", "#f39c12", "#9b59b6"],
    "csg_union": "#1abc9c",
    "csg_clip":  "#e67e22",
    "chan_vese":  "#8e44ad",
}


def _panel_color(result: dict, idx: int) -> str:
    ptype = result["panel_type"]
    if ptype == "phase":
        return _PANEL_COLORS["phase"][idx % len(_PANEL_COLORS["phase"])]
    return _PANEL_COLORS.get(ptype, "#aaaaaa")


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

    cols       = min(n, 3)
    rows_scene = (n + cols - 1) // cols
    specs_scenes = [[{"type": "scene"}] * cols for _ in range(rows_scene)]
    specs_table  = [[{"type": "table", "colspan": cols}] + [None] * (cols - 1)]

    subplot_titles = [r["name"] for r in all_results]
    while len(subplot_titles) < rows_scene * cols:
        subplot_titles.append("")
    subplot_titles.append("")

    fig = make_subplots(
        rows=rows_scene + 1,
        cols=cols,
        specs=specs_scenes + specs_table,
        subplot_titles=subplot_titles,
        horizontal_spacing=0.04,
        vertical_spacing=0.06,
    )

    for idx, result in enumerate(all_results):
        row = idx // cols + 1
        col = idx % cols + 1
        phi    = result["phi"]
        bounds = result["bounds"]
        (x0, x1), (y0, y1), (z0, z1) = bounds
        D, H, W = phi.shape

        xs = np.linspace(x0, x1, W)
        ys = np.linspace(y0, y1, H)
        zs = np.linspace(z0, z1, D)
        Z3, Y3, X3 = np.meshgrid(zs, ys, xs, indexing="ij")

        if phi.min() >= 0 or phi.max() <= 0:
            print(f"  SKIP '{result['name']}': no zero-crossing "
                  f"(phi ∈ [{phi.min():.2f},{phi.max():.2f}])")
            continue

        color = _panel_color(result, idx)
        fig.add_trace(
            go.Isosurface(
                x=X3.ravel(), y=Y3.ravel(), z=Z3.ravel(),
                value=phi.ravel(),
                isomin=0.0, isomax=0.0, surface_count=1,
                colorscale=[[0, color], [1, color]],
                showscale=False,
                caps=dict(x_show=False, y_show=False, z_show=False),
                lighting=dict(ambient=0.4, diffuse=0.9, specular=0.3, roughness=0.5),
                lightposition=dict(x=1000, y=1000, z=2000),
                name=result["name"],
            ),
            row=row, col=col,
        )

    # ---- Statistics table ----
    has_morpho = any(r["morpho"] for r in all_results)
    header_vals = [
        "<b>Microstructure</b>", "<b>Panel</b>",
        "<b>Shape (D×H×W)</b>", "<b>Vol %</b>", "<b>SDF time</b>",
    ]
    if has_morpho:
        header_vals += ["<b>Volume</b>", "<b>Surf Area</b>", "<b>Sphericity</b>"]

    rows_data: list[list] = [[] for _ in header_vals]
    for r in all_results:
        D, H, W = r["phi"].shape
        rows_data[0].append(r["name"])
        rows_data[1].append(r["panel_type"])
        rows_data[2].append(f"{D}×{H}×{W}")
        rows_data[3].append(f"{r['vol_frac']:.1f}%")
        rows_data[4].append(f"{r['time_s']:.2f}s")
        if has_morpho:
            m = r.get("morpho") or {}
            rows_data[5].append(f"{m.get('volume', '—'):.1f}" if m else "—")
            rows_data[6].append(f"{m.get('surface_area', '—'):.1f}" if m else "—")
            rows_data[7].append(f"{m.get('sphericity', '—'):.3f}" if m else "—")

    fig.add_trace(
        go.Table(
            header=dict(
                values=header_vals,
                fill_color="#0f3460",
                font=dict(color="#e0e0e0", size=12),
                align="left",
            ),
            cells=dict(
                values=rows_data,
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
            text=(
                f"microgen3D × img2sdf — ImageGeometry3D + CSG + Morphometry "
                f"({n} panels)"
            ),
            font=dict(size=18),
        ),
        width=max(900, 420 * cols),
        height=520 * rows_scene + 380,
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
            "Download BGLab/microgen3D sample archives from HuggingFace and "
            "process them through pySdf's img2sdf library "
            "(ImageGeometry3D + CSG + morphometrics)."
        )
    )
    parser.add_argument(
        "--max-samples", type=int, default=2,
        help="Max microstructures to process per HDF5 file (default: 2)",
    )
    parser.add_argument(
        "--token", type=str, default=None,
        help="HuggingFace API token (overrides HF_TOKEN env var / prompt)",
    )
    parser.add_argument(
        "--cache-dir", type=str, default=str(_OUTPUT_DIR / "hf_cache"),
        help="Directory for HuggingFace downloads (default: scripts/output/hf_cache)",
    )
    parser.add_argument(
        "--voxel-size", type=float, default=1.0,
        help="Physical voxel size in user units (default: 1.0)",
    )
    parser.add_argument(
        "--chan-vese", action="store_true",
        help="Also run img2sdf.volume_to_geometry_3d (Chan-Vese) on each sample",
    )
    parser.add_argument(
        "--no-sample", action="store_true",
        help="Use full (multi-GB) archives instead of sample subsets",
    )
    args = parser.parse_args()

    # Check all dependencies up-front
    _require("huggingface_hub", "pip install huggingface-hub>=0.20")
    _require("h5py",   "pip install h5py")
    _require("scipy",  "pip install scipy")
    _require("img2sdf", "ensure pySdf is on PYTHONPATH")

    token     = _get_token(args.token)
    cache_dir = Path(args.cache_dir)
    archives  = _SAMPLE_ARCHIVES  # always use samples unless overridden
    if args.no_sample:
        archives = [
            {"hf_path": "data/CH_two_phase.tar.gz",   "label": "CH 2-Phase (full)", "n_phases": 2},
            {"hf_path": "data/CH_three_phase.tar.gz",  "label": "CH 3-Phase (full)", "n_phases": 3},
            {"hf_path": "data/experimental.tar.gz",    "label": "Experimental (full)", "n_phases": 2},
        ]

    print(f"\n{'='*62}")
    print("  microgen3D × pySdf img2sdf Demo")
    print(f"  Dataset     : {_HF_DATASET}")
    print(f"  img2sdf API : ImageGeometry3D, compute_morphometry_3d,")
    print(f"                volume_to_geometry_3d (CSG + morphometry)")
    print(f"  Archives    : {[a['hf_path'] for a in archives]}")
    print(f"  Max samples : {args.max_samples} per HDF5 file")
    if args.chan_vese:
        print("  Chan-Vese   : enabled (volume_to_geometry_3d)")
    print(f"{'='*62}\n")

    all_results: list[dict] = []

    for archive_info in archives:
        hf_path  = archive_info["hf_path"]
        alabel   = archive_info["label"]
        n_phases = archive_info["n_phases"]

        print(f"\n{'─'*62}")
        print(f"  Archive : {hf_path}  [{alabel}]")
        print(f"{'─'*62}")

        # Download
        print("  Downloading …", flush=True)
        try:
            local_archive = _download_archive(hf_path, token, cache_dir)
        except Exception as exc:
            print(f"  ERROR downloading {hf_path}: {exc}", file=sys.stderr)
            continue
        print(f"  → {local_archive}")

        # Extract
        extract_dir = cache_dir / Path(hf_path).stem
        try:
            h5_files = _extract_archive(local_archive, extract_dir)
        except Exception as exc:
            print(f"  ERROR extracting: {exc}", file=sys.stderr)
            continue

        if not h5_files:
            print("  WARNING: no HDF5 files found after extraction.")
            continue

        # Process each HDF5 file
        for h5_path in h5_files:
            print(f"\n  HDF5 : {h5_path.name}", flush=True)
            sample_idx = 0
            for voxels in _iter_voxel_arrays(h5_path, max_samples=args.max_samples):
                sample_idx += 1
                D, H, W = voxels.shape
                unique = np.unique(np.round(voxels).astype(np.int32))
                print(
                    f"  Sample {sample_idx}: shape=({D},{H},{W})  "
                    f"phases={unique.tolist()}",
                    flush=True,
                )
                name = f"{alabel} · {h5_path.stem}[{sample_idx-1}]"
                results = _process_microstructure(
                    voxels, name, n_phases,
                    voxel_size=args.voxel_size,
                    chan_vese=args.chan_vese,
                )
                all_results.extend(results)

    if not all_results:
        print(
            "\nNo microstructures could be processed.\n"
            "Check your HF token and network connection.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Summary
    print(f"\n{'='*62}")
    print(f"  Total panels   : {len(all_results)}")
    phases_count = sum(1 for r in all_results if r["panel_type"] == "phase")
    csg_count    = sum(1 for r in all_results
                       if r["panel_type"] in ("csg_union", "csg_clip"))
    cv_count     = sum(1 for r in all_results if r["panel_type"] == "chan_vese")
    print(f"  Per-phase SDF  : {phases_count}")
    print(f"  CSG operations : {csg_count}")
    if cv_count:
        print(f"  Chan-Vese      : {cv_count}")
    total_time = sum(r["time_s"] for r in all_results)
    print(f"  Total SDF time : {total_time:.2f}s")
    print(f"{'='*62}")

    out_html = _OUTPUT_DIR / "microgen3d_img2sdf_report.html"
    _build_report(all_results, out_html)


if __name__ == "__main__":
    main()
