"""
microgen3d_img2sdf_demo_mesh.py

Efficient visualization version.

Major improvements:
• Uses marching cubes meshes instead of voxel isosurfaces
• Saves EACH component as an individual HTML file
• Generates a small index page linking all results
• Reduces HTML size by ~1000×

Output structure:

output/
 ├── panels/
 │   ├── panel_000.html
 │   ├── panel_001.html
 │   └── ...
 └── index.html
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

_SCRIPT_DIR = Path(__file__).parent
_OUTPUT_DIR = _SCRIPT_DIR / "output"
_PANEL_DIR = _OUTPUT_DIR / "panels"

_HF_DATASET = "BGLab/microgen3D"

_SAMPLE_ARCHIVES = [
    {"hf_path": "data/sample_CH_two_phase.tar.gz", "label": "CH 2-Phase", "n_phases": 2},
    {"hf_path": "data/sample_CH_three_phase.tar.gz", "label": "CH 3-Phase", "n_phases": 3},
]


# ----------------------------------------------------------
# dependency check
# ----------------------------------------------------------

def _require(pkg: str):
    try:
        __import__(pkg)
    except ImportError:
        print(f"\nERROR: install {pkg}\n", file=sys.stderr)
        sys.exit(1)


# ----------------------------------------------------------
# huggingface
# ----------------------------------------------------------

def _get_token(cli):
    if cli:
        return cli
    return os.environ.get("HF_TOKEN", None)


# ----------------------------------------------------------
# download
# ----------------------------------------------------------

def _download_archive(hf_path, token, cache_dir):
    from huggingface_hub import hf_hub_download

    return Path(
        hf_hub_download(
            repo_id=_HF_DATASET,
            filename=hf_path,
            repo_type="dataset",
            token=token,
            cache_dir=str(cache_dir),
        )
    )


def _extract_archive(archive_path, dest):
    dest.mkdir(parents=True, exist_ok=True)

    print("extracting", archive_path.name)

    with tarfile.open(archive_path, "r:gz") as tf:
        tf.extractall(dest)

    return sorted(dest.rglob("*.h5"))


# ----------------------------------------------------------
# hdf5
# ----------------------------------------------------------

def _iter_voxel_arrays(h5_path, max_samples):
    import h5py

    with h5py.File(h5_path, "r") as f:

        for key in f.keys():

            arr = np.asarray(f[key])

            if arr.ndim == 4:
                for i in range(min(max_samples, arr.shape[0])):
                    yield arr[i].astype(np.float32)

            if arr.ndim == 3:
                yield arr.astype(np.float32)


# ----------------------------------------------------------
# sdf
# ----------------------------------------------------------

def _phase_to_phi(voxels, pid):

    from scipy.ndimage import distance_transform_edt

    mask = voxels == pid

    dist_inside = distance_transform_edt(mask)
    dist_outside = distance_transform_edt(~mask)

    return dist_outside - dist_inside


# ----------------------------------------------------------
# mesh generation
# ----------------------------------------------------------

def _phi_to_mesh(phi):

    from skimage.measure import marching_cubes

    verts, faces, _, _ = marching_cubes(phi, level=0)

    return verts, faces


# ----------------------------------------------------------
# save single html
# ----------------------------------------------------------

def _save_mesh_html(verts, faces, title, filename):

    import plotly.graph_objects as go

    fig = go.Figure(
        go.Mesh3d(
            x=verts[:, 0],
            y=verts[:, 1],
            z=verts[:, 2],
            i=faces[:, 0],
            j=faces[:, 1],
            k=faces[:, 2],
            color="lightblue",
            opacity=1,
        )
    )

    fig.update_layout(
        title=title,
        scene_aspectmode="data",
        margin=dict(l=0, r=0, t=40, b=0),
    )

    fig.write_html(filename, include_plotlyjs="cdn")


# ----------------------------------------------------------
# process microstructure
# ----------------------------------------------------------

def _process_microstructure(voxels, label, n_phases):

    results = []

    phases = np.unique(voxels)[:n_phases]

    for pid in phases:

        t0 = time.time()

        phi = _phase_to_phi(voxels, pid)

        dt = time.time() - t0

        vol = float((voxels == pid).mean()) * 100

        print(f"phase {pid} vol={vol:.1f}% time={dt:.2f}s")

        if phi.min() >= 0 or phi.max() <= 0:
            continue

        verts, faces = _phi_to_mesh(phi)

        results.append(
            {
                "name": f"{label} phase {pid}",
                "verts": verts,
                "faces": faces,
                "vol": vol,
                "time": dt,
            }
        )

    return results


# ----------------------------------------------------------
# save results
# ----------------------------------------------------------

def _save_results(results):

    import json

    _PANEL_DIR.mkdir(parents=True, exist_ok=True)

    links = []

    for i, r in enumerate(results):

        fname = _PANEL_DIR / f"panel_{i:04d}.html"

        _save_mesh_html(r["verts"], r["faces"], r["name"], fname)

        links.append((r["name"], fname.name))

    # index page
    index = ["<h1>microgen3D results</h1>"]

    for name, link in links:
        index.append(f'<p><a href="panels/{link}">{name}</a></p>')

    (_OUTPUT_DIR / "index.html").write_text("\n".join(index))

    print("\nindex saved:", _OUTPUT_DIR / "index.html")


# ----------------------------------------------------------
# main
# ----------------------------------------------------------

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--max-samples", type=int, default=2)

    parser.add_argument("--token")

    parser.add_argument(
        "--cache-dir", default=str(_OUTPUT_DIR / "hf_cache")
    )

    args = parser.parse_args()

    _require("huggingface_hub")
    _require("h5py")
    _require("scipy")
    _require("skimage")
    _require("plotly")

    token = _get_token(args.token)

    cache = Path(args.cache_dir)

    all_results = []

    for arc in _SAMPLE_ARCHIVES:

        print("\narchive:", arc["hf_path"])

        archive = _download_archive(arc["hf_path"], token, cache)

        h5files = _extract_archive(archive, cache / Path(arc["hf_path"]).stem)

        for h5 in h5files:

            print("hdf5:", h5.name)

            for vox in _iter_voxel_arrays(h5, args.max_samples):

                res = _process_microstructure(vox, h5.stem, arc["n_phases"])

                all_results.extend(res)

    print("\nTotal panels:", len(all_results))

    _save_results(all_results)


if __name__ == "__main__":
    main()
