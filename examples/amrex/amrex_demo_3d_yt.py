"""3D AMReX SDF demo.

    MultiFabGrid3D → shape.to_multifab(grid) → write_single_level_plotfile
    → yt (SurfaceSource or marching-cubes fallback) → PNG

Run with:
    conda run -n pyamrex python examples/amrex/amrex_demo_3d_yt.py

Outputs PNGs to examples/amrex/output/.

IMPORTANT: Run this script in its own process — amrex.space3d cannot coexist
with amrex.space2d in the same Python interpreter (pybind11 type-name conflict).
"""

import os
import sys
import amrex.space3d as amr

sys.path.insert(0, os.path.dirname(__file__))
from render_surface_from_plotfile import render_surface

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from sdf3d import Sphere3D, Box3D, RoundBox3D, MultiFabGrid3D

# ---------------------------------------------------------------------------
# AMReX initialisation
# ---------------------------------------------------------------------------
amr.initialize([])

# Grid: 48^3 cells covering [-0.6, 0.6]^3
n = 48
lo, hi = -0.6, 0.6
real_box = amr.RealBox([lo, lo, lo], [hi, hi, hi])
domain   = amr.Box([0, 0, 0], [n - 1, n - 1, n - 1])
geom     = amr.Geometry(domain, real_box, 0, [0, 0, 0])
ba       = amr.BoxArray(domain); ba.max_size(n // 2)
dm       = amr.DistributionMapping(ba)

grid = MultiFabGrid3D(geom, ba, dm)

# ---------------------------------------------------------------------------
# Build MultiFabs — fill each shape on the shared grid
# ---------------------------------------------------------------------------
mf_sphere    = Sphere3D(0.3).to_multifab(grid)
mf_box       = Box3D((0.25, 0.2, 0.15)).to_multifab(grid)
mf_round_box = RoundBox3D((0.25, 0.2, 0.15), 0.05).to_multifab(grid)
mf_union     = grid.union(mf_sphere, mf_box)
mf_subtract  = grid.subtract(mf_box, mf_sphere)   # box with sphere carved out

shapes = [
    ("sphere",    mf_sphere),
    ("box",       mf_box),
    ("round_box", mf_round_box),
    ("union",     mf_union),
    ("subtract",  mf_subtract),
]

# ---------------------------------------------------------------------------
# Write plotfiles and render PNGs
# ---------------------------------------------------------------------------
out_dir = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(out_dir, exist_ok=True)

for name, mf in shapes:
    pf_dir  = os.path.join(out_dir, f"plt_{name}")
    png_out = os.path.join(out_dir, f"{name}.png")

    amr.write_single_level_plotfile(pf_dir, mf, amr.Vector_string(["sdf"]), geom, 0.0, 0)
    render_surface(pf_dir, png_out)
    print(f"Saved {name}.png")

amr.finalize()
print("Done.")
