"""2D AMReX SDF demo.
    amr.Geometry → SDFMultiFab2D → MultiFab → write_single_level_plotfile
    → yt.load → SlicePlot → PNG

Run with:
    conda run -n pyamrex python examples/amrex/amrex_demo_2d_yt.py

Outputs PNGs to examples/amrex/output/.

Note: This script requires amrex.space2d (pyAMReX) and yt.
      It CANNOT run in the same process as amrex_demo_3d.py (space2d vs space3d conflict).
"""

import os
import sys
import amrex.space2d as amr
import yt

# pySdf root on the path so we can import SDFMultiFab2D
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from sdf2d.amrex import SDFMultiFab2D

# ---------------------------------------------------------------------------
# AMReX initialisation
# ---------------------------------------------------------------------------
amr.initialize([])

# Grid: 128x128 cells covering the [-1, 1]^2 domain
n = 128
real_box = amr.RealBox([-1.0, -1.0], [1.0, 1.0])
domain   = amr.Box([0, 0], [n - 1, n - 1])

# coord_sys=0 → Cartesian; is_periodic=[0,0] → non-periodic
geom = amr.Geometry(domain, real_box, 0, [0, 0])

# Decompose into 4 boxes (each 64x64) and map to MPI ranks
ba = amr.BoxArray(domain)
ba.max_size(n // 2)
dm = amr.DistributionMapping(ba)

lib = SDFMultiFab2D(geom, ba, dm)

# ---------------------------------------------------------------------------
# Build MultiFabs for each shape
# ---------------------------------------------------------------------------
from sdf2d import Circle2D, Box2D, RoundedBox2D, Hexagon2D
mf_circle      = lib.from_geometry(Circle2D(0.5))
mf_box         = lib.from_geometry(Box2D((0.4, 0.3)))
mf_rounded_box = lib.from_geometry(RoundedBox2D((0.4, 0.3), 0.1))
mf_hexagon     = lib.from_geometry(Hexagon2D(0.4))
mf_union       = lib.union(mf_circle, mf_box)
mf_subtract    = lib.subtract(mf_box, mf_circle)  # box with circle subtracted

# (name, MultiFab) pairs to iterate
shapes = [
    ("circle",      mf_circle),
    ("box",         mf_box),
    ("rounded_box", mf_rounded_box),
    ("hexagon",     mf_hexagon),
    ("union",       mf_union),
    ("subtract",    mf_subtract),
]

# ---------------------------------------------------------------------------
# Write plotfiles and render PNGs
# ---------------------------------------------------------------------------
out_dir = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(out_dir, exist_ok=True)

for name, mf in shapes:
    pf_dir  = os.path.join(out_dir, f"plt_{name}")
    png_out = os.path.join(out_dir, f"{name}.png")

    # --- Step 1: write AMReX plotfile (a directory tree, not a single file) ---
    # varnames must match what yt will look for: ("boxlib", "sdf")
    amr.write_single_level_plotfile(pf_dir, mf, amr.Vector_string(["sdf"]), geom, 0.0, 0)

    # --- Step 2: load with yt ---
    # yt reads the AMReX plotfile directory directly.
    # For 2D AMReX data yt uses a pseudo-3D representation with a thin z slab.
    ds = yt.load(pf_dir)

    # --- Step 3: SlicePlot along the synthetic z axis ---
    field = ("boxlib", "sdf")
    p = yt.SlicePlot(ds, "z", field)

    # SDF has negative values inside shapes — log scale would break/mislead.
    p.set_log(field, False)

    # seismic colormap: blue=inside (negative), red=outside (positive), white=surface
    p.set_cmap(field, "seismic")

    # Overlay the zero-level set (= the shape boundary).
    # plot_args are forwarded to matplotlib; clim pins the one contour at ~0.
    p.annotate_contour(
        field, levels=1, clim=(-1e-4, 1e-4),
        plot_args={"colors": "black", "linewidths": 1.5},
    )

    p.save(png_out)
    print(f"Saved {name}.png")

amr.finalize()
print("Done.")
