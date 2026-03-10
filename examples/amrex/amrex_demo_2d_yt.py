"""2D AMReX SDF demo.
    MultiFabGrid2D → shape.to_multifab(grid) → write_single_level_plotfile
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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from sdf2d import Circle2D, Box2D, RoundedBox2D, Hexagon2D, MultiFabGrid2D

# ---------------------------------------------------------------------------
# AMReX initialisation
# ---------------------------------------------------------------------------
amr.initialize([])

# Grid: 128x128 cells covering the [-1, 1]^2 domain
n = 128
real_box = amr.RealBox([-1.0, -1.0], [1.0, 1.0])
domain   = amr.Box([0, 0], [n - 1, n - 1])
geom     = amr.Geometry(domain, real_box, 0, [0, 0])
ba       = amr.BoxArray(domain); ba.max_size(n // 2)
dm       = amr.DistributionMapping(ba)

grid = MultiFabGrid2D(geom, ba, dm)

# ---------------------------------------------------------------------------
# Build MultiFabs — fill each shape on the shared grid
# ---------------------------------------------------------------------------
mf_circle      = Circle2D(0.5).to_multifab(grid)
mf_box         = Box2D((0.4, 0.3)).to_multifab(grid)
mf_rounded_box = RoundedBox2D((0.4, 0.3), 0.1).to_multifab(grid)
mf_hexagon     = Hexagon2D(0.4).to_multifab(grid)
mf_union       = grid.union(mf_circle, mf_box)
mf_subtract    = grid.subtract(mf_box, mf_circle)   # box with circle carved out

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

    amr.write_single_level_plotfile(pf_dir, mf, amr.Vector_string(["sdf"]), geom, 0.0, 0)

    ds = yt.load(pf_dir)
    field = ("boxlib", "sdf")
    p = yt.SlicePlot(ds, "z", field)
    p.set_log(field, False)
    p.set_cmap(field, "seismic")
    p.annotate_contour(
        field, levels=1, clim=(-1e-4, 1e-4),
        plot_args={"colors": "black", "linewidths": 1.5},
    )
    p.save(png_out)
    print(f"Saved {name}.png")

amr.finalize()
print("Done.")
