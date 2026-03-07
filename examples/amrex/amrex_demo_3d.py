"""3D AMReX SDF demo.

Pipeline:
    Geometry → SDFLibrary3D → MultiFab → write_single_level_plotfile
    → VisIt (isosurface at SDF=0) → PNG

Run with:
    conda run -n pyamrex python examples/amrex/amrex_demo_3d.py

Outputs plotfile directories and PNGs to examples/amrex/output/.

Requirements:
  - pyAMReX  (amrex.space3d) — conda install pyamrex -c conda-forge
  - VisIt    — https://visit-dav.github.io/visit-website/
              must be on PATH, or set VISIT_EXE environment variable

IMPORTANT: Run this script in its own process.
  amrex.space3d and amrex.space2d share pybind11 type names and cannot
  coexist in the same Python interpreter.
"""

import os
import sys
import glob
import shutil
import subprocess

# ---------------------------------------------------------------------------
# Locate VisIt executable
# ---------------------------------------------------------------------------
def _find_visit():
    """Return path to the visit executable, or raise SystemExit with instructions."""
    exe = os.environ.get("VISIT_EXE")
    if exe and os.path.isfile(exe):
        return exe

    exe = shutil.which("visit")
    if exe:
        return exe

    pattern = os.path.expanduser(
        "~/AppData/Local/Programs/LLNL/VisIt*/visit.exe"
    )
    candidates = sorted(glob.glob(pattern))
    if candidates:
        return candidates[-1]

    raise SystemExit(
        "VisIt not found.\n"
        "  • Download from https://visit-dav.github.io/visit-website/\n"
        "  • After installing, either add VisIt to PATH, or set:\n"
        "      VISIT_EXE=C:\\path\\to\\visit.exe"
    )


# ---------------------------------------------------------------------------
# AMReX import
# ---------------------------------------------------------------------------
try:
    import amrex.space3d as amr
except ImportError as exc:
    raise SystemExit(
        "amrex.space3d not found.\n"
        "  conda install pyamrex -c conda-forge"
    ) from exc

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from sdf3d.amrex import SDFLibrary3D

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
visit_exe    = _find_visit()
visit_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "visit_render_3d.py")
out_dir      = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(out_dir, exist_ok=True)

# ---------------------------------------------------------------------------
# AMReX initialisation — 48³ grid over [-0.6, 0.6]³
# (48 cells keeps render time short; raise to 64+ for higher fidelity)
# ---------------------------------------------------------------------------
amr.initialize([])

n        = 48
lo, hi   = -0.6, 0.6
real_box = amr.RealBox([lo, lo, lo], [hi, hi, hi])
domain   = amr.Box([0, 0, 0], [n - 1, n - 1, n - 1])
geom     = amr.Geometry(domain, real_box, 0, [0, 0, 0])
ba       = amr.BoxArray(domain)
ba.max_size(n // 2)
dm       = amr.DistributionMapping(ba)
lib      = SDFLibrary3D(geom, ba, dm)

# ---------------------------------------------------------------------------
# Build MultiFabs for each demo shape
# ---------------------------------------------------------------------------
mf_sphere    = lib.sphere((0, 0, 0), 0.3)
mf_box       = lib.box((0, 0, 0), (0.25, 0.2, 0.15))
mf_round_box = lib.round_box((0, 0, 0), (0.25, 0.2, 0.15), 0.05)
mf_union     = lib.union(mf_sphere, mf_box)
mf_subtract  = lib.subtract(mf_box, mf_sphere)  # box with sphere cut out

shapes = [
    ("sphere",    mf_sphere),
    ("box",       mf_box),
    ("round_box", mf_round_box),
    ("union",     mf_union),
    ("subtract",  mf_subtract),
]

# ---------------------------------------------------------------------------
# Write plotfiles → render each with VisIt
# ---------------------------------------------------------------------------
for name, mf in shapes:
    pf_dir  = os.path.join(out_dir, f"plt_{name}")
    png_out = os.path.join(out_dir, f"{name}.png")

    # Step 1: write the AMReX plotfile
    amr.write_single_level_plotfile(
        pf_dir, mf, amr.Vector_string(["sdf"]), geom, 0.0, 0
    )
    print(f"  Wrote plotfile: {pf_dir}")

    # Step 2: VisIt renders the SDF=0 isosurface
    env = {
        **os.environ,
        "VISIT_PF_DIR":  os.path.abspath(pf_dir),
        "VISIT_PNG_OUT": os.path.abspath(png_out),
    }
    log_path = os.path.join(out_dir, f"{name}.visit.log")
    with open(log_path, "w") as log:
        result = subprocess.run(
            [visit_exe, "-cli", "-nowin", "-s", visit_script],
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            timeout=120,
        )
    if result.returncode != 0:
        print(f"  [VisIt ERROR] {name} (see {name}.visit.log):")
        with open(log_path) as f:
            print(f.read()[-600:])
    else:
        print(f"  Saved: {name}.png")

amr.finalize()
print("Done.")
