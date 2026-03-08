"""2D AMReX SDF demo.

Pipeline:
    Geometry → SDFMultiFab2D → MultiFab → write_single_level_plotfile
    → VisIt (pseudocolor heatmap + zero-level contour) → PNG

Run with:
    conda run -n pyamrex python examples/amrex/amrex_demo_2d.py

Outputs plotfile directories and PNGs to examples/amrex/output/.

Requirements:
  - pyAMReX  (amrex.space2d) — conda install pyamrex -c conda-forge
  - VisIt    — https://visit-dav.github.io/visit-website/
              must be on PATH, or set VISIT_EXE environment variable

Note: this script CANNOT run in the same process as amrex_demo_3d.py
      (amrex.space2d and amrex.space3d share pybind11 type names).
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
    # 1. Explicit override via env var (e.g. set VISIT_EXE=C:\...\visit.exe)
    exe = os.environ.get("VISIT_EXE")
    if exe and os.path.isfile(exe):
        return exe

    # 2. On PATH (works on Linux/macOS and Windows if VisIt installer added it)
    exe = shutil.which("visit")
    if exe:
        return exe

    # 3. Common Windows install location
    pattern = os.path.expanduser(
        "~/AppData/Local/Programs/LLNL/VisIt*/visit.exe"
    )
    candidates = sorted(glob.glob(pattern))
    if candidates:
        return candidates[-1]   # pick the newest version

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
    import amrex.space2d as amr
except ImportError as exc:
    raise SystemExit(
        "amrex.space2d not found.\n"
        "  conda install pyamrex -c conda-forge"
    ) from exc

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from sdf2d.amrex import SDFMultiFab2D

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
visit_exe       = _find_visit()
visit_script    = os.path.join(os.path.dirname(os.path.abspath(__file__)), "visit_render_2d.py")
out_dir         = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(out_dir, exist_ok=True)

# ---------------------------------------------------------------------------
# AMReX initialisation — 128×128 grid over [-1,1]²
# ---------------------------------------------------------------------------
amr.initialize([])

n        = 128
real_box = amr.RealBox([-1.0, -1.0], [1.0, 1.0])
domain   = amr.Box([0, 0], [n - 1, n - 1])
geom     = amr.Geometry(domain, real_box, 0, [0, 0])  # coord_sys=0 (Cartesian)
ba       = amr.BoxArray(domain)
ba.max_size(n // 2)                                    # 2×2 decomposition → 4 boxes
dm       = amr.DistributionMapping(ba)
lib      = SDFMultiFab2D(geom, ba, dm)

# ---------------------------------------------------------------------------
# Build MultiFabs for each demo shape
# ---------------------------------------------------------------------------
from sdf2d import Circle2D, Box2D, RoundedBox2D, Hexagon2D
mf_circle      = lib.from_geometry(Circle2D(0.5))
mf_box         = lib.from_geometry(Box2D((0.4, 0.3)))
mf_rounded_box = lib.from_geometry(RoundedBox2D((0.4, 0.3), 0.1))
mf_hexagon     = lib.from_geometry(Hexagon2D(0.4))
mf_union       = lib.union(mf_circle, mf_box)
mf_subtract    = lib.subtract(mf_box, mf_circle)  # box with circle cut out

shapes = [
    ("circle",      mf_circle),
    ("box",         mf_box),
    ("rounded_box", mf_rounded_box),
    ("hexagon",     mf_hexagon),
    ("union",       mf_union),
    ("subtract",    mf_subtract),
]

# ---------------------------------------------------------------------------
# Write plotfiles → render each with VisIt
# ---------------------------------------------------------------------------
for name, mf in shapes:
    pf_dir  = os.path.join(out_dir, f"plt_{name}")
    png_out = os.path.join(out_dir, f"{name}.png")

    # Step 1: write the AMReX plotfile directory tree
    #   pf_dir/Header       — ASCII metadata (field name, grid info, domain bounds)
    #   pf_dir/Level_0/Cell — binary SDF values
    amr.write_single_level_plotfile(
        pf_dir, mf, amr.Vector_string(["sdf"]), geom, 0.0, 0
    )
    print(f"  Wrote plotfile: {pf_dir}")

    # Step 2: launch VisIt in headless CLI mode to render the plotfile.
    #   -cli   → Python scripting interface (no GUI)
    #   -nowin → offscreen rendering (no display needed)
    #   -s     → run the given Python script
    #
    # The two paths are passed via environment variables because VisIt's
    # handling of extra command-line arguments varies between versions.
    env = {
        **os.environ,
        "VISIT_PF_DIR":  os.path.abspath(pf_dir),
        "VISIT_PNG_OUT": os.path.abspath(png_out),
    }
    log_path = os.path.join(out_dir, f"{name}.visit.log")
    with open(log_path, "w") as log:
        # stdin=DEVNULL: if the script errors, VisIt won't drop to an interactive
        # REPL that blocks forever — it will fail immediately with no input.
        # Redirect to a log file instead of PIPE: VisIt launches a separate
        # viewer.exe that would keep a PIPE open indefinitely.
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
