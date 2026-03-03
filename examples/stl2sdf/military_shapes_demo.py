"""military_shapes_demo.py — Artillery shell & missile STL → SDF demo.

Loads the two STL files from the same directory, computes their signed distance
fields on a regular 3-D grid, and produces an interactive Plotly HTML report
showing the φ = 0 isosurface for each shape.

Usage
-----
uv run python examples/stl2sdf/military_shapes_demo.py            # res=50
uv run python examples/stl2sdf/military_shapes_demo.py --res 80  # higher quality
uv run python examples/stl2sdf/military_shapes_demo.py --out my_report.html
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from stl2sdf import stl_to_geometry, mesh_bounds
from sdf3d import save_plotly_html_grid

_HERE   = Path(__file__).parent

SHAPES = [
    ("Artillery Shell", _HERE / "artillery_shell.stl"),
    ("Missile",         _HERE / "missile.stl"),
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Artillery shell & missile STL → SDF demo"
    )
    parser.add_argument("--res", type=int, default=50,
                        help="Grid resolution per axis (default 50)")
    parser.add_argument("--out", type=Path, default=Path("military_shapes_report.html"),
                        help="Output HTML report path")
    args = parser.parse_args()

    panels = []
    for name, stl in SHAPES:
        if not stl.exists():
            print(f"SKIP: {stl.name} not found in examples/stl2sdf/")
            continue
        print(f"Loading {stl.name} ...", flush=True)
        geom   = stl_to_geometry(stl)
        bounds = mesh_bounds(stl)
        panels.append((geom, name, bounds))

    if not panels:
        print("No shapes found — nothing to render.")
        return

    save_plotly_html_grid(
        panels,
        args.out,
        resolution=(args.res, args.res, args.res),
        title="Military Shape Gallery — φ = 0 Surfaces",
        color="#c0392b",
    )


if __name__ == "__main__":
    main()
