"""Union of two overlapping spheres.

Demonstrates: Sphere3D, union(), save_png()
Output:       examples/sdf3d/output/union_example.png
"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sdf3d import Sphere3D

_BOUNDS = ((-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0))
_OUT    = Path(__file__).parent / "output" / "union_example.png"

s1   = Sphere3D(0.3).translate(-0.2, 0.0, 0.0)
s2   = Sphere3D(0.3).translate( 0.2, 0.0, 0.0)
geom = s1.union(s2)
geom.save_png(_OUT, bounds=_BOUNDS, title="Union: S1 ∪ S2")
