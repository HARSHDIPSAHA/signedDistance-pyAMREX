"""Intersection of two overlapping spheres.

Demonstrates: Sphere3D, intersect(), save_png()
Output:       examples/sdf3d/output/intersection_example.png
"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sdf3d import Sphere3D

_BOUNDS = ((-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0))
_OUT    = Path(__file__).parent / "output" / "intersection_example.png"

s1   = Sphere3D(0.35)
s2   = Sphere3D(0.35).translate(0.2, 0.0, 0.0)
geom = s1.intersect(s2)
geom.save_png(_OUT, bounds=_BOUNDS, color=(0.3, 0.7, 1.0),
              title="Intersection: S1 ∩ S2")
