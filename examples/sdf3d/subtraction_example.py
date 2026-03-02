"""Subtraction: sphere with a spherical cavity.

Demonstrates: Sphere3D, subtract(), save_png()
Output:       examples/sdf3d/output/subtraction_example.png

Argument order reminder:
    opSubtraction(d1, d2) = max(-d1, d2)   where d1=CUTTER, d2=BASE
    base.subtract(cutter)                   fluent form
"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sdf3d import Sphere3D

_BOUNDS = ((-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0))
_OUT    = Path(__file__).parent / "output" / "subtraction_example.png"

base   = Sphere3D(0.40)
cutter = Sphere3D(0.25).translate(0.2, 0.0, 0.0)
geom   = base.subtract(cutter)
geom.save_png(_OUT, bounds=_BOUNDS, color=(1.0, 0.4, 0.3),
              title="Subtraction: base − cutter")
