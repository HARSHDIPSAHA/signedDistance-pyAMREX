import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sdf3d import Sphere3D

s1   = Sphere3D(0.3).translate(-0.2, 0.0, 0.0)
s2   = Sphere3D(0.3).translate( 0.2, 0.0, 0.0)
s1.union(s2).save_png("union_example.png", title="Union: S1 ∪ S2")
