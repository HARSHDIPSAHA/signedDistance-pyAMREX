import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sdf3d import Sphere3D

s1   = Sphere3D(0.35)
s2   = Sphere3D(0.35).translate(0.2, 0.0, 0.0)
s1.intersect(s2).save_png("intersection_example.png", color=(0.3, 0.7, 1.0),
                           title="Intersection: S1 ∩ S2")
