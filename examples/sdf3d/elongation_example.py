import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sdf3d import Sphere3D

R = 0.25
H = 0.30   # elongation half-length along X

Sphere3D(R).elongate(H, 0.0, 0.0).save_png("elongation_example.png", color=(0.3, 0.9, 0.4),
                                             title=f"Elongation: sphere r={R}, h={H}")
