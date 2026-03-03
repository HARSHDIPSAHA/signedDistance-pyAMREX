import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sdf3d import Sphere3D

base   = Sphere3D(0.40)
cutter = Sphere3D(0.25).translate(0.2, 0.0, 0.0)
base.subtract(cutter).save_png("subtraction_example.png", color=(1.0, 0.4, 0.3),
                                title="Subtraction: base − cutter")
