from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sdf3d import Sphere3D

_BOUNDS = ((-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0))
_OUT    = Path(__file__).parent / "output" / "elongation_example.png"

R = 0.25
H = 0.30   # elongation half-length along X

geom = Sphere3D(R).elongate(H, 0.0, 0.0)
geom.save_png(_OUT, bounds=_BOUNDS, color=(0.3, 0.9, 0.4),
              title=f"Elongation: sphere r={R}, h={H}")