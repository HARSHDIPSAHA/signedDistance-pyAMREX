from pathlib import Path
import sys
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sdf3d import Box3D
from sdf3d.examples import NATOFragment

_OUT_DIR = Path(__file__).parent / "output"


class _MockLib:
    def from_geometry(self, geom):
        return geom


# Fragment dimensions (STANAG-4496 standard)
diameter = 14.30e-3   # 14.30 mm
lib = _MockLib()
_, fragment_geom = NATOFragment(lib, diameter=diameter, L_over_D=1.09,
                                cone_angle_deg=20.0)
total_length = diameter * 1.09   # ~ 15.56 mm

# Fragment alone
frag_bounds = (
    (-diameter, diameter),
    (-diameter, diameter),
    (-diameter * 0.2, total_length + diameter * 0.2),
)
fragment_geom.save_png(
    _OUT_DIR / "nato_fragment.png",
    bounds=frag_bounds,
    resolution=(48, 48, 80),
    color=(0.7, 0.75, 0.8),
    title=f"STANAG-4496 fragment  dia{diameter*1e3:.1f} mm",
    elev=20, azim=45,
)

# Impact scene: fragment positioned 20 mm in front of a 50 mm target cube
target_half  = 0.025
target_z     = 0.060
gap          = 0.020
target_front = target_z - target_half
z_shift      = (target_front - gap) - total_length

positioned = (fragment_geom
              .rotate_y(np.radians(5.0))
              .translate(0.0, 0.0, z_shift))
target = Box3D([target_half] * 3).translate(0.0, 0.0, target_z)
scene  = positioned.union(target)

scene_bounds = ((-0.04, 0.10), (-0.04, 0.10), (-0.04, 0.10))
scene.save_png(
    _OUT_DIR / "nato_impact_scene.png",
    bounds=scene_bounds,
    color=(0.5, 0.65, 0.9),
    title="Impact scene: fragment + target",
    elev=20, azim=45,
)
