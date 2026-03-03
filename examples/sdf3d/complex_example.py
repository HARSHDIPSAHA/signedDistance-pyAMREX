import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sdf3d import Sphere3D, Box3D

# Step 1 — base box
base = Box3D([0.30, 0.30, 0.30])
base.save_png("complex_example_step1.png", color=(0.5, 0.7, 1.0), title="Step 1: Box")

# Step 2 — capsule (elongated sphere)
capsule = Sphere3D(0.18).elongate(0.25, 0.0, 0.0)
capsule.save_png("complex_example_step2.png", color=(0.3, 0.9, 0.4), title="Step 2: Capsule")

# Step 3 — union
union_geom = base.union(capsule)
union_geom.save_png("complex_example_step3.png", color=(0.9, 0.8, 0.2), title="Step 3: Union")

# Step 4 — intersection with a large sphere -> rounds the top
rounder = Sphere3D(0.60).translate(0.0, 0.2, 0.0)
rounded = union_geom.intersect(rounder)
rounded.save_png("complex_example_step4.png", color=(1.0, 0.5, 0.3), title="Step 4: Intersection (rounded)")

# Step 5 — subtract a small central box -> cavity
cutter = Box3D([0.08, 0.08, 0.08]).translate(0.0, 0.05, 0.0)
rounded.subtract(cutter).save_png("complex_example_final.png", color=(0.7, 0.4, 1.0), title="Final: with cavity")
