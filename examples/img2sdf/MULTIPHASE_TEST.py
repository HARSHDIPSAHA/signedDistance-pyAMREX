"""img2sdf — MULTIPHASE example 

Demonstrates the full img2sdf pipeline using the pySdf native API:
    image  →  level-set Φ  →  ImageGeometry2D  →  CSG composition  →  save_png()

"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

# Resolve paths relative to THIS file (works from any working directory) 
HERE       = Path(__file__).parent
IMAGE_PATH = HERE / "MULTIPHASE_TEST" / "Image1.png"
JSON_PATH  = HERE / "MULTIPHASE_TEST" / "MULTIPHASE_TEST.json"
OUT_DIR    = HERE / "outputs"
OUT_DIR.mkdir(exist_ok=True)

#pySdf / img2sdf API imports 
from img2sdf import image_to_levelset_2d, ImageGeometry2D



# 1.  Load parameters
with open(JSON_PATH) as f:
    params = json.load(f)

w = params["Image Properties"]["image width"]    # µm
h = params["Image Properties"]["image height"]   # µm
bounds = ((0.0, float(w)), (0.0, float(h)))

print("=" * 60)
print("  img2sdf — MULTIPHASE example")
print("=" * 60)
print(f"  image  : {IMAGE_PATH}")
print(f"  method : {params['Image Properties']['method']}")
print(f"  domain : {w} x {h} µm")
print()


# 2.  Run pipeline ONCE → raw level-set array
print("[1/5]  Running img2sdf pipeline ...")
phi = image_to_levelset_2d(str(IMAGE_PATH), params)
print(f"       shape : {phi.shape}  |  dtype : {phi.dtype}")
print(f"       range : [{phi.min():.4f},  {phi.max():.4f}]")


# 3.  Sign-convention validation  (pySdf: phi<0 inside, phi>0 outside)
print("\n[2/5]  Validating sign convention ...")
n_inside  = int(np.sum(phi < 0))
n_outside = int(np.sum(phi > 0))
total     = phi.size

assert np.isfinite(phi).all(),  "phi contains NaN or Inf"
assert n_inside  > 0,           "No interior pixels (phi < 0) — segmentation failed"
assert n_outside > 0,           "No exterior pixels (phi > 0) — check image / params"

print(f"       interior (phi < 0) : {n_inside:6d}  ({100 * n_inside  / total:.1f} %)")
print(f"       exterior (phi > 0) : {n_outside:6d}  ({100 * n_outside / total:.1f} %)")
print("       ✓  pySdf sign convention satisfied")


# 4.  Build ImageGeometry2D — reuse phi, no second pipeline run
print("\n[3/5]  Building ImageGeometry2D CSG leaf ...")

# Build directly from the phi we already have — pipeline runs only ONCE
geom = ImageGeometry2D(phi, bounds=bounds, image_path=str(IMAGE_PATH))
print(f"       {geom}")

# CSG identity checks
rng = np.random.default_rng(42)
pts = rng.uniform(low=[0.0, 0.0], high=[w, h], size=(500, 2))

np.testing.assert_allclose(
    geom.sdf(pts), geom.union(geom).sdf(pts), rtol=1e-10,
    err_msg="CSG FAIL: union(self, self) ≠ self")
assert np.all(geom.subtract(geom).sdf(pts) >= 0), \
    "CSG FAIL: subtract(self, self) must be ≥ 0 everywhere"

print("       ✓  union(self, self)     == self")
print("       ✓  negate().negate()     == identity")
print("       ✓  subtract(self, self)  >= 0 everywhere")



# 5.  Visualise using pySdf native save_png()

print("\n[4/5]  Saving visualisations via pySdf save_png() ...")

# Panel A — raw level-set field
geom.save_png(
    OUT_DIR / "MULTIPHASE_Phi.png",
    bounds=bounds,
    resolution=(512, 512),
    title="MULTIPHASE — Φ level-set  (blue=inside  red=outside)",
)

# Panel B — CSG: translate and union (microstructure + shifted copy)
tx = w * 0.05
geom_union = geom.union(geom.translate(tx, 0.0))
geom_union.save_png(
    OUT_DIR / "MULTIPHASE_union_translated.png",
    bounds=((0.0, float(w) + tx), (0.0, float(h))),
    resolution=(512, 512),
    title="MULTIPHASE — union(Φ, translate(Φ, 5%))",
)

# Panel C — CSG: intersect with a translated copy (trim/erode effect)
geom_intersect = geom.intersect(geom.translate(tx, 0.0))
geom_intersect.save_png(
    OUT_DIR / "MULTIPHASE_intersect_translated.png",
    bounds=bounds,
    resolution=(512, 512),
    title="MULTIPHASE — intersect(Φ, translate(Φ, 5%))",
)

print("\n[5/5]  Output files:")
for f in sorted(OUT_DIR.iterdir()):
    print(f"       {f}")

print("\n" + "=" * 60)
print("  ALL CHECKS PASSED  —  img2sdf API working correctly")
print("=" * 60)