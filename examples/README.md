# pySdf — Examples

Standalone scripts demonstrating the `sdf3d`, `stl2sdf`, and `img2sdf` APIs.
**No AMReX required** — all geometry is evaluated in pure NumPy.

Output files (PNG, HTML, NPY) are written to `examples/*/output/` (gitignored).

```bash
# Run from the repo root:
python examples/sdf3d/union_example.py
python examples/sdf3d/intersection_example.py
python examples/sdf3d/subtraction_example.py
python examples/sdf3d/elongation_example.py
python examples/sdf3d/complex_example.py
python examples/sdf3d/nato_stanag_4496_test.py
python examples/stl2sdf/military_shapes_demo.py
python examples/img2sdf/Example_heds.py
python examples/img2sdf/MULTIPHASE_TEST.py
python examples/img2sdf/morphometry_3d_demo.py
python examples/img2sdf/img2sdf_3d_showcase.py
python examples/img2sdf/volume_3d_example.py
```

---

## sdf3d examples

### `sdf3d/union_example.py`
Two overlapping spheres combined with `Union3D`.

### `sdf3d/intersection_example.py`
Intersection of two overlapping spheres via `Intersection3D`.

### `sdf3d/subtraction_example.py`
Sphere with a spherical cavity cut using `Subtraction3D(base, cutter)`.

### `sdf3d/elongation_example.py`
Sphere elongated along X into a capsule with `.elongate(h, 0, 0)`.

### `sdf3d/complex_example.py`
Chains all four boolean operations in sequence, saving a PNG per step:

| File | Description |
|------|-------------|
| `complex_example_step1.png` | Base box |
| `complex_example_step2.png` | Elongated sphere (capsule) |
| `complex_example_step3.png` | Union: box ∪ capsule |
| `complex_example_step4.png` | Intersection: rounded top |
| `complex_example_final.png` | Subtraction: central cavity |

### `sdf3d/nato_stanag_4496_test.py`
NATO STANAG-4496 fragment impact scene.

---

## stl2sdf examples

### `stl2sdf/military_shapes_demo.py`
Loads several STL meshes and renders an interactive Plotly HTML report.

---

## img2sdf examples

> **Requirements:** `pip install scipy scikit-image plotly`

### `img2sdf/morphometry_3d_demo.py` ⭐

**3D Morphometry showcase** — demonstrates `compute_morphometry_3d` on four
synthetic shapes (sphere, cube, elongated box, torus).

Saves `output/morphometry_3d_demo.html` with:
- Four interactive 3D isosurface viewers (one per shape)
- Bar charts comparing volume, surface area, and sphericity vs analytic references
- Sphericity vs relative-volume scatter plot
- Formula reference (`ψ = π^(1/3)(6V)^(2/3) / A`) and results table

```bash
python examples/img2sdf/morphometry_3d_demo.py
# Output: examples/img2sdf/output/morphometry_3d_demo.html
```

| Shape | Sphericity ψ | Notes |
|-------|-------------|-------|
| Sphere (r=20) | ≈ 0.998 | Most compact |
| Cube (h=16) | ≈ 0.756 | Less spherical |
| Elongated box | ≈ 0.672 | Even less |
| Torus (R=18, r=6) | ≈ 0.610 | Least spherical |

### `img2sdf/img2sdf_3d_showcase.py` ⭐

**Complete img2sdf 3D pipeline walkthrough** — covers every API in one script.

Steps demonstrated:
1. Synthetic 3D volume with Gaussian noise
2. `chan_vese_3d()` — low-level segmentation
3. Sign-convention validation (pySdf: φ < 0 inside)
4. `ImageGeometry3D` — construction + trilinear `sdf()` evaluation
5. CSG `union`, `subtract`, `intersect` with `Sphere3D` / `Box3D`
6. `negate()` — sign-convention flip
7. `compute_morphometry_3d` on all four CSG variants
8. `volume_to_levelset_3d` — high-level pipeline
9. `volume_to_geometry_3d` — direct `ImageGeometry3D` shortcut
10. Save/reload `.npy`

Saves `output/img2sdf_3d_showcase.html` with:
- Four 3D isosurface viewers (original + three CSG variants)
- Morphometry bar chart comparison
- API code snippets for each feature

```bash
python examples/img2sdf/img2sdf_3d_showcase.py
# Outputs: examples/img2sdf/output/img2sdf_3d_showcase.html
#          examples/img2sdf/output/showcase_levelset.npy
```

### `img2sdf/volume_3d_example.py`
Simple 3D Chan-Vese example: synthetic 64³ sphere → segment → morphometrics → save `.npy`.

```bash
python examples/img2sdf/volume_3d_example.py
# Output: examples/img2sdf/output/sphere_sdf.npy
```

### `img2sdf/Example_heds.py`
Loads the HEDS microscopy image, runs Chan-Vese segmentation, and demonstrates CSG
composition with an analytic circle.

```bash
python examples/img2sdf/Example_heds.py
```

### `img2sdf/MULTIPHASE_TEST.py`
Runs multiphase Chan-Vese segmentation on a synthetic test image.

```bash
python examples/img2sdf/MULTIPHASE_TEST.py
```

---

## SDF sign convention

| Value | Meaning |
|-------|---------|
| φ < 0 | inside the solid |
| φ = 0 | on the surface |
| φ > 0 | outside the solid |

> **img2sdf note:** uSCMAN outputs φ > 0 inside; pySdf negates automatically.
