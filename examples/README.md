# pySdf — Examples

Standalone scripts demonstrating the `sdf3d`, `stl2sdf`, and `img2sdf` APIs.
**No AMReX required** — all geometry is evaluated in pure NumPy.

Output files (PNG, HTML, NPY) are written to **this folder** (`examples/`).

```bash
# Run from the repo root:
uv run python examples/sdf3d/union_example.py
uv run python examples/sdf3d/intersection_example.py
uv run python examples/sdf3d/subtraction_example.py
uv run python examples/sdf3d/elongation_example.py
uv run python examples/sdf3d/complex_example.py
uv run python examples/sdf3d/nato_stanag_4496_test.py
uv run python examples/stl2sdf/military_shapes_demo.py
uv run python examples/img2sdf/Example_heds.py
uv run python examples/img2sdf/MULTIPHASE_TEST.py
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

### `img2sdf/Example_heds.py`
Loads the HEDS microscopy image, runs Chan-Vese segmentation via the `img2sdf`
pipeline, and demonstrates CSG composition with an analytic circle:

| File | Description |
|------|-------------|
| `outputs/HEDS_phi.png` | Raw level-set field from segmentation |
| `outputs/HEDS_result.png` | Segmented image geometry |
| `outputs/HEDS_union_translated.png` | Image SDF ∪ translated circle |
| `outputs/HEDS_intersect_translated.png` | Image SDF ∩ translated circle |

```bash
uv run python examples/img2sdf/Example_heds.py
```

### `img2sdf/MULTIPHASE_TEST.py`
Runs multiphase Chan-Vese segmentation on a synthetic test image.

| File | Description |
|------|-------------|
| `outputs/MULTIPHASE_Phi.png` | Multiphase level-set field |
| `outputs/MULTIPHASE_union_translated.png` | Phase 0 SDF ∪ translated circle |
| `outputs/MULTIPHASE_intersect_translated.png` | Phase 0 SDF ∩ translated circle |

```bash
uv run python examples/img2sdf/MULTIPHASE_TEST.py
```

---

## SDF sign convention

| Value | Meaning |
|-------|---------|
| φ < 0 | inside the solid |
| φ = 0 | on the surface |
| φ > 0 | outside the solid |

> **img2sdf note:** uSCMAN outputs φ > 0 inside; pySdf negates automatically.

