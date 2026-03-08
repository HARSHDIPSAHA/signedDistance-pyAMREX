# pySdf Library Structure

A 2D and 3D Signed Distance Function library with STL mesh conversion and optional pyAMReX integration.

### Top-level

- `_sdf_common.py` — Shared math helpers (`vec2`/`vec3`, `length`, `dot`, `clamp`, boolean ops) used by all packages.
- `sdf2d/` — 2D geometry package.
- `sdf3d/` — 3D geometry package.
- `stl2sdf/` — STL mesh → SDF package (pure NumPy).
- `tests/` — pytest suite; no AMReX required (`test_amrex.py` skips automatically).
- `scripts/` — Gallery and plotfile rendering utilities (not part of the library API).
- `examples/` — Standalone runnable demos; outputs written to this folder.
- `gallery_2d.png`, `gallery_3d.png` — Pre-rendered shape galleries.

---

### `sdf2d/` contents

- `primitives.py` — NumPy implementations of all ~50 2D SDF formulas (`sdCircle`, `sdBox2D`, …) and `opTx2D`. Re-exports everything from `_sdf_common`. No AMReX dependency.
- `geometry.py` — All 2D geometry classes: `Circle2D`, `Box2D`, `Hexagon2D`, … Operators `|`, `-`, `/` compose shapes (union, subtraction, intersection). Transforms: `translate`, `rotate`, `scale`, `round`, `onion`. Every class wraps a lambda over `primitives` and exposes `sdf(p)`.
- `grid.py` — `sample_levelset_2d(geom, bounds, resolution)` → `ndarray` of shape `(ny, nx)`. Also provides `save_npy`.
- `amrex.py` — `MultiFabGrid2D`: a named grid context for AMReX `MultiFab` output. Construct once with `(geom, ba, dm)`. Use `shape.fill(grid)` to create + fill a MultiFab, or `shape.to_multifab(geom, ba, dm)` as a single-call convenience. Boolean ops on raw MultiFabs: `grid.union(a, b)`, `grid.subtract(base, cutter)`, `grid.intersect(a, b)`, `grid.negate(a)`. Requires `amrex.space2d`. Import-guarded so the module loads without AMReX.
- `__init__.py` — Re-exports all public symbols.

---

### `sdf3d/` contents

- `primitives.py` — NumPy implementations of all ~30 3D SDF formulas (`sdSphere`, `sdBox`, …), smooth boolean ops (`opSmoothUnion`, …), and space-warps (`opElongate`, `opRevolution`, `opExtrusion`, `opTwist`, …). Re-exports everything from `_sdf_common`. No AMReX dependency.
- `geometry.py` — All 3D geometry classes: `Sphere3D`, `Box3D`, `RoundBox3D`, `Cylinder3D`, `ConeExact3D`, `Torus3D`, … Operators `|`, `-`, `/` compose shapes (union, subtraction, intersection). Transforms: `translate`, `rotate_x/y/z`, `scale`, `elongate`, `round`, `onion`.
- `grid.py` — `sample_levelset_3d(geom, bounds, resolution)` → `ndarray` of shape `(nz, ny, nx)`. Also provides `save_npy`.
- `amrex.py` — `MultiFabGrid3D`: a named grid context for AMReX `MultiFab` output. Construct once with `(geom, ba, dm)`. Use `shape.fill(grid)` to create + fill a MultiFab, or `shape.to_multifab(geom, ba, dm)` as a single-call convenience. Boolean ops on raw MultiFabs: `grid.union(a, b)`, `grid.subtract(base, cutter)`, `grid.intersect(a, b)`, `grid.negate(a)`. Requires `amrex.space3d`. Import-guarded so the module loads without AMReX.
- `examples/` — High-level geometry assemblies:
  - `nato_stanag.py` — `NATOFragment(…)`: NATO STANAG fragmentation cylinder with conical nose.
  - `rocket_assembly.py` — `RocketAssembly(…)`: multi-part rocket with body, nose, and fins.
- `__init__.py` — Re-exports all public symbols.

---

### `stl2sdf/` contents

Converts binary or ASCII STL meshes into signed distance fields via pure NumPy.
Requires a **watertight** (closed, 2-manifold) mesh for correct sign determination.

- `_math.py` — Private internals:
  - `_stl_to_triangles(path)` → `(F, 3, 3)` float64 — binary + ASCII loader
  - `_triangles_to_sdf(points, triangles)` → `(N,)` — Ericson closest-point + Möller–Trumbore sign
- `geometry.py` — Public API:
  - `stl_to_geometry(path, *, ray_dir=None)` → `SDF3D` — load STL and return a composable geometry object
- `__init__.py` — Re-exports `stl_to_geometry`.

**Algorithms:**
- *Unsigned distance*: Christer Ericson's Voronoi-region closest-point (7 regions, O(F×N))
- *Sign*: Möller–Trumbore ray casting — odd hit count → inside (φ < 0)

---

### `tests/` contents

All tests pass with `pytest` and require only `numpy`. AMReX is not needed.

| File | What it tests |
|------|--------------|
| `test_sdf2d_lib.py` | Every function in `sdf2d/primitives.py` at analytically known points |
| `test_sdf3d_lib.py` | Every function in `sdf3d/primitives.py` at analytically known points |
| `test_sdf2d_geometry.py` | Every 2D geometry class: sign correctness, transforms, booleans |
| `test_sdf2d_grid.py` | `sample_levelset_2d` shape/sign, `save_npy` round-trip |
| `test_sdf3d_geometry.py` | Every 3D geometry class: sign correctness, transforms, booleans |
| `test_sdf3d_grid.py` | `sample_levelset_3d` shape/sign, `save_npy` round-trip |
| `test_complex.py` | `NATOFragment` and `RocketAssembly` (mock lib, no AMReX) |
| `test_stl2sdf.py` | `_stl_to_triangles` (binary + ASCII), `_triangles_to_sdf` (7 Voronoi regions, ray casting, sign), `stl_to_geometry` — all synthetic, no downloads |
| `test_amrex_2d.py` | `MultiFabGrid2D` — **skipped automatically without pyAMReX** |
| `test_amrex_3d.py` | `MultiFabGrid3D` — **skipped automatically without pyAMReX; run in isolation** |

```bash
uv run pytest tests/ -v
```

### `scripts/` contents

Rendering and visualization utilities. Not part of the library API; not installed.

- `gallery_2d.py` — Renders all `sdf2d` shapes on a single page (requires matplotlib).
- `gallery_3d.py` — Renders all `sdf3d` primitives (requires matplotlib + scikit-image).
- `render_surface_from_plotfile.py` — Renders an AMReX plotfile SDF=0 surface (requires pyAMReX + yt).

### `examples/` contents

Standalone runnable demos. Outputs (PNG, HTML, NPY) are written to `examples/`.

| File | Description |
|------|-------------|
| `sdf3d/union_example.py` | Two spheres joined with `\|` operator |
| `sdf3d/intersection_example.py` | Sphere–sphere intersection with `/` operator |
| `sdf3d/subtraction_example.py` | Sphere with spherical cavity via `-` operator |
| `sdf3d/elongation_example.py` | Sphere elongated into a capsule |
| `sdf3d/complex_example.py` | Chains all four operations, one PNG per step |
| `stl2sdf/nasa_shapes_demo.py` | Downloads 4 NASA meshes (Orion/CubeSat/wheel/Eros), saves Plotly HTML report |
| `stl2sdf/nasa_boolean_demo.py` | Boolean ops demo: mesh union/subtract with analytic sphere |

### Design

```
User parameters          STL file
      ↓                     ↓
Geometry classes      stl2sdf.stl_to_geometry
(sdf2d / sdf3d)             ↓
      └──────── SDF3D ───────┘
                    ↓
            SDF evaluation
            (primitives.py)
                    ↓
      Level-set field   φ(x, y[, z]) on a grid
                    ↓
      Output:  NumPy ndarray  OR  AMReX MultiFab
```

- **NumPy path** (no AMReX): `shape.to_array(bounds, resolution)` → `np.ndarray`
- **AMReX path**: `shape.fill(MultiFabGrid2D/3D)` or `shape.to_multifab(geom, ba, dm)` → `amr.MultiFab`
