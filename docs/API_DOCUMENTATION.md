1. [What is implemented](#what-is-implemented)
2. [Quick start](#quick-start)
3. [2D API — `sdf2d`](#2d-api--sdf2d)
4. [3D API — `sdf3d`](#3d-api--sdf3d)
5. [STL → SDF — `stl2sdf`](#stl--sdf--stl2sdf)
6. [AMReX integration](#amrex-integration)
7. [Low-level math — `sdf2d.primitives` / `sdf3d.primitives`](#low-level-math--sdf2dprimitives--sdf3dprimitives)
8. [Tips](#tips)

## What is implemented

| Feature | Where |
|---------|-------|
| 2D SDF primitives | `sdf2d.primitives`, `sdf2d.geometry` |
| 3D SDF primitives + warps + smooth ops | `sdf3d.primitives`, `sdf3d.geometry` |
| Boolean ops (Union, Intersection, Subtraction) | both packages |
| Smooth boolean ops (smooth union/subtraction/intersection) | `sdf3d.primitives` |
| Transforms (translate, rotate, scale, round, onion, elongate) | both packages |
| Grid sampling to NumPy arrays | `sdf2d.grid`, `sdf3d.grid` |
| STL mesh → SDF grid | `stl2sdf` |
| Image → SDF via Chan-Vese segmentation | `img2sdf` |
| AMReX MultiFab output | `sdf2d.amrex`, `sdf3d.amrex`, `img2sdf.amrex` |
| Complex assemblies | `sdf3d.examples` (`NATOFragment`, `RocketAssembly`) |
| Comprehensive unit tests | `tests/` |

## Quick start

### NumPy mode (no AMReX)

```python
from sdf3d import Sphere3D, Box3D
import numpy as np

sphere = Sphere3D(radius=0.3)
box    = Box3D(half_size=(0.2, 0.2, 0.2)).translate(0.4, 0.0, 0.0)
shape  = sphere | box              # union operator

phi = shape.to_numpy(bounds=((-1,1),(-1,1),(-1,1)), resolution=(64,64,64))
# phi.shape == (64, 64, 64);  phi < 0 inside, phi > 0 outside
```

### AMReX mode

```python
import amrex.space3d as amr
from sdf3d import Sphere3D, Box3D, MultiFabGrid3D

amr.initialize([])
try:
    real_box = amr.RealBox([-1,-1,-1], [1,1,1])
    domain   = amr.Box(amr.IntVect(0,0,0), amr.IntVect(63,63,63))
    geom     = amr.Geometry(domain, real_box, 0, [0,0,0])
    ba       = amr.BoxArray(domain); ba.max_size(32)
    dm       = amr.DistributionMapping(ba)

    # Grid context — use it for all fills and boolean ops
    grid = MultiFabGrid3D(geom, ba, dm)
    mf_a = Sphere3D(0.3).to_multifab(grid)
    mf_b = Box3D((0.2, 0.2, 0.2)).to_multifab(grid)
    mf_u = grid.union(mf_a, mf_b)
finally:
    amr.finalize()
```

## 2D API — `sdf2d`

### Base class

```python
from sdf2d import SDF2D
```

`SDF2D(func)` wraps any callable `func(p: ndarray) -> ndarray` where `p` has shape `(..., 2)`.

#### Methods on every geometry

| Method | Signature | Returns |
|--------|-----------|---------|
| `sdf` | `sdf(p)` | signed distance values |
| `translate` | `translate(tx, ty)` | `SDF2D` |
| `rotate` | `rotate(angle_rad)` | `SDF2D` |
| `scale` | `scale(factor)` | `SDF2D` |
| `round` | `round(radius)` | `SDF2D` |
| `onion` | `onion(thickness)` | `SDF2D` |
| `union` | `union(other)` | `SDF2D` |
| `subtract` | `subtract(other)` | `SDF2D` |
| `intersect` | `intersect(other)` | `SDF2D` |

### Primitive shapes

```python
from sdf2d import (
    Circle2D, Box2D, RoundedBox2D, OrientedBox2D,
    Segment2D, Rhombus2D, Trapezoid2D, Parallelogram2D,
    EquilateralTriangle2D, TriangleIsosceles2D, Triangle2D,
    UnevenCapsule2D,
    Pentagon2D, Hexagon2D, Octagon2D, NGon2D,
    Hexagram2D, Star2D,
    Pie2D, CutDisk2D, Arc2D, Ring2D, Horseshoe2D,
    Vesica2D, Moon2D, RoundedCross2D, Egg2D, Heart2D,
    Cross2D, RoundedX2D,
    Polygon2D, Ellipse2D, Parabola2D, ParabolaSegment2D,
    Bezier2D, BlobbyCross2D, Tunnel2D, Stairs2D,
    QuadraticCircle2D, Hyperbola2D,
)
```

| Class | Constructor |
|-------|-------------|
| `Circle2D` | `Circle2D(radius)` |
| `Box2D` | `Box2D(half_size)` — `half_size = (hx, hy)` |
| `RoundedBox2D` | `RoundedBox2D(half_size, radius)` |
| `OrientedBox2D` | `OrientedBox2D(point_a, point_b, width)` |
| `Segment2D` | `Segment2D(a, b)` — unsigned distance to segment |
| `Rhombus2D` | `Rhombus2D(half_size)` |
| `Trapezoid2D` | `Trapezoid2D(r1, r2, height)` |
| `Parallelogram2D` | `Parallelogram2D(width, height, skew)` |
| `EquilateralTriangle2D` | `EquilateralTriangle2D(size)` |
| `TriangleIsosceles2D` | `TriangleIsosceles2D(width, height)` |
| `Triangle2D` | `Triangle2D(p0, p1, p2)` |
| `UnevenCapsule2D` | `UnevenCapsule2D(r1, r2, height)` |
| `Pentagon2D` | `Pentagon2D(radius)` |
| `Hexagon2D` | `Hexagon2D(radius)` — `radius` is the inradius (flat-face distance) |
| `Octagon2D` | `Octagon2D(radius)` |
| `NGon2D` | `NGon2D(radius, n_sides)` |
| `Hexagram2D` | `Hexagram2D(radius)` |
| `Star2D` | `Star2D(radius, n_points, m)` — `2 ≤ m ≤ n_points`; m=2 sharpest, m=n_points → regular polygon |
| `Pie2D` | `Pie2D(sin_cos, radius)` |
| `CutDisk2D` | `CutDisk2D(radius, cut_height)` — the circular cap above `y = cut_height` |
| `Arc2D` | `Arc2D(sin_cos, radius, thickness)` |
| `Ring2D` | `Ring2D(inner_radius, outer_radius)` |
| `Horseshoe2D` | `Horseshoe2D(sin_cos, radius, widths)` |
| `Vesica2D` | `Vesica2D(radius, offset)` |
| `Moon2D` | `Moon2D(d, ra, rb)` |
| `RoundedCross2D` | `RoundedCross2D(size)` |
| `Egg2D` | `Egg2D(ra, rb)` |
| `Heart2D` | `Heart2D()` |
| `Cross2D` | `Cross2D(size, r)` |
| `RoundedX2D` | `RoundedX2D(size, r)` |
| `Polygon2D` | `Polygon2D(vertices)` — list of `[x, y]` |
| `Ellipse2D` | `Ellipse2D(radii)` — `radii = (rx, ry)` |
| `Parabola2D` | `Parabola2D(k)` |
| `ParabolaSegment2D` | `ParabolaSegment2D(width, height)` |
| `Bezier2D` | `Bezier2D(a, b, c)` — quadratic Bézier, unsigned |
| `BlobbyCross2D` | `BlobbyCross2D(size)` |
| `Tunnel2D` | `Tunnel2D(size)` — `size = (wx, wy)` |
| `Stairs2D` | `Stairs2D(size, n)` — `size = (tread, rise)` |
| `QuadraticCircle2D` | `QuadraticCircle2D()` |
| `Hyperbola2D` | `Hyperbola2D(k, he)` |

### Boolean operations

```python
u = a | b        # union        — SDF = min(a, b)
i = a / b        # intersection — SDF = max(a, b)
s = a - b        # subtraction  — SDF = max(-b, a), reads "a minus b"

# Equivalent method syntax:
u = a.union(b)
i = a.intersect(b)
s = a.subtract(b)
```

Chain multiple shapes with `functools.reduce` or repeated operators:

```python
import functools, operator
shape = functools.reduce(operator.or_, [circle_a, circle_b, circle_c])
```

### Grid sampling

```python
from sdf2d import sample_levelset_2d, save_npy

phi = sample_levelset_2d(
    geom,                          # SDF2D
    bounds=((-1,1), (-1,1)),       # ((xlo,xhi), (ylo,yhi))
    resolution=(nx, ny),
)
# phi.shape == (ny, nx)  — y-first

save_npy("output/levelset.npy", phi)  # creates parent dirs automatically
```

### AMReX (2D)

`MultiFabGrid2D` is a **named grid context**: construct it once with the AMReX
grid layout; all fill and boolean operations go through it.

```python
from sdf2d import MultiFabGrid2D, Circle2D, Box2D
import amrex.space2d as amr

# Grid context — use it for all fills and boolean ops
grid   = MultiFabGrid2D(geom, ba, dm)
mf_c   = Circle2D(0.3).to_multifab(grid)
mf_b   = Box2D((0.2, 0.3)).to_multifab(grid)

# Boolean operations on MultiFabs (element-wise on already-filled grids)
mf_u = grid.union(mf_c, mf_b)          # min(c, b)
mf_s = grid.subtract(mf_b, mf_c)       # box with circle carved out
mf_i = grid.intersect(mf_c, mf_b)      # max(c, b)
mf_n = grid.negate(mf_c)               # flip sign
```

## 3D API — `sdf3d`

### Base class

```python
from sdf3d import SDF3D
```

`SDF3D(func)` wraps any callable `func(p: ndarray) -> ndarray` where `p` has shape `(..., 3)`.

#### Methods on every geometry

| Method | Signature | Returns |
|--------|-----------|---------|
| `sdf` | `sdf(p)` | signed distance values |
| `translate` | `translate(tx, ty, tz)` | `SDF3D` |
| `rotate_x` | `rotate_x(angle_rad)` | `SDF3D` |
| `rotate_y` | `rotate_y(angle_rad)` | `SDF3D` |
| `rotate_z` | `rotate_z(angle_rad)` | `SDF3D` |
| `scale` | `scale(factor)` | `SDF3D` |
| `elongate` | `elongate(hx, hy, hz)` | `SDF3D` |
| `round` | `round(radius)` | `SDF3D` |
| `onion` | `onion(thickness)` | `SDF3D` |
| `union` | `union(other)` | `SDF3D` |
| `subtract` | `subtract(other)` | `SDF3D` |
| `intersect` | `intersect(other)` | `SDF3D` |

### Primitive shapes

```python
from sdf3d import Sphere3D, Box3D, RoundBox3D, Cylinder3D, ConeExact3D, Torus3D
```

| Class | Constructor | Notes |
|-------|-------------|-------|
| `Sphere3D` | `Sphere3D(radius)` | Exact SDF |
| `Box3D` | `Box3D(half_size)` — `(hx, hy, hz)` | Exact SDF |
| `RoundBox3D` | `RoundBox3D(half_size, radius)` | Rounded corners only; flat faces stay at `half_size` |
| `Cylinder3D` | `Cylinder3D(axis_offset, radius)` | Infinite cylinder along Z |
| `ConeExact3D` | `ConeExact3D(sincos, height)` — `[sin θ, cos θ]` | Finite cone |
| `Torus3D` | `Torus3D(radii)` — `(R, r)` | Major/minor radii; lies in XZ plane |

### Boolean operations

```python
u = a | b        # union        — SDF = min(a, b)
i = a / b        # intersection — SDF = max(a, b)
s = a - b        # subtraction  — SDF = max(-b, a), reads "a minus b"

# Equivalent method syntax:
u = a.union(b)
i = a.intersect(b)
s = a.subtract(b)
```

Chain multiple shapes:

```python
import functools, operator
shape = functools.reduce(operator.or_, [sphere_a, sphere_b, sphere_c])
```

### Grid sampling

```python
from sdf3d import sample_levelset_3d, save_npy

phi = sample_levelset_3d(
    geom,
    bounds=((-1,1), (-1,1), (-1,1)),   # ((xlo,xhi), (ylo,yhi), (zlo,zhi))
    resolution=(nx, ny, nz),
)
# phi.shape == (nz, ny, nx)  — z-first

save_npy("output/levelset.npy", phi)
```

### Complex assemblies

```python
from sdf3d.examples import NATOFragment, RocketAssembly
```

Both return an `SDF3D` object — composable with all the usual operators and methods.

#### `NATOFragment`

```python
geom = NATOFragment(
    diameter=14.30e-3,    # fragment diameter (m)
    L_over_D=1.09,        # length-to-diameter ratio
    cone_angle_deg=20.0,  # nose cone half-angle (degrees)
)
```

#### `RocketAssembly`

```python
geom = RocketAssembly(
    body_radius=0.15,
    L_extra=0.40,
    nose_len=0.25,
    fin_span=0.12,
    fin_height=0.18,
    fin_thickness=0.03,
    n_fins=4,
)
```

### AMReX (3D)

`MultiFabGrid3D` is a **named grid context**: construct it once with the AMReX
grid layout; all fill and boolean operations go through it.

```python
from sdf3d import MultiFabGrid3D, Sphere3D, Box3D
import amrex.space3d as amr

# Grid context — use it for all fills and boolean ops
grid   = MultiFabGrid3D(geom, ba, dm)
mf_s   = Sphere3D(0.3).to_multifab(grid)
mf_b   = Box3D((0.2, 0.2, 0.2)).to_multifab(grid)

# Boolean operations on MultiFabs (element-wise on already-filled grids)
mf_u = grid.union(mf_s, mf_b)           # min(s, b)
mf_s = grid.subtract(mf_b, mf_s)        # box with sphere carved out
mf_i = grid.intersect(mf_s, mf_b)       # max(s, b)
mf_n = grid.negate(mf_s)                # flip sign
```

## STL → SDF — `stl2sdf`

Converts binary or ASCII STL meshes into an `SDF3D` object using pure NumPy.
Requires a **watertight** (closed, 2-manifold) mesh for correct sign determination.
Complexity is O(F × N) — no BVH acceleration.

```python
from stl2sdf import stl_to_geometry
```

### `stl_to_geometry`

```python
from stl2sdf import stl_to_geometry
from sdf3d import Sphere3D
from sdf3d.grid import sample_levelset_3d

wheel = stl_to_geometry("mars_wheel.stl")
# Returns an SDF3D — compatible with all analytic primitives

# Combine with analytic shapes
hollowed = wheel.subtract(Sphere3D(0.3))
phi = sample_levelset_3d(hollowed, bounds=((-1,1),(-1,1),(-1,1)), resolution=(32,32,32))
# phi.shape == (32, 32, 32)
```

Optional `ray_dir` keyword argument overrides the default irrational ray direction used
for sign determination. Avoid axis-aligned directions with axis-aligned meshes.

The returned `SDF3D` supports all the usual methods: `.translate()`, `.rotate_x/y/z()`,
`.scale()`, `.union()`, `.subtract()`, `.intersect()`, `.round()`, `.onion()`.

### Demos

```bash
uv run python examples/stl2sdf/nasa_shapes_demo.py           # 4 NASA meshes, res=20
uv run python examples/stl2sdf/nasa_shapes_demo.py --res 30  # higher quality
uv run python examples/stl2sdf/nasa_shapes_demo.py --skip-eros  # skip 200K-tri Eros
uv run python examples/stl2sdf/nasa_boolean_demo.py          # mesh union/subtract with sphere
```

## Image → SDF — `img2sdf`

Converts microscopy/X-ray images into a `Geometry2D` object via the uSCMAN
Chan-Vese segmentation pipeline (pure NumPy).

> **Sign convention:** uSCMAN outputs φ > 0 inside; pySdf uses φ < 0 inside.
> The negation is applied automatically — no manual adjustment needed.

```python
from img2sdf import image_to_geometry_2d, image_to_levelset_2d, ImageGeometry2D
```

### `ImageGeometry2D`

```python
ImageGeometry2D(phi, bounds, image_path=None)
```

A `Geometry2D` subclass backed by a bilinearly interpolated level-set field.
Supports all `Geometry2D` methods: `.translate()`, `.rotate()`, `.scale()`,
`.union()`, `.subtract()`, `.intersect()`, `.round()`, `.onion()`.

| Parameter | Type | Description |
|-----------|------|-------------|
| `phi` | `ndarray (ny, nx)` | Level-set field; φ < 0 inside, φ > 0 outside |
| `bounds` | `((x0,x1),(y0,y1))` | Physical domain extents |
| `image_path` | `str \| None` | Source path (stored for reference only) |

### `image_to_levelset_2d`

```python
phi = image_to_levelset_2d(image_path, params, *, levelset_index=0)
# Returns ndarray (ny, nx); phi < 0 inside
```

Runs the full uSCMAN pipeline and returns the raw level-set array.
`params` is the uSCMAN JSON config dict (keys: `"Image Properties"`,
`"Preprocessing Properties"`, `"Segmentation"`). `levelset_index` selects
which phase in multiphase segmentation (0 or 1).

### `image_to_geometry_2d`

```python
geom = image_to_geometry_2d(image_path, params, bounds=None, *, levelset_index=0)
# Returns ImageGeometry2D ready for CSG composition
```

Same as `image_to_levelset_2d` but returns an `ImageGeometry2D`.
If `bounds` is `None`, the image is mapped to pixel coordinates `((0, nx), (0, ny))`.

```python
import json
from img2sdf import image_to_geometry_2d
from sdf2d import Circle2D, sample_levelset_2d

params = json.load(open("HEDS/HEDS.json"))
geom = image_to_geometry_2d("HEDS/HEDS.jpg", params, bounds=((-1, 1), (-1, 1)))

# Compose with analytic shapes
scene = geom.union(Circle2D(radius=0.1).translate(0.5, 0.0))
phi = sample_levelset_2d(scene, bounds=((-1, 1), (-1, 1)), resolution=(256, 256))
```

### `ImageExtruded3D`

```python
from img2sdf.image_leaf import ImageExtruded3D

geom3d = ImageExtruded3D(hdf5_path, dataset_path, physical_size_xy, thickness_z)
```

Reads a 2D Chan-Vese SDF from a uSCMAN HDF5 results file and extrudes it into
a `Geometry3D` node. Supports all `Geometry3D` methods.

| Parameter | Description |
|-----------|-------------|
| `hdf5_path` | Path to the uSCMAN results HDF5 file |
| `dataset_path` | HDF5 group path containing `Phi1`, `I`, `J` datasets |
| `physical_size_xy` | `(width, height)` physical extent in world units |
| `thickness_z` | Extrusion depth along the Z axis |

### `SDFLibraryImg2D` (AMReX)

```python
from img2sdf import SDFLibraryImg2D
import amrex.space2d as amr

lib = SDFLibraryImg2D(geom, ba, dm)
mf  = lib.from_image(image_path, params, bounds=((x0,x1),(y0,y1)))
# mf is an amr.MultiFab
```

Runs the NumPy Chan-Vese pipeline first, then evaluates the resulting
`ImageGeometry2D` at each AMReX cell centre via `SDFLibrary2D.from_geometry()`.
The image-derived SDF is available on the AMReX grid without porting Chan-Vese
to AMReX — the same bridge used by all `Geometry2D` subclasses.

## AMReX integration

`MultiFabGrid2D` / `MultiFabGrid3D` are **named grid contexts**: construct one
with the AMReX grid layout (`geom`, `ba`, `dm`) and it holds that layout for
all fill and boolean operations.

```python
import amrex.space3d as amr   # or amrex.space2d for 2D

amr.initialize([])
try:
    real_box = amr.RealBox([xlo,ylo,zlo], [xhi,yhi,zhi])
    domain   = amr.Box(amr.IntVect(0,0,0), amr.IntVect(nx-1,ny-1,nz-1))
    geom     = amr.Geometry(domain, real_box, 0, [0,0,0])
    ba       = amr.BoxArray(domain); ba.max_size(32)
    dm       = amr.DistributionMapping(ba)

    from sdf3d import Sphere3D, Box3D, MultiFabGrid3D

    # Grid context — use it for all fills and boolean ops
    grid = MultiFabGrid3D(geom, ba, dm)
    mf_s = Sphere3D(0.3).to_multifab(grid)
    mf_b = Box3D((0.2, 0.2, 0.2)).to_multifab(grid)
    mf_u = grid.union(mf_s, mf_b)

    varnames = amr.Vector_string(["phi"])
    amr.write_single_level_plotfile("output/levelset", mf_u, varnames, geom, 0.0, 0)
finally:
    amr.finalize()
```

### Reading MultiFab values

```python
for mfi in mf:
    arr = mf.array(mfi).to_numpy()
    phi = arr[..., 0]   # shape (ny, nx[, nz]) — one component, no ghost cells
```

See [INSTALLATION.md](INSTALLATION.md) for all three pyAMReX installation methods.

## Low-level math — `sdf2d.primitives` / `sdf3d.primitives`

```python
from sdf3d import primitives as sdf3d
from sdf2d import primitives as sdf2d

p = np.array([0.0, 0.0, 0.0])
d = sdf3d.sdSphere(p, 0.3)    # == -0.3

q = np.array([0.0, 0.0])
d = sdf2d.sdCircle(q, 0.3)    # == -0.3
```

Functions follow `sd<Shape>` (signed) or `ud<Shape>` (unsigned); operators use `op` prefix.

**3D primitives:** `sdSphere`, `sdBox`, `sdRoundBox`, `sdBoxFrame`, `sdTorus`, `sdCappedTorus`, `sdLink`, `sdCylinder`, `sdConeExact`, `sdConeBound`, `sdConeInfinite`, `sdPlane`, `sdHexPrism`, `sdTriPrism`, `sdCapsule`, `sdVerticalCapsule`, `sdCappedCylinder`, `sdCappedCylinderSegment`, `sdRoundedCylinder`, `sdCappedCone`, `sdCappedConeSegment`, `sdSolidAngle`, `sdCutSphere`, `sdCutHollowSphere`, `sdDeathStar`, `sdRoundCone`, `sdRoundConeSegment`, `sdEllipsoid`, `sdVesicaSegment`, `sdRhombus`, `sdOctahedronExact`, `sdOctahedronBound`, `sdPyramid`

**3D unsigned:** `udTriangle`, `udQuad`

**2D primitives:** `sdCircle`, `sdBox2D`, `sdRoundedBox2D`, `sdOrientedBox2D`, `sdSegment`, `sdRhombus2D`, `sdTrapezoid2D`, `sdParallelogram2D`, `sdEquilateralTriangle`, `sdTriangleIsosceles`, `sdTriangle2D`, `sdUnevenCapsule2D`, `sdPentagon`, `sdHexagon`, `sdOctagon`, `sdNGon`, `sdHexagram`, `sdStar`, `sdPie2D`, `sdCutDisk`, `sdArc`, `sdRing`, `sdHorseshoe`, `sdVesica2D`, `sdMoon`, `sdRoundedCross`, `sdEgg`, `sdHeart`, `sdCross`, `sdRoundedX`, `sdPolygon`, `sdEllipse2D`, `sdParabola`, `sdParabolaSegment`, `sdBezier`, `sdBlobbyCross`, `sdTunnel`, `sdStairs`, `sdQuadraticCircle`, `sdHyperbola`

**Boolean:** `opUnion`, `opSubtraction`, `opIntersection`, `opSmoothUnion`, `opSmoothSubtraction`, `opSmoothIntersection`

**Warp/space:** `opRound`, `opOnion`, `opElongate1`, `opElongate2`, `opRevolution`, `opExtrusion`, `opTwist`, `opCheapBend`, `opTx`, `opTx2D`, `opScale`, `opSymX`, `opSymXZ`, `opRepetition`, `opLimitedRepetition`, `opDisplace`

## Tips

- **Sign convention:** `phi < 0` inside, `phi = 0` on surface, `phi > 0` outside.
- **Grid layout:** `sample_levelset_3d` returns shape `(nz, ny, nx)`. Access as `phi[iz, iy, ix]`.
- **Subtraction argument order:** `a - b` / `a.subtract(b)` — `a` is the base, `b` is what gets cut away. `MultiFabGrid3D.subtract(base, cutter)` follows the same order.
- **AMReX initialize/finalize:** Always wrap in `try/finally` with `amr.finalize()`.
- **Chaining transforms:**
  ```python
  shape = Sphere3D(0.3).translate(0.5, 0, 0).rotate_z(np.pi/4).scale(1.2)
  ```
- **stl2sdf resolution:** O(F×N) — use `--res 20` for drafts, `--res 30`+ for quality renders.
- **Watertight check:** Verify your STL has no boundary edges before using `stl_to_geometry`. See the troubleshooting section in [INSTALLATION.md](INSTALLATION.md).
