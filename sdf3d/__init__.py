"""
sdf3d — 3D Signed Distance Function Library
============================================

A library for creating and composing 3D signed distance fields (SDFs),
based on Inigo Quilez's distance function collection.

Implemented features
--------------------
- Primitive shapes: Sphere, Box, RoundBox, Cylinder, ConeExact, Torus
- Boolean operations: ``|`` (union), ``-`` (subtraction), ``/`` (intersection)
- Transforms: translate, rotate_x/y/z, scale, elongate, round, onion
- Grid sampling: :meth:`SDF3D.to_numpy`
- AMReX MultiFab output: :class:`MultiFabGrid3D` (requires pyAMReX 3-D build)
- Example assemblies: :class:`~sdf3d.examples.NATOFragment`,
  :class:`~sdf3d.examples.RocketAssembly`

Quick start
-----------

NumPy mode (no AMReX required)::

    from sdf3d import Sphere3D, Box3D

    sphere = Sphere3D(radius=0.3)
    box    = Box3D(half_size=(0.2, 0.2, 0.2)).translate(0.4, 0.0, 0.0)
    shape  = sphere | box              # union operator

    phi = shape.to_numpy(bounds=((-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0)), resolution=(64, 64, 64))

AMReX mode::

    import amrex.space3d as amr
    from sdf3d import MultiFabGrid3D

    amr.initialize([])
    # ... set up geom, ba, dm ...
    grid     = MultiFabGrid3D(geom, ba, dm)
    mf       = Sphere3D(0.3).to_multifab(grid)
    amr.finalize()
"""

from .geometry import (
    SDF3D,
    Sphere3D,
    Box3D,
    RoundBox3D,
    Cylinder3D,
    ConeExact3D,
    Torus3D,
    save_plotly_html_grid,
)
from .geometry import save_npy
from .amrex import MultiFabGrid3D
from .examples import NATOFragment, RocketAssembly
from .distributions import (
    generate_centers_random,
    generate_centers_inline,
    generate_centers_staggered,
    distribute_shape,
)
from .metamaterials import (
    Gyroid3D,
    SchwarzP3D,
    SchwarzD3D,
    Neovius3D,
    BCCLattice3D,
    FCCLattice3D,
)

__version__ = "0.2.0"

__all__ = [
    # Base
    "SDF3D",

    # Primitives
    "Sphere3D",
    "Box3D",
    "RoundBox3D",
    "Cylinder3D",
    "ConeExact3D",
    "Torus3D",

    # Grid utilities
    "save_npy",

    # AMReX integration
    "MultiFabGrid3D",

    # Complex assemblies
    "NATOFragment",
    "RocketAssembly",

    # Module-level visualisation
    "save_plotly_html_grid",

    # Distributions
    "generate_centers_random",
    "generate_centers_inline",
    "generate_centers_staggered",
    "distribute_shape",

    # Metamaterials / TPMS
    "Gyroid3D",
    "SchwarzP3D",
    "SchwarzD3D",
    "Neovius3D",
    "BCCLattice3D",
    "FCCLattice3D",
]
