"""
sdf3d — 3D Signed Distance Function Library
============================================

A library for creating and composing 3D signed distance fields (SDFs),
based on Inigo Quilez's distance function collection.

Implemented features
--------------------
- Primitive shapes: Sphere, Box, RoundBox, Cylinder, ConeExact, Torus
- Boolean operations: Union, Intersection, Subtraction
- Transforms: translate, rotate_x/y/z, scale, elongate, round, onion
- Grid sampling: :meth:`SDF3D.to_array`
- AMReX MultiFab output: :class:`SDFMultiFab3D` (requires pyAMReX 3-D build)
- Example assemblies: :class:`~sdf3d.examples.NATOFragment`,
  :class:`~sdf3d.examples.RocketAssembly`

Quick start
-----------

NumPy mode (no AMReX required)::

    from sdf3d import Sphere3D, Box3D, Union3D
    import numpy as np

    sphere = Sphere3D(radius=0.3)
    box    = Box3D(half_size=(0.2, 0.2, 0.2)).translate(0.4, 0.0, 0.0)
    shape  = Union3D(sphere, box)

    bounds     = ((-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0))
    resolution = (64, 64, 64)
    phi = shape.to_array(bounds, resolution)

AMReX mode::

    import amrex.space3d as amr

    amr.initialize([])
    # ... set up geom, ba, dm ...
    levelset = Sphere3D(0.3).to_multifab(geom, ba, dm)
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
    Union3D,
    Intersection3D,
    Subtraction3D,
    save_plotly_html_grid,
)
from .geometry import save_npy
from .amrex import SDFMultiFab3D
from .examples import NATOFragment, RocketAssembly

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

    # Boolean operations
    "Union3D",
    "Intersection3D",
    "Subtraction3D",

    # Grid utilities
    "save_npy",

    # AMReX integration
    "SDFMultiFab3D",

    # Complex assemblies
    "NATOFragment",
    "RocketAssembly",

    # Module-level visualisation
    "save_plotly_html_grid",
]
