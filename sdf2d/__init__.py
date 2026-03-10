"""
sdf2d — 2D Signed Distance Function Library
============================================

A library for creating and composing 2D signed distance fields (SDFs),
based on Inigo Quilez's distance function collection.

Implemented features
--------------------
- Primitive shapes: Circle, Box, triangles, polygons, stars, arcs, ...
- Boolean operations: ``|`` (union), ``-`` (subtraction), ``/`` (intersection)
- Transforms: translate, rotate, scale, round, onion
- Grid sampling: :meth:`SDF2D.to_numpy`
- AMReX MultiFab output: :class:`MultiFabGrid2D` (requires pyAMReX 2-D build)

Quick start
-----------

NumPy mode (no AMReX required)::

    from sdf2d import Circle2D, Box2D

    circle = Circle2D(radius=0.3)
    box    = Box2D(half_size=(0.2, 0.2)).translate(0.4, 0.0)
    shape  = circle | box              # union operator

    phi = shape.to_numpy(bounds=((-1.0, 1.0), (-1.0, 1.0)), resolution=(512, 512))

AMReX mode::

    import amrex.space2d as amr
    from sdf2d import MultiFabGrid2D

    amr.initialize([])
    # ... set up geom, ba, dm ...
    grid     = MultiFabGrid2D(geom, ba, dm)
    mf       = Circle2D(0.3).to_multifab(grid)
    amr.finalize()
"""

from .geometry import (
    # Base class
    SDF2D,

    # Primitive shapes
    Circle2D,
    Box2D,
    RoundedBox2D,
    OrientedBox2D,
    Segment2D,
    Rhombus2D,
    Trapezoid2D,
    Parallelogram2D,

    # Triangles
    EquilateralTriangle2D,
    TriangleIsosceles2D,
    Triangle2D,

    # Capsules
    UnevenCapsule2D,

    # Regular polygons
    Pentagon2D,
    Hexagon2D,
    Octagon2D,
    NGon2D,

    # Stars
    Hexagram2D,
    Star2D,

    # Arcs and sectors
    Pie2D,
    CutDisk2D,
    Arc2D,
    Ring2D,
    Horseshoe2D,

    # Special shapes
    Vesica2D,
    Moon2D,
    RoundedCross2D,
    Egg2D,
    Heart2D,
    Cross2D,
    RoundedX2D,

    # Complex shapes
    Polygon2D,
    Ellipse2D,
    Parabola2D,
    ParabolaSegment2D,
    Bezier2D,
    BlobbyCross2D,
    Tunnel2D,
    Stairs2D,
    QuadraticCircle2D,
    Hyperbola2D,

)

from .geometry import save_npy
from .amrex import MultiFabGrid2D

__version__ = "0.2.0"

__all__ = [
    # Base
    "SDF2D",

    # Primitive shapes
    "Circle2D",
    "Box2D",
    "RoundedBox2D",
    "OrientedBox2D",
    "Segment2D",
    "Rhombus2D",
    "Trapezoid2D",
    "Parallelogram2D",

    # Triangles
    "EquilateralTriangle2D",
    "TriangleIsosceles2D",
    "Triangle2D",

    # Capsules
    "UnevenCapsule2D",

    # Regular polygons
    "Pentagon2D",
    "Hexagon2D",
    "Octagon2D",
    "NGon2D",

    # Stars
    "Hexagram2D",
    "Star2D",

    # Arcs and sectors
    "Pie2D",
    "CutDisk2D",
    "Arc2D",
    "Ring2D",
    "Horseshoe2D",

    # Special shapes
    "Vesica2D",
    "Moon2D",
    "RoundedCross2D",
    "Egg2D",
    "Heart2D",
    "Cross2D",
    "RoundedX2D",

    # Complex shapes
    "Polygon2D",
    "Ellipse2D",
    "Parabola2D",
    "ParabolaSegment2D",
    "Bezier2D",
    "BlobbyCross2D",
    "Tunnel2D",
    "Stairs2D",
    "QuadraticCircle2D",
    "Hyperbola2D",

    # Grid utilities
    "save_npy",

    # AMReX integration
    "MultiFabGrid2D",
]
