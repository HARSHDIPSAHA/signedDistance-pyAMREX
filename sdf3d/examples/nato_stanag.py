"""NATO STANAG-4496 fragment geometry.

Usage::

    from sdf3d.examples import NATOFragment

    geom = NATOFragment(diameter=14.30e-3, L_over_D=1.09)
"""

from __future__ import annotations

import numpy as np

from .. import primitives as sdf
from sdf3d.geometry import Cylinder3D, Box3D, Intersection3D, Union3D, Geometry3D


def NATOFragment(
    diameter: float = 14.30e-3,
    L_over_D: float = 1.09,
    cone_angle_deg: float = 20.0,
) -> Geometry3D:
    """Build a NATO STANAG-4496 fragment geometry.

    The fragment is a cylinder capped with a cone (ogive nose).

    Parameters
    ----------
    diameter:
        Fragment diameter in metres (default 14.3 mm).
    L_over_D:
        Length-to-diameter ratio (default 1.09, giving 15.56/14.3).
    cone_angle_deg:
        Cone half-angle in degrees (default 20°).  Currently unused in the
        SDF formula but retained for documentation purposes.

    Returns
    -------
    Geometry3D
        The composable geometry object.  To fill an AMReX MultiFab, call
        ``lib.from_geometry(NATOFragment(...))`` yourself.
    """
    fragment_radius  = diameter / 2.0
    total_length     = diameter * L_over_D
    cylinder_height  = diameter
    cone_height      = total_length - cylinder_height

    # Cylinder (infinite cylinder intersected with a bounding box)
    cyl_inf  = Cylinder3D(axis_offset=[0.0, 0.0], radius=fragment_radius)
    cyl_box  = Box3D(half_size=[fragment_radius * 1.2, cylinder_height / 2, fragment_radius * 1.2])
    cyl_geom = (
        Intersection3D(cyl_inf, cyl_box)
        .rotate_x(np.pi / 2)
        .translate(0.0, 0.0, cylinder_height / 2)
    )

    # Cone
    def _sharp_cone_sdf(p: np.ndarray) -> np.ndarray:
        return sdf.sdCappedCone(p, cone_height, 0.0, fragment_radius)

    cone_geom = (
        Geometry3D(_sharp_cone_sdf)
        .rotate_x(np.pi / 2)
        .translate(0.0, 0.0, cylinder_height + cone_height)
    )

    # Union of cylinder + cone
    return Union3D(cyl_geom, cone_geom)
