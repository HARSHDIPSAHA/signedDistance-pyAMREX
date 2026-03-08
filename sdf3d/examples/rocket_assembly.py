"""Parametric rocket assembly geometry.

Usage::

    from sdf3d.examples import RocketAssembly

    geom = RocketAssembly(body_radius=0.15)
"""

from __future__ import annotations

import numpy as np

from .. import primitives as sdf
from sdf3d.geometry import Sphere3D, Box3D, Union3D, SDF3D


def RocketAssembly(
    body_radius: float = 0.15,
    L_extra: float = 0.40,
    nose_len: float = 0.25,
    fin_span: float = 0.12,
    fin_height: float = 0.18,
    fin_thickness: float = 0.03,
    n_fins: int = 4,
) -> SDF3D:
    """Build a parametric rocket assembly.

    The rocket consists of:

    * A capsule body — a sphere elongated along Z.
    * A nose cone — a capped cone on top of the body.
    * *n_fins* rectangular fins arranged radially around the body base.

    Parameters
    ----------
    body_radius:
        Sphere radius of the body capsule (m).
    L_extra:
        Elongation length added to the body along Z (m).
    nose_len:
        Length of the nose cone (m).
    fin_span:
        Radial span of each fin (m).
    fin_height:
        Axial height of each fin (m).
    fin_thickness:
        Thickness of each fin (m).
    n_fins:
        Number of fins (default 4).

    Returns
    -------
    SDF3D
        The composable geometry object.  To fill an AMReX MultiFab, call
        ``lib.from_geometry(RocketAssembly(...))`` yourself.
    """
    R = body_radius

    # Body: elongated sphere
    body_geom = Sphere3D(R).elongate(0.0, 0.0, L_extra)

    # Nose cone
    z_body_top   = (L_extra / 2.0) + R
    z_cone_center = z_body_top + nose_len / 2.0
    h_cone        = nose_len / 2.0

    def _nose_sdf(p: np.ndarray) -> np.ndarray:
        qx = p[..., 0]
        qy = p[..., 1]
        qz = p[..., 2] - z_cone_center
        q  = np.stack([qx, qz, qy], axis=-1)
        return sdf.sdCappedCone(q, h_cone, 0.0, R)

    nose_geom = SDF3D(_nose_sdf)

    # Fins
    fin_half    = [fin_span / 2.0, fin_thickness / 2.0, fin_height / 2.0]
    z_fin_center = -0.18

    fins_geom: SDF3D | None = None
    for i in range(n_fins):
        angle      = i * (2 * np.pi / n_fins)
        radial_dist = R + fin_half[0]
        dx = radial_dist * np.cos(angle)
        dy = radial_dist * np.sin(angle)

        single_fin = (
            Box3D(half_size=fin_half)
            .rotate_z(angle)
            .translate(dx, dy, z_fin_center)
        )

        fins_geom = single_fin if fins_geom is None else Union3D(fins_geom, single_fin)

    rocket: SDF3D = Union3D(body_geom, nose_geom)
    if fins_geom is not None:
        rocket = Union3D(rocket, fins_geom)

    return rocket
