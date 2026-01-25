import numpy as np

import amrex.space3d as amr

import sdf_lib as sdf
from .geometry import Box, RoundBox, Sphere


def _get_component_view(arr):
    if arr.ndim == 4:
        return arr[:, :, :, 0]
    return arr[:, :, :, 0, 0]


def _copy_mf(src):
    mf = amr.MultiFab(src.box_array, src.distribution_map, src.n_comp, src.n_grow)
    for mfi in src:
        arr_src = src.array(mfi).to_numpy()
        arr_dst = mf.array(mfi).to_numpy()
        _get_component_view(arr_dst)[...] = _get_component_view(arr_src)
    return mf


class SDFLibrary:
    """
    Build SDF fields directly in AMReX MultiFabs.
    """

    def __init__(self, geom, ba, dm):
        self.geom = geom
        self.ba = ba
        self.dm = dm

    def create_field(self):
        return amr.MultiFab(self.ba, self.dm, 1, 0)

    def from_geometry(self, geometry):
        mf = self.create_field()
        self._fill_multifab(mf, geometry.sdf)
        return mf

    def sphere(self, center, radius):
        geom = Sphere(radius).translate(*center)
        return self.from_geometry(geom)

    def box(self, center, half_size):
        geom = Box(half_size).translate(*center)
        return self.from_geometry(geom)

    def round_box(self, center, half_size, radius):
        geom = RoundBox(half_size, radius).translate(*center)
        return self.from_geometry(geom)

    def union(self, a, b):
        out = _copy_mf(a)
        for mfi in out:
            arr_out = out.array(mfi).to_numpy()
            arr_b = b.array(mfi).to_numpy()
            _get_component_view(arr_out)[...] = sdf.opUnion(
                _get_component_view(arr_out), _get_component_view(arr_b)
            )
        return out

    def subtract(self, a, b):
        out = _copy_mf(a)
        for mfi in out:
            arr_out = out.array(mfi).to_numpy()
            arr_b = b.array(mfi).to_numpy()
            _get_component_view(arr_out)[...] = sdf.opSubtraction(
                _get_component_view(arr_out), _get_component_view(arr_b)
            )
        return out

    def intersect(self, a, b):
        out = _copy_mf(a)
        for mfi in out:
            arr_out = out.array(mfi).to_numpy()
            arr_b = b.array(mfi).to_numpy()
            _get_component_view(arr_out)[...] = sdf.opIntersection(
                _get_component_view(arr_out), _get_component_view(arr_b)
            )
        return out

    def negate(self, a):
        out = _copy_mf(a)
        for mfi in out:
            arr_out = out.array(mfi).to_numpy()
            _get_component_view(arr_out)[...] *= -1.0
        return out

    def _fill_multifab(self, mf, sdf_func):
        dx = self.geom.data().CellSize()
        prob_lo = self.geom.ProbLoArray()

        for mfi in mf:
            arr = mf.array(mfi).to_numpy()
            bx = mfi.validbox()

            i_lo, j_lo, k_lo = bx.lo_vect
            i_hi, j_hi, k_hi = bx.hi_vect

            i = np.arange(i_lo, i_hi + 1)
            j = np.arange(j_lo, j_hi + 1)
            k = np.arange(k_lo, k_hi + 1)

            x = (i + 0.5) * dx[0] + prob_lo[0]
            y = (j + 0.5) * dx[1] + prob_lo[1]
            z = (k + 0.5) * dx[2] + prob_lo[2]

            Z, Y, X = np.meshgrid(z, y, x, indexing="ij")
            p = sdf.vec3(X, Y, Z)

            _get_component_view(arr)[...] = sdf_func(p)
