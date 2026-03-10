"""ImageGeometry3D — CSG leaf node built from a 3D volume level-set field.

Inherits from sdf3d.geometry.Geometry3D so that volumetric image-derived SDFs
automatically get all pySdf 3D methods: translate(), rotate(), scale(),
union(), subtract(), intersect(), round(), onion(), etc.
"""
from __future__ import annotations
import numpy as np
import numpy.typing as npt
from sdf3d.geometry import Geometry3D

_Array = npt.NDArray[np.floating]


class ImageGeometry3D(Geometry3D):
    """A 3-D SDF geometry sourced from a segmented volumetric level-set field.

    Inherits the full pySdf Geometry3D API — including translate(),
    rotate_x/y/z(), scale(), union(), subtract(), intersect().

    Parameters
    ----------
    phi:
        3-D ndarray of shape (D, H, W).  phi < 0 inside, phi > 0 outside
        (pySdf convention).
    bounds:
        Physical extent ``((x0, x1), (y0, y1), (z0, z1))`` matching the
        volume region.
    image_path:
        Optional path to the source volume file (stored for reference only).
    """

    def __init__(
        self,
        phi: _Array,
        bounds: tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
        image_path: str | None = None,
    ) -> None:
        try:
            from scipy.ndimage import map_coordinates as _map_coordinates
            self._map_coordinates = _map_coordinates
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "scipy is required for ImageGeometry3D. "
                "Install it with: pip install scipy"
            ) from exc

        self._phi = np.asarray(phi, dtype=np.float64)
        self._bounds = bounds
        self.image_path = image_path

        super().__init__(self._eval_sdf)

    def _eval_sdf(self, p: _Array) -> _Array:
        """Trilinear interpolation of the stored φ grid at query points *p*.

        Parameters
        ----------
        p:
            Array of shape ``(..., 3)`` — physical (x, y, z) coordinates.

        Returns
        -------
        numpy.ndarray
            Shape ``(...)`` — interpolated signed distances.
        """
        pts = np.asarray(p)
        original_shape = pts.shape[:-1]
        pts_flat = pts.reshape(-1, 3)

        (x0, x1), (y0, y1), (z0, z1) = self._bounds
        D, H, W = self._phi.shape

        # Map physical coordinates to voxel index coordinates
        # phi is indexed as phi[d, h, w] where d~z, h~y, w~x
        px = pts_flat[:, 0]
        py = pts_flat[:, 1]
        pz = pts_flat[:, 2]

        # Normalise to [0, shape-1].  When bounds have zero extent (degenerate
        # dimension), all query points map to index 0 — this is intentional so
        # that 1-voxel-thick slabs do not raise division-by-zero errors.
        wi = (px - x0) / (x1 - x0) * (W - 1) if x1 != x0 else np.zeros_like(px)
        hi = (py - y0) / (y1 - y0) * (H - 1) if y1 != y0 else np.zeros_like(py)
        di = (pz - z0) / (z1 - z0) * (D - 1) if z1 != z0 else np.zeros_like(pz)

        # map_coordinates expects coordinates in array index order (d, h, w)
        coords = np.array([di, hi, wi])

        result = self._map_coordinates(
            self._phi, coords, order=1, mode="nearest"
        )
        return result.reshape(original_shape)

    # ------------------------------------------------------------------
    # negate — flip sign convention
    # ------------------------------------------------------------------

    def negate(self) -> "ImageGeometry3D":
        """Return a new ImageGeometry3D with the sign of phi flipped.

        Converts between uSCMAN convention (phi > 0 inside) and
        pySdf convention (phi < 0 inside), and back.
        """
        return ImageGeometry3D(-self._phi, self._bounds, self.image_path)

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    @property
    def phi(self) -> _Array:
        """Raw level-set ndarray (D, H, W)."""
        return self._phi

    @property
    def bounds(self):
        return self._bounds

    def __repr__(self) -> str:
        D, H, W = self._phi.shape
        src = f" from '{self.image_path}'" if self.image_path else ""
        return f"ImageGeometry3D(shape=({D},{H},{W}){src})"
