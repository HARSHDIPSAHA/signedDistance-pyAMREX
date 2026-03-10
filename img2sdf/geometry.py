"""ImageGeometry2D — CSG leaf node built from an image's level-set field.

Inherits from sdf2d.Geometry2D so that image-derived SDFs automatically
get all pySdf methods: save_png(), translate(), rotate(), scale(),
union(), subtract(), intersect(), round(), onion(), etc.
"""
from __future__ import annotations
import numpy as np
import numpy.typing as npt
from scipy.interpolate import RegularGridInterpolator
from sdf2d import SDF2D

_Array = npt.NDArray[np.floating]


class ImageGeometry2D(SDF2D):
    """A 2-D SDF geometry sourced from a segmented image level-set field.

    Inherits the full pySdf SDF2D API — including save_png(),
    translate(), rotate(), scale(), union(), subtract(), intersect().

    Parameters
    ----------
    phi:
        2-D ndarray of shape (ny, nx).  phi < 0 inside, phi > 0 outside.
    bounds:
        Physical extent ``((x0, x1), (y0, y1))`` matching the image region.
    image_path:
        Optional path to the source image (stored for reference only).
    """

    def __init__(
        self,
        phi: _Array,
        bounds: tuple[tuple[float, float], tuple[float, float]],
        image_path: str | None = None,
    ) -> None:
        self._phi       = np.asarray(phi, dtype=np.float64)
        self._bounds    = bounds
        self.image_path = image_path

        ny, nx = self._phi.shape
        (x0, x1), (y0, y1) = bounds
        xs = np.linspace(x0, x1, nx)
        ys = np.linspace(y0, y1, ny)

        self._interp = RegularGridInterpolator(
            (ys, xs), self._phi,
            method="linear",
            bounds_error=False,
            fill_value=None,   # extrapolate by nearest edge value
        )

        # Pass the interpolator as the SDF callable to Geometry2D.__init__
        # This gives us save_png(), translate(), rotate(), etc.
        super().__init__(self._eval_sdf)

    def _eval_sdf(self, p: _Array) -> _Array:
        """Internal SDF callable passed to Geometry2D base class."""
        pts = np.asarray(p)
        return self._interp(pts[..., ::-1])   # (y, x) order for interpolator

    
    # ------------------------------------------------------------------
    # negate() — not in Geometry2D base, defined here explicitly
    # ------------------------------------------------------------------

    def negate(self) -> "ImageGeometry2D":
        """Return a new ImageGeometry2D with the sign of phi flipped.

        Converts between uSCMAN convention (phi > 0 inside) and
        pySdf convention (phi < 0 inside), and back.
        """
        return ImageGeometry2D(-self._phi, self._bounds, self.image_path)

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    @property
    def phi(self) -> _Array:
        """Raw level-set ndarray (ny, nx)."""
        return self._phi

    @property
    def bounds(self):
        return self._bounds

    def __repr__(self) -> str:
        ny, nx = self._phi.shape
        src = f" from '{self.image_path}'" if self.image_path else ""
        return f"ImageGeometry2D(shape=({ny},{nx}){src})"