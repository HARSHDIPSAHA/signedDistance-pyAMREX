"""img2sdf — Image-to-SDF package (uSCMAN integrated into pySdf).

Third input source for SDFs: raw microscopy/X-ray images → level-set Φ.
Alongside analytical shapes (sdf2d/sdf3d) and CAD meshes (stl2sdf).

Full uSCMAN pipeline included: Preprocessing → Morphometry → Chan-Vese segmentation.
Output plugs into the pySdf CSG tree as ImageGeometry2D (2D) or ImageGeometry3D (3D).

Backends
--------
NumPy / CuPy  — default, works without AMReX
AMReX         — SDFLibraryImg2D fills MultiFab grids for HPC solvers

All heavy dependencies (cv2, skimage, h5py, cupy, amrex) are import-guarded.
img2sdf imports without any of them installed.
"""
from .geometry import ImageGeometry2D
from .geometry3d import ImageGeometry3D
from .grid import image_to_levelset_2d, image_to_geometry_2d
from .grid3d import volume_to_levelset_3d, volume_to_geometry_3d
from .morphometry.morphometry3d import compute_morphometry_3d

__all__ = [
    "ImageGeometry2D",
    "ImageGeometry3D",
    "image_to_levelset_2d",
    "image_to_geometry_2d",
    "volume_to_levelset_3d",
    "volume_to_geometry_3d",
    "compute_morphometry_3d",
    "SDFLibraryImg2D",
]

def __getattr__(name: str):
    if name == "SDFLibraryImg2D":
        from .amrex import SDFLibraryImg2D
        return SDFLibraryImg2D
    raise AttributeError(f"module 'img2sdf' has no attribute {name!r}")