"""AMReX MultiFab integration for image-derived 2-D SDF fields.

Architecture
------------
The Chan-Vese segmentation pipeline runs entirely in NumPy (via
``img2sdf.grid.image_to_geometry_2d``).  The resulting ``ImageGeometry2D``
is a ``Geometry2D`` subclass, so it plugs directly into
``sdf2d.amrex.SDFLibrary2D.from_geometry()``, which evaluates the geometry's
``.sdf(p)`` at each AMReX cell centre and stores the values in a
``MultiFab``.

This is the same architecture used for all ``Geometry2D`` subclasses —
no need to rewrite Chan-Vese on AMReX.  The NumPy level-set is computed
once, interpolated on-the-fly at every cell, and written into the
``MultiFab`` grid.

Requires ``amrex.space2d`` (pyAMReX built in 2-D mode).
"""


class SDFLibraryImg2D:
    """Evaluate an image-derived SDF on an AMReX ``MultiFab`` grid.

    Runs the NumPy Chan-Vese pipeline (``img2sdf`` package) to produce an
    ``ImageGeometry2D``, then delegates cell-by-cell SDF evaluation to
    ``sdf2d.amrex.MultiFabGrid2D.from_geometry()``.

    Parameters
    ----------
    geom:
        ``amr.Geometry`` object defining the physical domain and cell spacing.
    ba:
        ``amr.BoxArray`` describing the domain decomposition.
    dm:
        ``amr.DistributionMapping`` assigning boxes to MPI ranks.
    """

    def __init__(self, geom, ba, dm):
        self.geom = geom
        self.ba   = ba
        self.dm   = dm

    def from_image(self, image_path, params, bounds=None, *, preprocess=True):
        """Run the Chan-Vese pipeline and fill a ``MultiFab`` with the SDF.

        Parameters
        ----------
        image_path:
            Path to the input image (PNG, JPG, TIFF, BMP, or HDF5 ``.h5``).
        params:
            uSCMAN parameter dict (same schema as the JSON config file).
        bounds:
            Physical domain ``((x0, x1), (y0, y1))``.  When *None* the image
            is mapped to pixel coordinates.
        preprocess:
            Unused — kept for API compatibility.

        Returns
        -------
        amr.MultiFab
            Single-component ``MultiFab`` filled with the image-derived SDF
            (φ < 0 inside, φ > 0 outside — pySdf convention).
        """
        # Step 1 — NumPy pipeline: image → ImageGeometry2D
        from .grid import image_to_geometry_2d
        img_geom = image_to_geometry_2d(image_path, params, bounds)

        # Step 2 — evaluate ImageGeometry2D on the AMReX grid
        from sdf2d.amrex import MultiFabGrid2D
        lib = MultiFabGrid2D(self.geom, self.ba, self.dm)
        return img_geom.to_multifab(lib)
