class SDFLibraryImg2D:
    def __init__(self, geom, ba, dm):
        self.geom = geom
        self.ba   = ba
        self.dm   = dm

    def from_image(self, image_path, params, bounds=None, *, preprocess=True):
        # Step 1 — NumPy pipeline (existing img2sdf)
        from .grid import image_to_geometry_2d
        geom = image_to_geometry_2d(image_path, params, bounds)

        # Step 2 — reuse sdf2d's existing AMReX bridge
        from sdf2d.amrex import SDFLibrary2D
        lib = SDFLibrary2D(geom, self.ba, self.dm)
        return lib.fill_multifab()