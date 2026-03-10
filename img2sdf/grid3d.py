"""NumPy path: 3D volume → level-set ndarray or ImageGeometry3D.

Mirrors img2sdf/grid.py but for 3-D volumetric inputs.
"""
from __future__ import annotations
import os
import numpy as np
import numpy.typing as npt
from .geometry3d import ImageGeometry3D

_Array = npt.NDArray[np.floating]


# ---------------------------------------------------------------------------
# Private loader
# ---------------------------------------------------------------------------

def _load_volume(source, params: dict) -> np.ndarray:
    """Load a 3D volume from *source*.

    Parameters
    ----------
    source:
        Either a numpy array of shape (D, H, W) or a file path string.
        Supported file formats: ``.npy``, ``.npz``, ``.h5``/``.hdf5``,
        ``.tif``/``.tiff`` (multi-page TIFF stack).
    params:
        Parameter dictionary (used for dataset key hints on HDF5 inputs).

    Returns
    -------
    numpy.ndarray
        Shape (D, H, W).
    """
    if isinstance(source, np.ndarray):
        if source.ndim != 3:
            raise ValueError(
                f"Volume array must have 3 dimensions (D, H, W); got {source.ndim}."
            )
        return source.astype(np.float64)

    ext = os.path.splitext(source)[1].lower()

    if ext == ".npy":
        vol = np.load(source)
        if vol.ndim != 3:
            raise ValueError(
                f"Loaded .npy array has shape {vol.shape}; expected 3D (D, H, W)."
            )
        return vol.astype(np.float64)

    if ext == ".npz":
        data = np.load(source)
        keys = list(data.keys())
        if not keys:
            raise KeyError(f"No arrays found in .npz file: {source}")
        if len(keys) > 1:
            import warnings
            warnings.warn(
                f".npz file '{source}' contains multiple arrays {keys}; "
                f"loading the first one ('{keys[0]}'). "
                "Pass a numpy array directly to select a specific array.",
                UserWarning,
                stacklevel=3,
            )
        vol = data[keys[0]]
        if vol.ndim != 3:
            raise ValueError(
                f"Loaded .npz array '{keys[0]}' has shape {vol.shape}; "
                "expected 3D (D, H, W)."
            )
        return vol.astype(np.float64)

    if ext in (".h5", ".hdf5"):
        import h5py  # import-guarded
        with h5py.File(source, "r") as f:
            all_keys = list(f.keys())
            if not all_keys:
                raise KeyError(f"No datasets found in HDF5 file: {source}")
            for key in ("volume", "image", "phi", all_keys[0]):
                if key in f:
                    vol = np.array(f[key], dtype=np.float64)
                    if vol.ndim == 3:
                        return vol
                    raise ValueError(
                        f"HDF5 dataset '{key}' has shape {vol.shape}; "
                        "expected 3D."
                    )
        raise KeyError(f"No recognised dataset in HDF5 file: {source}")

    if ext in (".tif", ".tiff"):
        try:
            import tifffile
            vol = tifffile.imread(source)
        except ImportError:
            try:
                from PIL import Image
                img = Image.open(source)
                frames = []
                for i in range(getattr(img, "n_frames", 1)):
                    img.seek(i)
                    frames.append(np.array(img))
                vol = np.stack(frames, axis=0)
            except ImportError as exc:
                raise ImportError(
                    "Install 'tifffile' or 'Pillow' to load TIFF stacks."
                ) from exc
        if vol.ndim != 3:
            raise ValueError(
                f"TIFF file produced array with shape {vol.shape}; "
                "expected 3D (D, H, W)."
            )
        return vol.astype(np.float64)

    raise ValueError(
        f"Unsupported file extension '{ext}'. "
        "Use .npy, .npz, .h5/.hdf5, or .tif/.tiff."
    )


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def volume_to_levelset_3d(
    volume_path_or_array,
    params: dict,
    *,
    levelset_index: int = 0,
) -> _Array:
    """Run 3D Chan-Vese segmentation and return the level-set φ array.

    Parameters
    ----------
    volume_path_or_array:
        Either a numpy array of shape (D, H, W) or a file path string
        (``.npy``, ``.npz``, ``.h5``/``.hdf5``, ``.tif``/``.tiff``).
    params:
        Parameter dictionary.  The ``"Segmentation"`` sub-dict may contain
        any of the ``chan_vese_3d`` keyword arguments (``mu``, ``lambda1``,
        ``lambda2``, ``tol``, ``max_iter``, ``dt``, ``sigma``,
        ``reinit_interval``).
    levelset_index:
        Which level-set to return for multiphase cases (currently unused;
        reserved for future multi-phase 3D support).

    Returns
    -------
    numpy.ndarray
        Shape (D, H, W) with pySdf sign convention: φ < 0 inside the
        segmented object, φ > 0 outside.
    """
    from .segmentation.cv_single_3d import chan_vese_3d

    volume = _load_volume(volume_path_or_array, params)

    seg_params = params.get("Segmentation", {})
    cv_kwargs = {
        k: seg_params[k]
        for k in (
            "mu", "lambda1", "lambda2", "tol", "max_iter", "dt",
            "sigma", "reinit_interval",
        )
        if k in seg_params
    }

    _segmentation, phi_list = chan_vese_3d(volume, **cv_kwargs)

    phi = phi_list[levelset_index]  # uSCMAN: phi > 0 inside
    return -phi                      # pySdf:   phi < 0 inside


def volume_to_geometry_3d(
    volume_path_or_array,
    params: dict,
    bounds: (
        tuple[tuple[float, float], tuple[float, float], tuple[float, float]]
        | None
    ) = None,
    *,
    levelset_index: int = 0,
) -> ImageGeometry3D:
    """Run 3D Chan-Vese and return an ``ImageGeometry3D`` CSG leaf.

    Parameters
    ----------
    volume_path_or_array, params, levelset_index:
        Same as :func:`volume_to_levelset_3d`.
    bounds:
        Physical domain ``((x0, x1), (y0, y1), (z0, z1))``.  When *None*,
        the volume is mapped to voxel coordinates
        ``((0, W), (0, H), (0, D))``.

    Returns
    -------
    ImageGeometry3D
        Ready to compose with other sdf3d shapes via ``.union()``,
        ``.intersect()``, ``.subtract()``, etc.
    """
    phi = volume_to_levelset_3d(
        volume_path_or_array, params, levelset_index=levelset_index
    )
    D, H, W = phi.shape
    if bounds is None:
        bounds = ((0.0, float(W)), (0.0, float(H)), (0.0, float(D)))
    src = (
        volume_path_or_array
        if isinstance(volume_path_or_array, str)
        else None
    )
    return ImageGeometry3D(phi, bounds, image_path=src)
