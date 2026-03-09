"""Internal orchestrator: replaces uSCMAN's Analysis.py + RunBatch.py.

Runs preprocessing → morphometry → segmentation for a single image and
returns the results dictionary. Parallel batch execution is handled by
callers (e.g. the user's Analysis script) using ProcessPoolExecutor.
"""
from __future__ import annotations
import os
import numpy as np


# ---------------------------------------------------------------------------
# GPU detection 
# ---------------------------------------------------------------------------

def _detect_gpu() -> bool:
    """Return True if CuPy + a CUDA device are available."""
    try:
        import cupy as cp  # noqa: F401
        cp.cuda.runtime.getDeviceCount()   # raises if no device
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Image loader (PNG / TIFF / JPG / BMP / PGM / HDF5)
# ---------------------------------------------------------------------------

def _load_image(image_path: str, params: dict) -> np.ndarray:
    """Load a grayscale image from *image_path*.

    Supported formats
    -----------------
    Standard raster  PNG, JPG, BMP, TIFF (any bit depth), PGM
    HDF5             .h5 / .hdf5 — reads dataset named ``image`` or ``phi``
                     (allows restarting from a saved level-set field)
    """
    ext = os.path.splitext(image_path)[1].lower()

    if ext in (".h5", ".hdf5"):
        import h5py  
        with h5py.File(image_path, "r") as f:
            # Try common dataset names
            for key in ("image", "phi", "Phi1", list(f.keys())[0]):
                if key in f:
                    return np.array(f[key], dtype=np.uint8)
        raise KeyError(f"No recognised dataset in HDF5 file: {image_path}")

    try:
        import cv2
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise IOError(f"cv2.imread returned None for: {image_path}")
        return img
    except ImportError:
        pass  # fall through to PIL

    try:
        from PIL import Image
        import numpy as np
        return np.array(Image.open(image_path).convert("L"))
    except ImportError:
        raise ImportError(
            "Neither 'cv2' nor 'Pillow' is installed. "
            "Install one to load image files: pip install opencv-python or pillow"
        )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_pipeline(image_path: str, params: dict) -> dict:
    """Run the full uSCMAN pipeline for a single image.

    Parameters
    ----------
    image_path:
        Path to input image (PNG / TIFF / JPG / BMP / PGM / HDF5).
    params:
        Parameter dictionary. Same schema as uSCMAN JSON config files.

    Returns
    -------
    dict
        Nested results dictionary with keys:
        ``"Preprocessing"`` — before/after arrays (if enabled)
        ``"Morphometry"``   — per-phase shape stats (if enabled)
        ``"Segmentation"``  — Phi arrays + region masks
    """
    from .preprocessing.preprocess import preprocessImage    
    from .morphometry.morphometry import computeMorphometry 
    from .segmentation.segmentation import SegmentImage      

    name = os.path.splitext(os.path.basename(image_path))[0]
    image_obj = {
        "name": name,
        "base material": params.get("Image Properties", {}).get("method", "binary"),
        "path": image_path,
    }

    results: dict = {
        "Preprocessing": {},
        "Morphometry": {"defect": {}, "crystal": {}, "binder": {}},
        "Extracted Defects": {},
        "Segmentation": {},
    }

    # --- Preprocessing ---
    run_preprocess = params.get("Preprocessing Properties", {}).get(
        "Preprocess Image", True
    )
    if run_preprocess:
        image, old_image, params = preprocessImage(image_obj, params)
        if params["Preprocessing Properties"].get("Display comparison", False):
            results["Preprocessing"] = {"original": old_image, "new": image}
    else:
        image = _load_image(image_path, params)

    # --- Morphometry ---
    run_morph = params.get("Morphometry", {}).get("run morphometry", False)
    run_defect = params.get("Morphometry", {}).get("run extractDefect", False)
    if run_morph or run_defect:
        morph_dict, defect_dict = computeMorphometry(
            image_obj, params, image, run_morph, run_defect
        )
        for key in ("defect", "crystal", "binder"):
            if key in morph_dict:
                results["Morphometry"][key] = morph_dict[key]
        if defect_dict:
            results["Extracted Defects"] = defect_dict

    # --- Segmentation ---
    run_seg = params.get("Segmentation", {}).get("run segmentImage", True)
    if run_seg:
        gpu = _detect_gpu()
        levelset_dict = SegmentImage(name, image, params, gpu)
        results["Segmentation"] = levelset_dict

    return results