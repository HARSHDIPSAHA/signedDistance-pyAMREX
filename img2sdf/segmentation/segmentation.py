"""Dispatcher: routes to binary or multiphase Chan-Vese (2D) or 3D Chan-Vese.
Adapted from uSCMAN Segmentation.py — unchanged logic, new import paths.
"""
from __future__ import annotations
import numpy as np
from .cv_single import chan_vese
from .cv_multi import chan_vese_multi


def create_levelset_dictionary(Phi: list, segmentation: list, energies=None) -> dict:
    # Get the number of levelsets
    num_levelsets = len(Phi)

    # Get the size of the levelset field
    row, col = np.shape(Phi[0])
    xvec = np.arange(col) + 0.5
    yvec = np.arange(row) + 0.5
    X, Y = np.meshgrid(xvec, yvec)

    # Create the dictionary
    d = {"X-Coord": X.ravel(), "Y-Coord": Y.ravel(), "I": col, "J": row}

    # Add Phi
    for i, phi in enumerate(Phi):
        d[f"Phi{i+1}"] = np.flipud(phi).ravel()
    # Add segmentation  
    for i, region in enumerate(segmentation):
        d[f"R{i+1}"] = np.flipud(region).ravel()
    
    # Add energies
    if energies:
        d["Energies"] = energies
    return d


def _create_levelset_dictionary_3d(Phi: list, segmentation: list) -> dict:
    """Build a result dictionary for a 3D level-set field."""
    phi0 = Phi[0]
    D, H, W = phi0.shape
    d = {"D": D, "H": H, "W": W}
    for i, phi in enumerate(Phi):
        d[f"Phi{i + 1}"] = phi
    for i, region in enumerate(segmentation):
        d[f"R{i + 1}"] = region
    return d


def SegmentImage(img_name: str, image: np.ndarray, params: dict, gpu_available: bool) -> dict:
    # Dispatch to 3D solver when the input volume is 3-dimensional
    if image.ndim == 3:
        from .cv_single_3d import chan_vese_3d
        seg_params = params.get("Segmentation", {})
        cv_kwargs = {
            k: seg_params[k]
            for k in (
                "mu", "lambda1", "lambda2", "tol", "max_iter", "dt",
                "sigma", "reinit_interval",
            )
            if k in seg_params
        }
        segmentation, Phi = chan_vese_3d(image, gpu_available=gpu_available, **cv_kwargs)
        return _create_levelset_dictionary_3d(Phi, segmentation)

    # 2D path (unchanged)
    # Define method ('binary' or 'multiphase')
    method = params["Segmentation"]["segmentation method"]

    # Define segmentation function based on method
    fn = chan_vese if method == "binary" else chan_vese_multi

    # Run segmentation
    result = fn(img_name, image, params, gpu_available)
    # result is (segmentation, Phi) or (segmentation, Phi, energies)
    segmentation, Phi = result[0], result[1]
    energies = result[2] if len(result) > 2 else None

    # Create dictionary
    return create_levelset_dictionary(Phi, segmentation, energies)