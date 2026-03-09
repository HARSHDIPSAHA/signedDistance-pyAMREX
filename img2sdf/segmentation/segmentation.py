"""Dispatcher: routes to binary or multiphase Chan-Vese.
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


def SegmentImage(img_name: str, image: np.ndarray, params: dict, gpu_available: bool) -> dict:
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