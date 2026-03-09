"""Binary (2-phase) Chan-Vese active contour segmentation.

Adapted from uSCMAN Modules/Analysis/Segmentation/CV_single.py.
GPU backend uses CuPy when available; falls back to NumPy automatically.
The ``torch`` dependency from the original uSCMAN has been removed.
"""
from __future__ import annotations
import numpy as np


# ---------------------------------------------------------------------------
# Private helpers (unchanged from uSCMAN)
# ---------------------------------------------------------------------------

def _cv_heavyside(phi, xp):
    """Returns the result of a regularized heavyside function of the
    input value(s).
    """
    return 0.5 * (1.0 + (2.0 / xp.pi) * xp.arctan(phi / 1.0))


def _cv_delta(phi, xp):
    """Returns the result of a regularized dirac function of the
    input value(s).
    """
    return (1.0 / xp.pi) * (1.0 / (1.0 + phi ** 2))


def _cv_calculate_averages(image, Hphi, xp):
    """Returns the average values 'inside' and 'outside'."""
    H = Hphi.astype(image.dtype)
    Hinv = 1.0 - H
    Hsum = xp.sum(H)
    Hinvsum = xp.sum(Hinv)
    avg_inside = xp.einsum('ij,ij->',image,H)
    avg_outside = xp.einsum('ij,ij->',image,Hinv)
    if Hsum != 0:
        avg_inside /= Hsum
    if Hinvsum != 0:
        avg_outside /= Hinvsum
    return avg_inside, avg_outside

def _cv_calculate_variation(image, phi, mu, lambda1, lambda2, dt, xp):
    """Returns the variation of level set 'phi' based on algorithm parameters."""
    eta = 1e-16
    P = xp.pad(phi, 1, mode='edge')

    phixp = P[1:-1, 2:] - P[1:-1, 1:-1]
    phixn = P[1:-1, 1:-1] - P[1:-1, :-2]
    phix0 = (P[1:-1, 2:] - P[1:-1, :-2]) / 2.0

    phiyp = P[2:, 1:-1] - P[1:-1, 1:-1]
    phiyn = P[1:-1, 1:-1] - P[:-2, 1:-1]
    phiy0 = (P[2:, 1:-1] - P[:-2, 1:-1]) / 2.0

    C1 = 1.0 / xp.sqrt(eta + phixp**2 + phiy0**2)
    C2 = 1.0 / xp.sqrt(eta + phixn**2 + phiy0**2)
    C3 = 1.0 / xp.sqrt(eta + phix0**2 + phiyp**2)
    C4 = 1.0 / xp.sqrt(eta + phix0**2 + phiyn**2)

    K = P[1:-1, 2:] * C1 + P[1:-1, :-2] * C2 + P[2:, 1:-1] * C3 + P[:-2, 1:-1] * C4

    Hphi = (phi > 0).astype(image.dtype)
    c1, c2 = _cv_calculate_averages(image, Hphi, xp)

    difference_from_average_term = (-lambda1 * (image - c1)**2 + lambda2 * (image - c2)**2)
    new_phi = phi + (dt * _cv_delta(phi, xp)) * (mu * K + difference_from_average_term)
    return new_phi / (1 + mu * dt * _cv_delta(phi, xp) * (C1 + C2 + C3 + C4))

def _cv_difference_from_average_term(image, Hphi, lambda_pos, lambda_neg, xp):
    """Returns the 'energy' contribution due to the difference from
    the average value within a region at each point.
    """
    (c1, c2) = _cv_calculate_averages(image, Hphi, xp)
    Hinv = 1.0 - Hphi
    return lambda_pos * (image - c1) ** 2 * Hphi + lambda_neg * (image - c2) ** 2 * Hinv




def _cv_edge_length_term(phi, mu, xp):
    """Returns the 'energy' contribution due to the length of the
    edge between regions at each point, multiplied by a factor 'mu'.
    """
    P = xp.pad(phi, 1, mode='edge')
    fy = (P[2:, 1:-1] - P[:-2, 1:-1]) / 2.0
    fx = (P[1:-1, 2:] - P[1:-1, :-2]) / 2.0
    return mu * _cv_delta(phi, xp) * xp.sqrt(fx**2 + fy**2)



def _cv_energy(image, phi, mu, lambda1, lambda2, xp):
    """Returns the total 'energy' of the current level set function."""
    H = _cv_heavyside(phi, xp)
    avg_energy = _cv_difference_from_average_term(image, H, lambda1, lambda2, xp)
    len_energy = _cv_edge_length_term(phi, mu, xp)
    return xp.sum(avg_energy) + xp.sum(len_energy)


def _cv_reset_level_set(phi, xp, tau=0.6):
    """Resets the level set based on Godunov's method."""
    P = xp.pad(phi, 1, mode='edge')

    phixp = P[1:-1, 2:] - P[1:-1, 1:-1]
    phixn = P[1:-1, 1:-1] - P[1:-1, :-2]

    phiyp = P[2:, 1:-1] - P[1:-1, 1:-1]
    phiyn = P[1:-1, 1:-1] - P[:-2, 1:-1]

    G = xp.zeros_like(phi, dtype=phi.dtype)

    ap = xp.maximum(phixn, G)
    am = xp.minimum(phixn, G)
    bp = xp.maximum(phixp, G)
    bm = xp.minimum(phixp, G)
    cp = xp.maximum(phiyn, G)
    cm = xp.minimum(phiyn, G)
    dp = xp.maximum(phiyp, G)
    dm = xp.minimum(phiyp, G)

    indp = phi > 0
    indn = phi < 0
    G[indp] = xp.sqrt(xp.maximum(ap[indp]**2, bm[indp]**2) + xp.maximum(cp[indp]**2, dm[indp]**2)) - 1
    G[indn] = xp.sqrt(xp.maximum(am[indn]**2, bp[indn]**2) + xp.maximum(cm[indn]**2, dp[indn]**2)) - 1

    SGN = phi / xp.sqrt(phi**2 + 1.0)
    phi -= tau * SGN * G

    return phi


def _cv_init_level_set(image, xp, dtype):
    """Initializes the level set function."""
    minval = xp.amin(image)

    phi = xp.ones_like(image, dtype=dtype)
    phi[image == minval] = -1

    return _cv_reset_level_set(phi, xp, tau=0.6).astype(dtype, copy=False)


# ---------------------------------------------------------------------------
# Public function
# ---------------------------------------------------------------------------

def chan_vese(ImgName: str, image, Inputs: dict, GPU_available):
    """Chan-Vese segmentation algorithm."""
    # Load Cupy if GPU enabled or NumPy if CPU
    if GPU_available:
        import cupy as xp
    else:
        import numpy as xp

    # Determine variables from input
    mu = Inputs["Segmentation"]["mu"]
    lambda1 = Inputs["Segmentation"]["lambda1"]
    lambda2 = Inputs["Segmentation"]["lambda2"]
    phi_tol = Inputs["Segmentation"]["phi tolerance"]
    energy_tol = Inputs["Segmentation"]["energy tolerance"]
    max_num_iter = Inputs["Segmentation"]["maximum number of iterations"]
    dt = Inputs["Segmentation"]["dt"]
    extended_output = Inputs["Segmentation"]["return energies"]
    print_iter = Inputs["Segmentation"]["print steps"]
    dtype_string = Inputs["Segmentation"]["datatype"]

    # Define datatype
    if dtype_string == "DOUBLE_LONG":
        float_dtype = xp.float64
    else:
        float_dtype = xp.float32

    # If GPUs are available, move variables to GPU
    if GPU_available:
        image = xp.asarray(image, dtype=float_dtype)
        mu = xp.asarray(mu, dtype=float_dtype)
        lambda1 = xp.asarray(lambda1, dtype=float_dtype)
        lambda2 = xp.asarray(lambda2, dtype=float_dtype)
        dt = xp.asarray(dt, dtype=float_dtype)
        phi_tol = xp.asarray(phi_tol, dtype=float_dtype)
        energy_tol = xp.asarray(energy_tol, dtype=float_dtype)
        max_num_iter = xp.asarray(max_num_iter, dtype=float_dtype)

    # Check image shape
    if len(image.shape) != 2:
        raise ValueError("Input image should be a 2D array.")
    
    # Initialize level set
    phi = _cv_init_level_set(image, xp, float_dtype)
    
    # Check dimensions of initial level set
    if phi.shape != image.shape:
        raise ValueError("Dimensions of initial level set do not match image dimensions.")
    
    # Normalize the image
    image = image.astype(float_dtype, copy=False)
    image = (image - xp.min(image)) / (xp.max(image) - xp.min(image))
    
    # Initialize variables
    i = 0
    old_energy = _cv_energy(image, phi, mu, lambda1, lambda2, xp)
    energies = []
    phivar = phi_tol + 1
    energyvar = energy_tol + 1
    
    while phivar > phi_tol and energyvar > energy_tol and i < max_num_iter:
        oldphi = phi
        phi = _cv_calculate_variation(image, phi, mu, lambda1, lambda2, dt, xp)
        phi = _cv_reset_level_set(phi, xp)
        phivar = xp.sqrt(((phi - oldphi) ** 2).mean())
        
        R1 = (phi >= 0).astype(xp.int32)
        R2 = (phi < 0).astype(xp.int32)
        segmentation = [R1, R2]
        new_energy = _cv_energy(image, phi, mu, lambda1, lambda2, xp)
        energies.append(old_energy)
        energyvar = abs(new_energy - old_energy) / old_energy
        old_energy = new_energy
        
        i += 1
        
        if print_iter and i % xp.floor(max_num_iter / 10) == 0:
            if GPU_available:
                print(f'Completed iteration {i} of {xp.asnumpy(max_num_iter)} for {ImgName}')
            else:
                print(f'Completed iteration {i} of {max_num_iter} for {ImgName}')
    
    # Print convergence status
    print(f'{ImgName} converged on iteration {i}' if i < max_num_iter else f'{ImgName} did not converge')
    
    # Ensure all GPU computations are finished before moving data back to CPU
    if GPU_available:
        xp.cuda.Device().synchronize()
        xp.get_default_memory_pool().free_all_blocks()
        
        # Convert variables back to NumPy arrays
        phi = xp.asnumpy(phi)
        segmentation = xp.asnumpy(segmentation)
        energies = [arr.get() for arr in energies]
    
    # Return segmentation and phi; optionally return energies if extended_output is True
    return (segmentation, [phi], energies) if extended_output else (segmentation, [phi])
