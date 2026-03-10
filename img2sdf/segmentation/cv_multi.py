"""Multiphase (4-region) Chan-Vese active contour segmentation.

Adapted from uSCMAN Modules/Analysis/Segmentation/CV_multi.py.
Torch dependency removed; CuPy probe used for GPU detection.
"""
from __future__ import annotations
import numpy as np


def _cv_heavyside(x, xp, eps=1.0):
    """Returns the result of a regularised heavyside function of the
    input value(s).
    """
    return 0.5 * (1.0 + (2.0 / xp.pi) * xp.arctan(x / eps))


def _cv_delta(x, xp, eps=1.0):
    """Returns the result of a regularised dirac function of the
    input value(s).
    """
    return (1.0 / xp.pi) *  (eps / (eps**2 + x**2))


def _cv_calculate_averages(image, H1phi, H2phi, xp):
    """Returns the average values in each of the 4 regions."""
    # Convert datatypes
    H1 = H1phi.astype(image.dtype)
    H2 = H2phi.astype(image.dtype)
    # Get inverse regions
    H1inv = 1.0 - H1
    H2inv = 1.0 - H2
    # Compute denominators
    den1 = xp.einsum('ij,ij->',H1,H2)
    den2 = xp.einsum('ij,ij->',H1,H2inv)
    den3 = xp.einsum('ij,ij->',H1inv,H2)
    den4 = xp.einsum('ij,ij->',H1inv,H2inv)
    # Compute region averages
    avg1 = xp.einsum('ij,ij,ij->',image,H1,H2)
    avg2 = xp.einsum('ij,ij,ij->',image,H1,H2inv)
    avg3 = xp.einsum('ij,ij,ij->',image,H1inv,H2)
    avg4 = xp.einsum('ij,ij,ij->',image,H1inv,H2inv)
    if den1 != 0:
        avg1 /= den1
    if den2 != 0:
        avg2 /= den2
    if den3 != 0:
        avg3 /= den3
    if den4 != 0:
        avg4 /= den4
    return (avg1, avg2, avg3, avg4)

def _cv_calculate_variation(image, phi1, phi2, mu, dt, xp):
    """Returns the variation of level set 'phi' based on algorithm parameters.

    This corresponds to equation (22) of the paper by Pascal Getreuer,
    which computes the next iteration of the level set based on a current
    level set.

    A full explanation regarding all the terms is beyond the scope of the
    present description, but there is one difference of particular import.
    In the original algorithm, convergence is accelerated, and required
    memory is reduced, by using a single array. This array, therefore, is a
    combination of non-updated and updated values. If this were to be
    implemented in python, this would require a double loop, where the
    benefits of having fewer iterations would be outweided by massively
    increasing the time required to perform each individual iteration. A
    similar approach is used by Rami Cohen, and it is from there that the
    C1-4 notation is taken.
    """
    eta = 1e-16
    P1 = xp.pad(phi1, 1, mode='edge')
    P2 = xp.pad(phi2, 1, mode='edge')

    phixp1 = P1[1:-1, 2:] - P1[1:-1, 1:-1]
    phixn1 = P1[1:-1, 1:-1] - P1[1:-1, :-2]
    phix01 = (P1[1:-1, 2:] - P1[1:-1, :-2]) / 2.0

    phiyp1 = P1[2:, 1:-1] - P1[1:-1, 1:-1]
    phiyn1 = P1[1:-1, 1:-1] - P1[:-2, 1:-1]
    phiy01 = (P1[2:, 1:-1] - P1[:-2, 1:-1]) / 2.0
    
    phixp2 = P2[1:-1, 2:] - P2[1:-1, 1:-1]
    phixn2 = P2[1:-1, 1:-1] - P2[1:-1, :-2]
    phix02 = (P2[1:-1, 2:] - P2[1:-1, :-2]) / 2.0

    phiyp2 = P2[2:, 1:-1] - P2[1:-1, 1:-1]
    phiyn2 = P2[1:-1, 1:-1] - P2[:-2, 1:-1]
    phiy02 = (P2[2:, 1:-1] - P2[:-2, 1:-1]) / 2.0

    C1 = 1.0 / xp.sqrt(eta + phixp1**2 + phiy01**2)
    C2 = 1.0 / xp.sqrt(eta + phixn1**2 + phiy01**2)
    C3 = 1.0 / xp.sqrt(eta + phix01**2 + phiyp1**2)
    C4 = 1.0 / xp.sqrt(eta + phix01**2 + phiyn1**2)
    
    D1 = 1.0 / xp.sqrt(eta + phixp2**2 + phiy02**2)
    D2 = 1.0 / xp.sqrt(eta + phixn2**2 + phiy02**2)
    D3 = 1.0 / xp.sqrt(eta + phix02**2 + phiyp2**2)
    D4 = 1.0 / xp.sqrt(eta + phix02**2 + phiyn2**2)

    K1 = P1[1:-1, 2:] * C1 + P1[1:-1, :-2] * C2 + P1[2:, 1:-1] * C3 + P1[:-2, 1:-1] * C4
    K2 = P2[1:-1, 2:] * D1 + P2[1:-1, :-2] * D2 + P2[2:, 1:-1] * D3 + P2[:-2, 1:-1] * D4

    H1phi = (phi1 > 0).astype(image.dtype)
    H2phi = (phi2 > 0).astype(image.dtype)
    (c1, c2, c3, c4) = _cv_calculate_averages(image, H1phi, H2phi, xp)

    difference_from_average_term1 = (
        -(image-c1)**2*H2phi-(image-c2)**2*(1.0-H2phi)+(image-c3)**2*H2phi+(image-c4)**2*(1.0-H2phi)
    )
    
    difference_from_average_term2 = (
        -(image-c1)**2*H1phi+(image-c2)**2*H1phi-(image-c3)**2*(1.0-H1phi)+(image-c4)**2*(1.0-H1phi)
    )
    
    new_phi1 = phi1 + (dt * _cv_delta(phi1, xp)) * (mu * K1 + difference_from_average_term1)
    new_phi1 = new_phi1 / (1 + mu * dt * _cv_delta(phi1, xp) * (C1 + C2 + C3 + C4))
    
    new_phi2 = phi2 + (dt * _cv_delta(phi2, xp)) * (mu * K2 + difference_from_average_term2)
    new_phi2 = new_phi2 / (1 + mu * dt * _cv_delta(phi2, xp) * (D1 + D2 + D3 + D4))

    return [new_phi1, new_phi2]

def _cv_difference_from_average_term(image, H1phi, H2phi, xp):
    """Returns the 'energy' contribution due to the difference from
    the average value within a region at each point.
    """
    (c1, c2, c3, c4) = _cv_calculate_averages(image, H1phi, H2phi, xp)
    H1inv = 1.0 - H1phi
    H2inv = 1.0 - H2phi
    mm1 = (image - c1)**2 * H1phi * H2phi
    mm2 = (image - c2)**2 * H1phi * H2inv
    mm3 = (image - c3)**2 * H1inv * H2phi
    mm4 = (image - c4)**2 * H1inv * H2inv
    return mm1 + mm2 + mm3 + mm4


def _cv_edge_length_term(phi1, phi2, mu, xp):
    """Returns the 'energy' contribution due to the length of the
    edge between regions at each point, multiplied by a factor 'mu'.
    """
    P1 = xp.pad(phi1, 1, mode='edge')
    fy1 = (P1[2:, 1:-1] - P1[:-2, 1:-1]) / 2.0
    fx1 = (P1[1:-1, 2:] - P1[1:-1, :-2]) / 2.0
    
    P2 = xp.pad(phi2, 1, mode='edge')
    fy2 = (P2[2:, 1:-1] - P2[:-2, 1:-1]) / 2.0
    fx2 = (P2[1:-1, 2:] - P2[1:-1, :-2]) / 2.0
    return [mu * _cv_delta(phi1, xp) * xp.sqrt(fx1**2 + fy1**2), mu * _cv_delta(phi2, xp) * xp.sqrt(fx2**2 + fy2**2)]


def _cv_energy(image, phi1, phi2, mu, xp):
    """Returns the total 'energy' of the current level set function.

    This corresponds to equation (7) of the paper by Pascal Getreuer,
    which is the weighted sum of the following:
    (A) the length of the contour produced by the zero values of the
    level set,
    (B) the area of the "foreground" (area of the image where the
    level set is positive),
    (C) the variance of the image inside the foreground,
    (D) the variance of the image outside of the foreground

    Each value is computed for each pixel, and then summed. The weight
    of (B) is set to 0 in this implementation.
    """
    H1 = _cv_heavyside(phi1, xp)
    H2 = _cv_heavyside(phi2, xp)
    avgenergy = _cv_difference_from_average_term(image, H1, H2, xp)
    lenenergy1, lenenergy2 = _cv_edge_length_term(phi1, phi2, mu, xp)
    return xp.sum(avgenergy) + xp.sum(lenenergy1) + xp.sum(lenenergy2)

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
    # Determine the number of unique values in the image

    unique = xp.unique(image)
    num_unique = len(unique)
    
    # Allow 3-phase images by expanding them to 4 regions
    if num_unique == 3:
        # duplicate the middle value so algorithm still works
        unique = xp.array([unique[0], unique[1], unique[1], unique[2]])
        num_unique = 4
    
    if num_unique not in [2, 4]:
        raise ValueError(f"Expected 2 or 4 phases but got {num_unique}")
        
    
    
    # Determine min value
    minval = xp.amin(image)
    maxval = xp.amax(image)
    idx_min = image == minval
    idx_max = image == maxval
    idx_mid = image == unique[1]

    # Initialize the levelsets
    phi1 = xp.ones_like(image, dtype=dtype) * -1.0
    phi1[xp.logical_or(idx_mid, idx_max)] = 1.0
    phi2 = xp.ones_like(image, dtype=dtype) 
    phi2[xp.logical_or(idx_min, idx_mid)] = -1.0

    return (_cv_reset_level_set(phi1, xp, tau=0.6).astype(dtype, copy=False),
            _cv_reset_level_set(phi2, xp, tau=0.6).astype(dtype, copy=False)
    )


def chan_vese_multi(ImgName: str, image, Inputs: dict, GPU_available):
    """Chan-Vese segmentation algorithm.

    Active contour model by evolving a level set. Can be used to
    segment objects without clearly defined boundaries.

    Parameters
    ----------
    image : (M, N) ndarray
        Grayscale image to be segmented.
    mu : float, optional
        'edge length' weight parameter. Higher `mu` values will
        produce a 'round' edge, while values closer to zero will
        detect smaller objects.
    lambda1 : float, optional
        'difference from average' weight parameter for the output
        region with value 'True'. If it is lower than `lambda2`, this
        region will have a larger range of values than the other.
    lambda2 : float, optional
        'difference from average' weight parameter for the output
        region with value 'False'. If it is lower than `lambda1`, this
        region will have a larger range of values than the other.
    tol : float, positive, optional
        Level set variation tolerance between iterations. If the
        L2 norm difference between the level sets of successive
        iterations normalized by the area of the image is below this
        value, the algorithm will assume that the solution was
        reached.
    max_num_iter : uint, optional
        Maximum number of iterations allowed before the algorithm
        interrupts itself.
    dt : float, optional
        A multiplication factor applied at calculations for each step,
        serves to accelerate the algorithm. While higher values may
        speed up the algorithm, they may also lead to convergence
        problems.

    Returns
    -------
    phi : (M, N) ndarray of floats
        Final level set computed by the algorithm.
    energies : list of floats
        Shows the evolution of the 'energy' for each step of the
        algorithm. This should allow to check whether the algorithm
        converged.

    Notes
    -----
    The Chan-Vese Algorithm is designed to segment objects without
    clearly defined boundaries. This algorithm is based on level sets
    that are evolved iteratively to minimize an energy, which is
    defined by weighted values corresponding to the sum of differences
    intensity from the average value outside the segmented region, the
    sum of differences from the average value inside the segmented
    region, and a term which is dependent on the length of the
    boundary of the segmented region.

    This algorithm was first proposed by Tony Chan and Luminita Vese,
    in a publication entitled "An Active Contour Model Without Edges"
    [1]_.

    This implementation of the algorithm is somewhat simplified in the
    sense that the area factor 'nu' described in the original paper is
    not implemented, and is only suitable for grayscale images.

    Typical values for `lambda1` and `lambda2` are 1. If the
    'background' is very different from the segmented object in terms
    of distribution (for example, a uniform black image with figures
    of varying intensity), then these values should be different from
    each other.

    Typical values for mu are between 0 and 1, though higher values
    can be used when dealing with shapes with very ill-defined
    contours.

    The 'energy' which this algorithm tries to minimize is defined
    as the sum of the differences from the average within the region
    squared and weighed by the 'lambda' factors to which is added the
    length of the contour multiplied by the 'mu' factor.

    Supports 2D grayscale images only, and does not implement the area
    term described in the original article.

    References
    ----------
    .. [1] An Active Contour Model without Edges, Tony Chan and
           Luminita Vese, Scale-Space Theories in Computer Vision,
           1999, :DOI:`10.1007/3-540-48236-9_13`
    .. [2] Chan-Vese Segmentation, Pascal Getreuer Image Processing On
           Line, 2 (2012), pp. 214-224,
           :DOI:`10.5201/ipol.2012.g-cv`
    .. [3] The Chan-Vese Algorithm - Project Report, Rami Cohen, 2011
           :arXiv:`1107.2782`
    """
    # Load Cupy if GPU enabled or NumPy if CPU
    if GPU_available:
        import cupy as xp
    else:
        import numpy as xp

    # Determine variables from input
    mu = Inputs["Segmentation"]["mu"]
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
        dt = xp.asarray(dt, dtype=float_dtype)
        phi_tol = xp.asarray(phi_tol, dtype=float_dtype)
        energy_tol = xp.asarray(energy_tol, dtype=float_dtype)
        max_num_iter = xp.asarray(max_num_iter, dtype=float_dtype)

    # Ensure image is 2D
    if len(image.shape) != 2:
        raise ValueError("Input image should be a 2D array.")
    
    # Initialize level set
    phi1, phi2 = _cv_init_level_set(image, xp, float_dtype)

    # Ensure phi is a numpy array the same size as the input image
    if type(phi1) != xp.ndarray or phi1.shape != image.shape or type(phi2) != xp.ndarray or phi2.shape != image.shape:
        raise ValueError(
            "The dimensions of initial level set do not "
            "match the dimensions of image."
        )
    
    # Normalize the image
    image = image.astype(float_dtype, copy=False)
    image = (image - xp.min(image)) / (xp.max(image) - xp.min(image))

    # Initialize variables
    i = 0
    old_energy = _cv_energy(image, phi1, phi2, mu, xp)
    energies = []
    phivar1 = phi_tol + 1
    phivar2 = phi_tol + 1
    energyvar = energy_tol + 1

    while phivar1 > phi_tol and phivar2 > phi_tol and energyvar > energy_tol and i < max_num_iter:
        # Save old level set values
        oldphi1 = phi1
        oldphi2 = phi2

        # Calculate new level set
        phi1, phi2 = _cv_calculate_variation(image, phi1, phi2, mu, dt, xp)
        phi1 = _cv_reset_level_set(phi1, xp)
        phi2 = _cv_reset_level_set(phi2, xp)
        phivar1 = xp.sqrt(((phi1 - oldphi1) ** 2).mean())
        phivar2 = xp.sqrt(((phi2 - oldphi2) ** 2).mean())

        # Extract energy and compare to previous level set and
        # segmentation to see if continuing is necessary
        R1 = xp.logical_and(phi1 >= 0, phi2 >= 0).astype(xp.int32)
        R2 = xp.logical_and(phi1 >= 0, phi2 < 0).astype(xp.int32)
        R3 = xp.logical_and(phi1 < 0, phi2 >= 0).astype(xp.int32)
        R4 = xp.logical_and(phi1 < 0, phi2 < 0).astype(xp.int32)
        segmentation = [R1, R2, R3, R4]
        new_energy = _cv_energy(image, phi1, phi2, mu, xp)

        # Save old energy values
        energies.append(old_energy)
        energyvar = abs(new_energy-old_energy) / old_energy   	
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
        phi1 = xp.asnumpy(phi1)
        phi2 = xp.asnumpy(phi2)
        # Convert Cupy arrays to NumPy arrays
        segmentation = [arr.get() for arr in segmentation]
        energies = [arr.get() for arr in energies]
        
    phi = [phi1, phi2]
        
    if extended_output:
        return (segmentation, phi, energies)
    else:
        return (segmentation, phi)
