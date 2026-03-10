"""3D Robust Chan-Vese active contour segmentation.

Implements the 3D Robust Chan-Vese (RCV) model based on:
  "3D Robust Chan-Vese model for industrial CT volume data segmentation"

The robust variant replaces the global intensity energy term with a local
Gaussian-filtered version for improved noise robustness:
  e_i(x) = (K_σ * |I - c_i|²)(x)

GPU backend uses CuPy when available; falls back to NumPy automatically.
"""
from __future__ import annotations
import numpy as np


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _heaviside(phi: np.ndarray, eps: float, xp) -> np.ndarray:
    """Regularised Heaviside: H_ε(φ) = 0.5*(1 + 2/π * arctan(φ/ε))."""
    return 0.5 * (1.0 + (2.0 / xp.pi) * xp.arctan(phi / eps))


def _delta(phi: np.ndarray, eps: float, xp) -> np.ndarray:
    """Regularised Dirac delta: δ_ε(φ) = ε / (π*(ε² + φ²))."""
    return eps / (xp.pi * (eps ** 2 + phi ** 2))


def _compute_averages_3d(image: np.ndarray, H: np.ndarray, xp):
    """Return (c1, c2) — mean intensities inside/outside the contour."""
    Hinv = 1.0 - H
    Hsum = xp.sum(H)
    Hinvsum = xp.sum(Hinv)
    c1 = xp.sum(image * H) / Hsum if Hsum > 0 else xp.array(0.0)
    c2 = xp.sum(image * Hinv) / Hinvsum if Hinvsum > 0 else xp.array(0.0)
    return c1, c2


def _curvature_3d(phi: np.ndarray, xp) -> np.ndarray:
    """3D curvature: div(∇φ / |∇φ|) via central differences (Neumann BCs)."""
    eta = 1e-16
    P = xp.pad(phi, 1, mode="edge")

    # Central differences for gradients
    gx = (P[1:-1, 1:-1, 2:] - P[1:-1, 1:-1, :-2]) / 2.0
    gy = (P[1:-1, 2:, 1:-1] - P[1:-1, :-2, 1:-1]) / 2.0
    gz = (P[2:, 1:-1, 1:-1] - P[:-2, 1:-1, 1:-1]) / 2.0

    norm = xp.sqrt(gx ** 2 + gy ** 2 + gz ** 2 + eta)
    nx_ = gx / norm
    ny_ = gy / norm
    nz_ = gz / norm

    # Pad the normalised gradient components and compute divergence
    Px = xp.pad(nx_, 1, mode="edge")
    Py = xp.pad(ny_, 1, mode="edge")
    Pz = xp.pad(nz_, 1, mode="edge")

    div_x = (Px[1:-1, 1:-1, 2:] - Px[1:-1, 1:-1, :-2]) / 2.0
    div_y = (Py[1:-1, 2:, 1:-1] - Py[1:-1, :-2, 1:-1]) / 2.0
    div_z = (Pz[2:, 1:-1, 1:-1] - Pz[:-2, 1:-1, 1:-1]) / 2.0

    return div_x + div_y + div_z


def _sussman_reinit_3d(phi: np.ndarray, tau: float, xp) -> np.ndarray:
    """Sussman reinitialization (Godunov scheme) for 3D level sets."""
    P = xp.pad(phi, 1, mode="edge")

    phixp = P[1:-1, 1:-1, 2:] - P[1:-1, 1:-1, 1:-1]
    phixn = P[1:-1, 1:-1, 1:-1] - P[1:-1, 1:-1, :-2]
    phiyp = P[1:-1, 2:, 1:-1] - P[1:-1, 1:-1, 1:-1]
    phiyn = P[1:-1, 1:-1, 1:-1] - P[1:-1, :-2, 1:-1]
    phizp = P[2:, 1:-1, 1:-1] - P[1:-1, 1:-1, 1:-1]
    phizn = P[1:-1, 1:-1, 1:-1] - P[:-2, 1:-1, 1:-1]

    G = xp.zeros_like(phi)

    ap = xp.maximum(phixn, G)
    am = xp.minimum(phixn, G)
    bp = xp.maximum(phixp, G)
    bm = xp.minimum(phixp, G)
    cp = xp.maximum(phiyn, G)
    cm = xp.minimum(phiyn, G)
    dp = xp.maximum(phiyp, G)
    dm = xp.minimum(phiyp, G)
    ep = xp.maximum(phizn, G)
    em = xp.minimum(phizn, G)
    fp = xp.maximum(phizp, G)
    fm = xp.minimum(phizp, G)

    indp = phi > 0
    indn = phi < 0
    G[indp] = xp.sqrt(
        xp.maximum(ap[indp] ** 2, bm[indp] ** 2) +
        xp.maximum(cp[indp] ** 2, dm[indp] ** 2) +
        xp.maximum(ep[indp] ** 2, fm[indp] ** 2)
    ) - 1.0
    G[indn] = xp.sqrt(
        xp.maximum(am[indn] ** 2, bp[indn] ** 2) +
        xp.maximum(cm[indn] ** 2, dp[indn] ** 2) +
        xp.maximum(em[indn] ** 2, fp[indn] ** 2)
    ) - 1.0

    SGN = phi / xp.sqrt(phi ** 2 + 1.0)
    return phi - tau * SGN * G


def _init_checkerboard_3d(shape: tuple, xp) -> np.ndarray:
    """Checkerboard level-set initialisation for 3D volumes."""
    D, H, W = shape
    d = np.arange(D)[:, None, None]
    h = np.arange(H)[None, :, None]
    w = np.arange(W)[None, None, :]
    phi = np.where((d + h + w) % 2 == 0, 1.0, -1.0).astype(np.float64)
    if xp is not np:
        phi = xp.asarray(phi)
    return phi


# ---------------------------------------------------------------------------
# Public function
# ---------------------------------------------------------------------------

def chan_vese_3d(
    volume: np.ndarray,
    mu: float = 0.25,
    lambda1: float = 1.0,
    lambda2: float = 1.0,
    tol: float = 1e-3,
    max_iter: int = 500,
    dt: float = 0.5,
    sigma: float = 3.0,
    reinit_interval: int = 5,
    gpu_available: bool = False,
) -> tuple:
    """3D Robust Chan-Vese segmentation.

    Segments a 3D volume using the robust Chan-Vese level-set model with
    local Gaussian-filtered intensity energy terms for noise robustness.

    Parameters
    ----------
    volume:
        3-D numpy array of shape (D, H, W), dtype float64, normalised to [0, 1].
    mu:
        Length/area regularisation weight.
    lambda1:
        Inside region energy weight.
    lambda2:
        Outside region energy weight.
    tol:
        Convergence tolerance on mean |Δφ|.
    max_iter:
        Maximum number of iterations.
    dt:
        Time-step for the level-set evolution.
    sigma:
        Standard deviation of the Gaussian kernel K_σ used for local
        intensity modelling (replaces explicit convolution).
    reinit_interval:
        Reinitialise φ every this many iterations (Sussman method).
    gpu_available:
        When True, use CuPy for GPU-accelerated computation.

    Returns
    -------
    tuple
        ``(segmentation, [phi])`` — the same format as :func:`chan_vese` in
        *cv_single.py*.  ``segmentation = [R1, R2]`` where R1 is the inside
        mask and R2 the outside mask.  ``phi`` has the same shape as
        *volume* and uses uSCMAN sign convention (φ > 0 inside).
    """
    if volume.ndim != 3:
        raise ValueError("Input volume must be a 3D array (D, H, W).")

    if gpu_available:
        import cupy as xp
    else:
        import numpy as xp

    try:
        from scipy.ndimage import gaussian_filter as _gauss_filter
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "scipy is required for chan_vese_3d. "
            "Install it with: pip install scipy"
        ) from exc

    eps = 1.0  # regularisation parameter for H_ε and δ_ε

    # Normalise to [0, 1]
    vol = xp.asarray(volume.astype(np.float64))
    vmin, vmax = xp.min(vol), xp.max(vol)
    if vmax > vmin:
        vol = (vol - vmin) / (vmax - vmin)

    # Initialise level set
    phi = _init_checkerboard_3d(vol.shape, xp)

    phivar = tol + 1.0
    for i in range(max_iter):
        if phivar <= tol:
            break

        oldphi = phi.copy()

        # Regularised Heaviside
        H = _heaviside(phi, eps, xp)

        # Global mean intensities (uSCMAN: phi>0 is "inside" / H≈1 region)
        c1, c2 = _compute_averages_3d(vol, H, xp)

        # Local energy terms via Gaussian filtering (robust variant)
        if gpu_available:
            try:
                from cupyx.scipy.ndimage import gaussian_filter as _gpu_gauss
                c1_val = float(xp.asnumpy(c1))
                c2_val = float(xp.asnumpy(c2))
                e1 = _gpu_gauss((vol - c1_val) ** 2, sigma=sigma)
                e2 = _gpu_gauss((vol - c2_val) ** 2, sigma=sigma)
            except ImportError:
                # Fall back to CPU scipy if cupyx is not available
                vol_np = xp.asnumpy(vol)
                c1_val = float(xp.asnumpy(c1))
                c2_val = float(xp.asnumpy(c2))
                e1 = xp.asarray(_gauss_filter((vol_np - c1_val) ** 2, sigma=sigma))
                e2 = xp.asarray(_gauss_filter((vol_np - c2_val) ** 2, sigma=sigma))
        else:
            c1_val = float(c1)
            c2_val = float(c2)
            e1 = _gauss_filter((vol - c1_val) ** 2, sigma=sigma)
            e2 = _gauss_filter((vol - c2_val) ** 2, sigma=sigma)

        # Curvature (mean curvature flow)
        kappa = _curvature_3d(phi, xp)

        # Delta
        delta = _delta(phi, eps, xp)

        # Level-set evolution
        phi = phi + dt * delta * (mu * kappa - lambda1 * e1 + lambda2 * e2)

        # Sussman reinitialization
        if (i + 1) % reinit_interval == 0:
            phi = _sussman_reinit_3d(phi, tau=0.5, xp=xp)

        phivar = float(xp.sqrt(xp.mean((phi - oldphi) ** 2)))

    # Build output (same format as cv_single.chan_vese)
    if gpu_available:
        phi_out = xp.asnumpy(phi)
    else:
        phi_out = np.asarray(phi)

    R1 = (phi_out >= 0).astype(np.int32)
    R2 = (phi_out < 0).astype(np.int32)
    segmentation = [R1, R2]

    return segmentation, [phi_out]
