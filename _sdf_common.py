"""Shared SDF helpers used by both sdf2d and sdf3d.

This module provides:

* **Type alias**: :data:`_F`
* **Vector constructors**: :func:`vec2`, :func:`vec3`
* **Math helpers**: :func:`length`, :func:`dot`, :func:`dot2`, :func:`clamp`,
  :func:`safe_div`
* **Shared boolean/domain operators** (used by both 2D and 3D geometry):
  :func:`opUnion`, :func:`opSubtraction`, :func:`opIntersection`,
  :func:`opRound`, :func:`opOnion`, :func:`opScale`

Not meant to be imported directly by end users — import from
``sdf2d.primitives`` or ``sdf3d.primitives`` instead.
"""

from __future__ import annotations
from typing import TypeAlias
import numpy as np
import numpy.typing as npt

# ---------------------------------------------------------------------------
# Type alias
# ---------------------------------------------------------------------------
FloatArray: TypeAlias = npt.NDArray[np.floating]
Distances: TypeAlias = npt.NDArray[np.floating]  # shape (...)

__all__ = [
    "FloatArray",
    "vec2", "vec3",
    "length", "dot", "dot2", "clamp", "safe_div",
    "opUnion", "opSubtraction", "opIntersection",
    "opRound", "opOnion", "opScale",
]


# ===========================================================================
# Vector constructors
# ===========================================================================

def vec2(x: FloatArray, y: FloatArray) -> FloatArray:
    """Stack *x* and *y* into a ``(..., 2)`` array."""
    x, y = np.broadcast_arrays(x, y)
    return np.stack([x, y], axis=-1)


def vec3(x: FloatArray, y: FloatArray, z: FloatArray) -> FloatArray:
    """Stack *x*, *y*, *z* into a ``(..., 3)`` array."""
    x, y, z = np.broadcast_arrays(x, y, z)
    return np.stack([x, y, z], axis=-1)


# ===========================================================================
# Math helpers
# ===========================================================================

def length(v: FloatArray) -> FloatArray:
    """Euclidean length along the last axis."""
    return np.linalg.norm(v, axis=-1)


def dot(a: FloatArray, b: FloatArray) -> FloatArray:
    """Dot product along the last axis."""
    return np.sum(a * b, axis=-1)


def dot2(a: FloatArray) -> FloatArray:
    """Squared length: ``dot(a, a)``."""
    return dot(a, a)


def clamp(x: FloatArray, lo: float | FloatArray, hi: float | FloatArray) -> FloatArray:
    """Clamp *x* element-wise to ``[lo, hi]``."""
    return np.minimum(np.maximum(x, lo), hi)


def safe_div(n: FloatArray, d: FloatArray, eps: float = 1e-12) -> FloatArray:
    """Division that avoids exact zero in the denominator."""
    return n / np.where(np.abs(d) < eps, np.sign(d) * eps + eps, d)


# ===========================================================================
# Shared boolean / domain operators (used by both sdf2d and sdf3d)
# ===========================================================================

def opUnion(d1: FloatArray, d2: FloatArray) -> FloatArray:
    """Union of two SDFs: ``min(d1, d2)``."""
    return np.minimum(d1, d2)


def opSubtraction(d1: FloatArray, d2: FloatArray) -> FloatArray:
    """Subtract *d1* from *d2*: ``max(-d1, d2)``."""
    return np.maximum(-d1, d2)


def opIntersection(d1: FloatArray, d2: FloatArray) -> FloatArray:
    """Intersection of two SDFs: ``max(d1, d2)``."""
    return np.maximum(d1, d2)


def opRound(p: FloatArray, primitive: "_SDFFunc", rad: float) -> FloatArray:  # type: ignore[name-defined]
    """Round a primitive outward by *rad*."""
    return primitive(p) - rad


def opOnion(sdf_val: FloatArray, thickness: float) -> FloatArray:
    """Turn a solid into a shell of *thickness*."""
    return np.abs(sdf_val) - thickness


def opScale(p: FloatArray, s: float, primitive: "_SDFFunc") -> FloatArray:  # type: ignore[name-defined]
    """Uniformly scale a primitive by factor *s*."""
    return primitive(p / s) * s
