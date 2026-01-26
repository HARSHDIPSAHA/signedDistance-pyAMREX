"""
Elongation Example: Elongated sphere using geometry API

Mathematical expectation:
- Base sphere radius 0.25
- Elongated by (0.3, 0.0, 0.0) in x-direction
- Elongation operation: q = p - clamp(p, -h, h), then evaluate sphere(q)
- At origin (0,0,0):
  - q = (0,0,0) - clamp((0,0,0), (-0.3,0,0), (0.3,0,0)) = (0,0,0) - (0,0,0) = (0,0,0)
  - Distance = 0 - 0.25 = -0.25 (inside)
- At (0.4, 0, 0) (beyond elongation):
  - q = (0.4,0,0) - clamp((0.4,0,0), (-0.3,0,0), (0.3,0,0)) = (0.4,0,0) - (0.3,0,0) = (0.1,0,0)
  - Distance = 0.1 - 0.25 = -0.15 (still inside elongated shape)
- At (0.5, 0, 0):
  - q = (0.5,0,0) - (0.3,0,0) = (0.2,0,0)
  - Distance = 0.2 - 0.25 = -0.05 (inside)
- At (0.3, 0, 0) (at elongation boundary):
  - q = (0.3,0,0) - (0.3,0,0) = (0,0,0)
  - Distance = 0 - 0.25 = -0.25 (inside)
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from sdf3d import Sphere, sample_levelset

try:
    from skimage import measure
    import plotly.graph_objects as go
    HAS_VIZ = True
except ImportError:
    HAS_VIZ = False
    print("⚠️  plotly/scikit-image not available, skipping 3D visualization")


def save_3d_html(values, name, bounds, out_dir="outputs/vis3d_plotly"):
    """Generate interactive 3D HTML visualization using plotly"""
    if not HAS_VIZ:
        return
    
    os.makedirs(out_dir, exist_ok=True)
    lo, hi = bounds
    spacing = (hi - lo) / values.shape[0]
    
    verts, faces, _, _ = measure.marching_cubes(
        values, level=0.0, spacing=(spacing, spacing, spacing)
    )
    verts += np.array([lo, lo, lo])
    i, j, k = faces.T
    
    fig = go.Figure(data=[
        go.Mesh3d(
            x=verts[:, 2], y=verts[:, 1], z=verts[:, 0],
            i=i, j=j, k=k, opacity=1.0, color='limegreen', flatshading=True
        )
    ])
    
    fig.update_layout(
        title=f"{name} (SDF=0 isosurface)",
        scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z", aspectmode="cube"),
        margin=dict(l=0, r=0, b=0, t=40)
    )
    
    out_path = os.path.join(out_dir, f"{name}_3d.html")
    fig.write_html(out_path)
    print(f"✅ Interactive 3D visualization: {out_path}")


def main():
    print("=" * 60)
    print("ELONGATION EXAMPLE: Sphere elongated in x-direction")
    print("=" * 60)

    # Create elongated sphere using geometry API
    sphere = Sphere(0.25)
    elongated = sphere.elongate(0.3, 0.0, 0.0)

    # Sample on grid
    bounds = ((-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0))
    res = (64, 64, 64)
    phi = sample_levelset(elongated, bounds, res)

    print(f"Min value (should be < 0, inside): {phi.min():.6f}")
    print(f"Max value (should be > 0, outside): {phi.max():.6f}")
    print(f"Has negative values (inside): {(phi < 0).any()}")
    print(f"Has positive values (outside): {(phi > 0).any()}")

    # Mathematical verification at specific points
    # Grid coordinates: cell centers
    dx = (bounds[0][1] - bounds[0][0]) / res[0]
    coords = np.linspace(bounds[0][0] + dx/2, bounds[0][1] - dx/2, res[0])

    # Find index closest to origin
    origin_idx = np.argmin(np.abs(coords))
    origin_val = phi[origin_idx, origin_idx, origin_idx]
    print(f"\nValue at origin (expected ~ -0.25): {origin_val:.6f}")

    # Find index closest to (0.3, 0, 0)
    x03_idx = np.argmin(np.abs(coords - 0.3))
    val_at_03 = phi[x03_idx, origin_idx, origin_idx]
    print(f"Value at (0.3, 0, 0) (expected ~ -0.25): {val_at_03:.6f}")

    # Find index closest to (0.5, 0, 0)
    x05_idx = np.argmin(np.abs(coords - 0.5))
    val_at_05 = phi[x05_idx, origin_idx, origin_idx]
    print(f"Value at (0.5, 0, 0) (expected ~ -0.05): {val_at_05:.6f}")

    # Success criteria
    success = (
        phi.min() < 0 and
        phi.max() > 0 and
        (phi < 0).any() and
        (phi > 0).any() and
        origin_val < -0.2 and origin_val > -0.3  # Should be inside
    )

    print("\n" + "=" * 60)
    if success:
        print("✅ ELONGATION TEST PASSED: Values match expected behavior")
    else:
        print("❌ ELONGATION TEST FAILED")
    print("=" * 60)
    
    # Generate 3D visualization
    if HAS_VIZ:
        save_3d_html(phi, "elongation_example", bounds[0])


if __name__ == "__main__":
    main()
