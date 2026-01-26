"""
Intersection Example: Two overlapping spheres

Mathematical expectation:
- Two spheres: S1 at (0, 0, 0) radius 0.35, S2 at (0.2, 0, 0) radius 0.35
- Intersection = max(S1, S2) at each point
- At origin (0,0,0):
  - S1 distance = 0 - 0.35 = -0.35 (inside)
  - S2 distance = sqrt(0.2^2) - 0.35 = 0.2 - 0.35 = -0.15 (inside)
  - Intersection = max(-0.35, -0.15) = -0.15 (inside, but less negative)
- At (0.1, 0, 0) (midpoint):
  - S1 distance = sqrt(0.1^2) - 0.35 = 0.1 - 0.35 = -0.25 (inside)
  - S2 distance = sqrt(0.1^2) - 0.35 = -0.25 (inside)
  - Intersection = max(-0.25, -0.25) = -0.25 (inside)
- Far outside: both positive, intersection = max(positive, positive) = positive
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import amrex.space3d as amr
from sdf3d import SDFLibrary
import numpy as np
import os

try:
    from skimage import measure
    import plotly.graph_objects as go
    HAS_VIZ = True
except ImportError:
    HAS_VIZ = False
    print("⚠️  plotly/scikit-image not available, skipping 3D visualization")


def gather_multifab_to_array(mf, shape):
    """Convert MultiFab to full numpy array"""
    full = np.zeros(shape, dtype=np.float32)
    for mfi in mf:
        arr = mf.array(mfi).to_numpy()
        vals = arr[..., 0] if arr.ndim == 4 else arr[..., 0, 0]
        bx = mfi.validbox()
        i_lo, j_lo, k_lo = bx.lo_vect
        i_hi, j_hi, k_hi = bx.hi_vect
        full[k_lo:k_hi+1, j_lo:j_hi+1, i_lo:i_hi+1] = vals
    return full


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
            i=i, j=j, k=k, opacity=1.0, color='lightblue', flatshading=True
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
    amr.initialize([])
    try:
        # Setup grid
        real_box = amr.RealBox([-1, -1, -1], [1, 1, 1])
        domain = amr.Box(amr.IntVect(0, 0, 0), amr.IntVect(63, 63, 63))
        geom = amr.Geometry(domain, real_box, 0, [0, 0, 0])
        ba = amr.BoxArray(domain)
        ba.max_size(32)
        dm = amr.DistributionMapping(ba)

        lib = SDFLibrary(geom, ba, dm)

        # Create two overlapping spheres
        s1 = lib.sphere(center=(0.0, 0.0, 0.0), radius=0.35)
        s2 = lib.sphere(center=(0.2, 0.0, 0.0), radius=0.35)
        inter = lib.intersect(s1, s2)

        # Gather values
        all_vals = []
        for mfi in inter:
            arr = inter.array(mfi).to_numpy()
            vals = arr[..., 0] if arr.ndim == 4 else arr[..., 0, 0]
            all_vals.append(vals.flatten())
        phi = np.concatenate(all_vals)

        print("=" * 60)
        print("INTERSECTION EXAMPLE: Two overlapping spheres")
        print("=" * 60)
        print(f"Min value (should be < 0, inside overlap): {phi.min():.6f}")
        print(f"Max value (should be > 0, outside both): {phi.max():.6f}")
        print(f"Has negative values (inside): {(phi < 0).any()}")
        print(f"Has positive values (outside): {(phi > 0).any()}")

        # Mathematical check: intersection should be >= max of individual spheres
        s1_vals = []
        s2_vals = []
        for mfi in s1:
            arr = s1.array(mfi).to_numpy()
            vals = arr[..., 0] if arr.ndim == 4 else arr[..., 0, 0]
            s1_vals.append(vals.flatten())
        for mfi in s2:
            arr = s2.array(mfi).to_numpy()
            vals = arr[..., 0] if arr.ndim == 4 else arr[..., 0, 0]
            s2_vals.append(vals.flatten())
        s1_phi = np.concatenate(s1_vals)
        s2_phi = np.concatenate(s2_vals)

        expected_inter = np.maximum(s1_phi, s2_phi)
        max_diff = np.abs(phi - expected_inter).max()
        print(f"Max difference from expected (max(s1, s2)): {max_diff:.6e}")
        
        # Generate 3D visualization
        if HAS_VIZ:
            n = 64
            full_array = gather_multifab_to_array(inter, (n, n, n))
            save_3d_html(full_array, "intersection_example", (-1, 1))

        success = (
            phi.min() < 0 and
            phi.max() > 0 and
            max_diff < 1e-5  # Should match exactly
        )
        print("\n" + "=" * 60)
        if success:
            print("✅ INTERSECTION TEST PASSED: Matches max(S1, S2) exactly")
        else:
            print("❌ INTERSECTION TEST FAILED")
        print("=" * 60)

    finally:
        amr.finalize()


if __name__ == "__main__":
    main()
