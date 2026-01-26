"""
Union Example: Two overlapping spheres

Mathematical expectation:
- Two overlapping spheres: S1 at (-0.2, 0, 0) radius 0.3, S2 at (0.2, 0, 0) radius 0.3
- Union = min(S1, S2) at each point
- At origin (0,0,0):
  - S1 distance = sqrt(0.2^2) - 0.3 = 0.2 - 0.3 = -0.1 (inside)
  - S2 distance = sqrt(0.2^2) - 0.3 = -0.1 (inside)
  - Union = min(-0.1, -0.1) = -0.1 (inside, in overlap region)
- At (-0.2, 0, 0) (center of S1):
  - S1 distance = 0 - 0.3 = -0.3 (inside)
  - S2 distance = sqrt(0.4^2) - 0.3 = 0.4 - 0.3 = 0.1 (outside)
  - Union = min(-0.3, 0.1) = -0.3 (inside, as expected)
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
    
    # Marching cubes to extract isosurface
    verts, faces, _, _ = measure.marching_cubes(
        values, level=0.0, spacing=(spacing, spacing, spacing)
    )
    
    # Shift vertices to actual coordinates
    verts += np.array([lo, lo, lo])
    
    i, j, k = faces.T
    
    fig = go.Figure(data=[
        go.Mesh3d(
            x=verts[:, 2],  # x
            y=verts[:, 1],  # y
            z=verts[:, 0],  # z
            i=i, j=j, k=k,
            opacity=1.0,
            color='gold',
            flatshading=True
        )
    ])
    
    fig.update_layout(
        title=f"{name} (SDF=0 isosurface)",
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            aspectmode="cube",
        ),
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

        # Create two overlapping spheres (centers closer so they merge)
        s1 = lib.sphere(center=(-0.2, 0.0, 0.0), radius=0.3)
        s2 = lib.sphere(center=(0.2, 0.0, 0.0), radius=0.3)
        union = lib.union(s1, s2)

        # Gather values for verification
        all_vals = []
        for mfi in union:
            arr = union.array(mfi).to_numpy()
            vals = arr[..., 0] if arr.ndim == 4 else arr[..., 0, 0]
            all_vals.append(vals.flatten())
        phi = np.concatenate(all_vals)
        
        # Generate 3D visualization
        if HAS_VIZ:
            n = 64
            full_array = gather_multifab_to_array(union, (n, n, n))
            save_3d_html(full_array, "union_example", (-1, 1))

        # Expected checks
        print("=" * 60)
        print("UNION EXAMPLE: Two overlapping spheres")
        print("=" * 60)
        print(f"Min value (should be < 0, inside): {phi.min():.6f}")
        print(f"Max value (should be > 0, outside): {phi.max():.6f}")
        print(f"Has negative values (inside): {(phi < 0).any()}")
        print(f"Has positive values (outside): {(phi > 0).any()}")
        print(f"Has near-zero (surface): {(np.abs(phi) < 0.05).any()}")

        # Mathematical verification at origin
        # At (0,0,0): distance to S1 center = 0.2, so S1 SDF = 0.2 - 0.3 = -0.1
        # Similarly S2 SDF = -0.1, so union = min(-0.1, -0.1) = -0.1
        # We expect some cells near origin to have negative values (inside overlap)
        near_origin = (np.abs(phi + 0.1) < 0.1).any()
        print(f"Has values near expected origin value (-0.1, inside overlap): {near_origin}")

        # Success criteria
        success = (
            phi.min() < 0 and
            phi.max() > 0 and
            (phi < 0).any() and
            (phi > 0).any()
        )
        print("\n" + "=" * 60)
        if success:
            print("✅ UNION TEST PASSED: Output matches expected behavior")
        else:
            print("❌ UNION TEST FAILED: Unexpected values")
        print("=" * 60)

    finally:
        amr.finalize()


if __name__ == "__main__":
    main()
