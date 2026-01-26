"""
Subtraction Example: Sphere with a hole cut out

Mathematical expectation:
- Base sphere at (0, 0, 0) radius 0.4
- Cutter sphere at (0.2, 0, 0) radius 0.25
- Subtraction = max(-base, cutter) = max(-base, cutter)
- At origin (0,0,0):
  - Base distance = 0 - 0.4 = -0.4 (inside)
  - Cutter distance = sqrt(0.2^2) - 0.25 = 0.2 - 0.25 = -0.05 (inside cutter)
  - Subtraction = max(-(-0.4), -0.05) = max(0.4, -0.05) = 0.4 (outside result)
- At (0.2, 0, 0) (cutter center):
  - Base distance = sqrt(0.2^2) - 0.4 = 0.2 - 0.4 = -0.2 (inside base)
  - Cutter distance = 0 - 0.25 = -0.25 (inside cutter)
  - Subtraction = max(0.2, -0.25) = 0.2 (outside result, hole created)
- At (0.5, 0, 0) (far from both):
  - Base distance = 0.5 - 0.4 = 0.1 (outside)
  - Cutter distance = sqrt(0.3^2) - 0.25 = 0.3 - 0.25 = 0.05 (outside)
  - Subtraction = max(0.1, 0.05) = 0.1 (outside)
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
    
    # Filter out small disconnected fragments (artifacts from low resolution)
    # Simple approach: remove vertices that appear in very few faces (likely isolated)
    if len(verts) > 0 and len(faces) > 0:
        vertex_face_count = np.zeros(len(verts))
        for face in faces:
            vertex_face_count[face] += 1
        
        # Keep vertices that appear in at least 3 faces (removes tiny fragments)
        # But only if we're not removing too much of the mesh
        min_faces = max(2, int(len(faces) / len(verts) * 0.1))  # Adaptive threshold
        valid_vertices = vertex_face_count >= min_faces
        
        if valid_vertices.sum() > len(verts) * 0.3:  # Only filter if we keep >30% of vertices
            # Remap vertex indices
            vertex_map = np.full(len(verts), -1, dtype=int)
            new_idx = 0
            for i, valid in enumerate(valid_vertices):
                if valid:
                    vertex_map[i] = new_idx
                    new_idx += 1
            
            # Filter faces: keep only faces where all vertices are valid
            valid_faces = valid_vertices[faces].all(axis=1)
            faces = faces[valid_faces]
            
            # Remap vertex indices in faces
            faces = np.array([[vertex_map[v] for v in face] for face in faces])
            
            # Filter vertices
            verts = verts[valid_vertices]
    
    i, j, k = faces.T
    
    fig = go.Figure(data=[
        go.Mesh3d(
            x=verts[:, 2], y=verts[:, 1], z=verts[:, 0],
            i=i, j=j, k=k, opacity=1.0, color='coral', flatshading=True
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
        # Setup grid (higher resolution for better visualization)
        real_box = amr.RealBox([-1, -1, -1], [1, 1, 1])
        domain = amr.Box(amr.IntVect(0, 0, 0), amr.IntVect(127, 127, 127))  # 128^3 for smoother visualization
        geom = amr.Geometry(domain, real_box, 0, [0, 0, 0])
        ba = amr.BoxArray(domain)
        ba.max_size(32)
        dm = amr.DistributionMapping(ba)

        lib = SDFLibrary(geom, ba, dm)

        # Create base sphere and cutter
        base = lib.sphere(center=(0.0, 0.0, 0.0), radius=0.4)
        cutter = lib.sphere(center=(0.2, 0.0, 0.0), radius=0.25)
        sub = lib.subtract(base, cutter)

        # Gather values
        all_vals = []
        for mfi in sub:
            arr = sub.array(mfi).to_numpy()
            vals = arr[..., 0] if arr.ndim == 4 else arr[..., 0, 0]
            all_vals.append(vals.flatten())
        phi = np.concatenate(all_vals)

        print("=" * 60)
        print("SUBTRACTION EXAMPLE: Sphere with hole cut out")
        print("=" * 60)
        print(f"Min value: {phi.min():.6f}")
        print(f"Max value: {phi.max():.6f}")
        print(f"Has negative values: {(phi < 0).any()}")
        print(f"Has positive values: {(phi > 0).any()}")

        # Mathematical verification: subtraction = max(-base, cutter)
        base_vals = []
        cutter_vals = []
        for mfi in base:
            arr = base.array(mfi).to_numpy()
            vals = arr[..., 0] if arr.ndim == 4 else arr[..., 0, 0]
            base_vals.append(vals.flatten())
        for mfi in cutter:
            arr = cutter.array(mfi).to_numpy()
            vals = arr[..., 0] if arr.ndim == 4 else arr[..., 0, 0]
            cutter_vals.append(vals.flatten())
        base_phi = np.concatenate(base_vals)
        cutter_phi = np.concatenate(cutter_vals)

        expected_sub = np.maximum(-base_phi, cutter_phi)
        max_diff = np.abs(phi - expected_sub).max()
        print(f"Max difference from expected max(-base, cutter): {max_diff:.6e}")

        success = (
            max_diff < 1e-5 and
            (phi > 0).any()  # Should have outside regions
        )
        print("\n" + "=" * 60)
        if success:
            print("✅ SUBTRACTION TEST PASSED: Matches max(-base, cutter) exactly")
        else:
            print("❌ SUBTRACTION TEST FAILED")
        print("=" * 60)
        
        # Generate 3D visualization
        if HAS_VIZ:
            n = 128  # Higher resolution to reduce artifacts
            full_array = gather_multifab_to_array(sub, (n, n, n))
            save_3d_html(full_array, "subtraction_example", (-1, 1))

    finally:
        amr.finalize()


if __name__ == "__main__":
    main()
