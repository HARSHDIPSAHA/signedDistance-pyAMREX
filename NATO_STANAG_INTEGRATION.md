# Integration Guide: save_levelset_html with NATO STANAG Example

This document explains how the NATO STANAG 4496 fragment impact test example has been integrated with the new `save_levelset_html` visualization function.

## Overview

The NATO STANAG 4496 example (`examples/nato_stanag_4496_test.py`) demonstrates how to create complex geometries for high-velocity impact simulations. The example has been updated to use the new standardized visualization function instead of custom visualization code.

## What Changed

### Before (Custom Visualization)
The example previously had its own custom `save_3d_html` function (~70 lines of code) that:
- Manually called `measure.marching_cubes`
- Manually filtered disconnected fragments
- Manually created plotly figures
- Duplicated code that now exists in `save_levelset_html`

### After (Standardized Visualization)
The example now uses a simple wrapper `save_3d_html_nato` (~25 lines) that:
- Calls the standardized `save_levelset_html` function
- Adds NATO-specific error handling
- Maintains the same output file naming convention
- Reduces code duplication

## How to Use

### Step 1: Import the visualization function

```python
from sdf3d import save_levelset_html
```

### Step 2: Convert MultiFab to numpy array

If you're working with AMReX MultiFab objects (as in the NATO example), convert them to numpy arrays first:

```python
def gather_multifab_to_array(mf, shape):
    """Convert MultiFab to full numpy array"""
    full = np.zeros(shape, dtype=np.float32)
    for mfi in mf:
        arr = mf.array(mfi).to_numpy()
        vals = arr[..., 0] if arr.ndim == 4 else arr[..., 0, 0]
        bx = mfi.validbox()
        i_lo, j_lo, k_lo = bx.lo_vect
        i_hi, j_hi, k_hi = bx.hi_vect
        full[k_lo : k_hi + 1, j_lo : j_hi + 1, i_lo : i_hi + 1] = vals
    return full

# Usage
fragment_array = gather_multifab_to_array(fragment_mf, (256, 256, 256))
```

### Step 3: Call save_levelset_html

```python
# Define your domain bounds
bounds = (-0.2, 0.2)  # 200mm domain in meters

# Save visualization
save_levelset_html(
    fragment_array,
    bounds=bounds,
    filename="outputs/nato_fragment.html"
)
```

## Complete Example

Here's a minimal example showing how to visualize NATO STANAG fragment geometry:

```python
import amrex.space3d as amr
from sdf3d import SDFLibrary, Cylinder, Box, Intersection, save_levelset_html
import numpy as np

amr.initialize([])
try:
    # Setup grid
    domain_size = 0.2  # 200 mm domain
    real_box = amr.RealBox(
        [-domain_size, -domain_size, -domain_size],
        [domain_size, domain_size, domain_size]
    )
    n = 256
    domain = amr.Box(amr.IntVect(0, 0, 0), amr.IntVect(n-1, n-1, n-1))
    geom = amr.Geometry(domain, real_box, 0, [0, 0, 0])
    ba = amr.BoxArray(domain)
    ba.max_size(32)
    dm = amr.DistributionMapping(ba)
    
    lib = SDFLibrary(geom, ba, dm)
    
    # Create simplified fragment (cylinder)
    fragment_radius = 7.15e-3  # 7.15 mm
    cylinder_height = 10e-3     # 10 mm
    
    cylinder_infinite = Cylinder(axis_offset=[0.0, 0.0], radius=fragment_radius)
    cylinder_box = Box(half_size=[fragment_radius*2, cylinder_height/2, fragment_radius*2])
    cylinder_geom = Intersection(cylinder_infinite, cylinder_box)
    cylinder_geom = cylinder_geom.rotate_x(np.pi/2)
    cylinder_geom = cylinder_geom.translate(0.0, 0.0, cylinder_height/2)
    
    # Convert to MultiFab and then to numpy
    fragment_mf = lib.from_geometry(cylinder_geom)
    fragment_array = gather_multifab_to_array(fragment_mf, (n, n, n))
    
    # Visualize with the new function
    bounds = (-domain_size, domain_size)
    save_levelset_html(
        fragment_array,
        bounds=bounds,
        filename="outputs/nato_fragment.html"
    )
    
    print("✅ Visualization saved!")
    
finally:
    amr.finalize()
```

## Benefits

1. **Code Reuse**: No need to duplicate visualization code across examples
2. **Consistency**: All visualizations use the same styling and features
3. **Maintainability**: Bug fixes and improvements to `save_levelset_html` automatically benefit all examples
4. **Simplicity**: Wrapper function is much shorter and easier to understand
5. **Flexibility**: Can still customize output paths and add domain-specific error handling

## Backward Compatibility

The NATO STANAG example maintains the same:
- Output file names and locations
- Error handling behavior
- Visualization quality
- Public API (function names and parameters)

Users of the NATO example will see no difference except potentially improved visualization quality from any enhancements to the core `save_levelset_html` function.

## Testing

To verify the integration works:

```bash
# Run the NATO STANAG example
python examples/nato_stanag_4496_test.py

# Check that visualizations were created
ls outputs/vis3d_plotly/nato_*_3d.html
```

Expected output files:
- `nato_fragment_3d.html` - Fragment geometry alone
- `nato_target_3d.html` - Target block alone
- `nato_fragment_positioned_3d.html` - Fragment with impact orientation
- `nato_full_domain_3d.html` - Complete test geometry

All visualizations should be interactive 3D meshes viewable in any web browser.
