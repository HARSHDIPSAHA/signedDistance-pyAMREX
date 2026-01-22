# Signed Distance Functions with pyAMReX

This repo implements the signed distance functions and operators listed on
the Inigo Quilez "Distance Functions" article, and evaluates them with
pyAMReX on a 2D grid (z = 0 slice) to generate visualization PNGs. The
visualization workflow uses pyAMReX's `MultiFab` to fill SDF values on a
structured grid. [pyAMReX](https://pyamrex.readthedocs.io/en/latest/) and
the original SDF formulas are referenced from
[iquilezles.org](https://iquilezles.org/articles/distfunctions/).

## Files

- `sdf_lib.py`: numpy implementations of the SDF primitives and operators.
- `render_all_sdfs.py`: builds a 2D AMReX grid, evaluates each SDF, and saves
  images into `vis/`.

## Run

```bash
python render_all_sdfs.py
```

## Output

Each SDF produces a `vis/<name>.png` visualization with a diverging colormap
and the zero level-set contour drawn in black.

## Notes

- All 3D SDFs are evaluated on the z = 0 slice for visualization.
- `udTriangle` and `udQuad` are unsigned distance fields by definition.
