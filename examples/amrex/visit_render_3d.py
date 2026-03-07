"""VisIt CLI script — render a 3D AMReX SDF plotfile to PNG (isosurface at SDF=0).

NOT imported as a module.  Run by amrex_demo_3d.py via subprocess:

    visit -cli -nowin -s visit_render_3d.py

Reads two environment variables set by the calling script:
    VISIT_PF_DIR   absolute path to the AMReX plotfile directory
    VISIT_PNG_OUT  absolute path for the output PNG

In VisIt -cli mode all VisIt API calls are *global functions* (no 'visit.' prefix).
"""

import os
import sys

pf_dir  = os.environ["VISIT_PF_DIR"]
png_out = os.environ["VISIT_PNG_OUT"]

import os as _os
header = _os.path.join(pf_dir, "Header")
OpenDatabase(header, 0, "Boxlib3D")

# -----------------------------------------------------------------------
# Isosurface at SDF = 0: this is the exact shape surface.
# "Contour" in VisIt means isosurface for 3D data.
# -----------------------------------------------------------------------
AddPlot("Contour", "sdf")
ct = ContourAttributes()
ct.SetContourMethod(1)              # 0 = levels, 1 = value list
ct.contourValue = (0.0,)            # the shape boundary
ct.colorType = ct.ColorBySingleColor
ct.singleColor = (100, 149, 237, 255)  # cornflower blue
SetPlotOptions(ct)

DrawPlots()

# -----------------------------------------------------------------------
# 3D view: angled camera so depth is visible
# -----------------------------------------------------------------------
v = GetView3D()
v.viewNormal   = (0.5, -0.35, 0.8)   # camera direction vector
v.viewUp       = (-0.2, 0.9, 0.3)    # up vector
v.parallelScale = 0.4
SetView3D(v)

# -----------------------------------------------------------------------
# Clean up annotations (no axis labels, no database path in the image)
# -----------------------------------------------------------------------
a = GetAnnotationAttributes()
a.axes3D.visible    = 0
a.userInfoFlag      = 0
a.databaseInfoFlag  = 0
SetAnnotationAttributes(a)

# -----------------------------------------------------------------------
# Save PNG
# -----------------------------------------------------------------------
out_dir  = os.path.dirname(os.path.abspath(png_out))
basename = os.path.splitext(os.path.basename(png_out))[0]

s = SaveWindowAttributes()
s.outputToCurrentDirectory = 0
s.outputDirectory = out_dir
s.fileName        = basename
s.format          = s.PNG
s.width           = 800
s.height          = 800
s.screenCapture   = 0
s.family          = 0   # 0 = exact filename, 1 = append frame number (e.g. sphere0000.png)
SetSaveWindowAttributes(s)
SaveWindow()

sys.exit()
