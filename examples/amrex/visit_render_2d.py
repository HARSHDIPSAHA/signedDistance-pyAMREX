"""VisIt CLI script — render a 2D AMReX SDF plotfile to PNG.

NOT imported as a module.  Run by amrex_demo_2d.py via subprocess:

    visit -cli -nowin -s visit_render_2d.py

Reads two environment variables set by the calling script:
    VISIT_PF_DIR   absolute path to the AMReX plotfile directory
    VISIT_PNG_OUT  absolute path for the output PNG

In VisIt -cli mode all VisIt API calls are *global functions* (no 'visit.' prefix).
"""

import os
import sys

pf_dir  = os.environ["VISIT_PF_DIR"]
png_out = os.environ["VISIT_PNG_OUT"]

# VisIt reads the AMReX BoxLib plotfile via its Header file.
# We must specify "BoxLib" explicitly — VisIt's auto-detection tries PDB/Silo
# first and gives up before reaching the BoxLib reader.
import os as _os
header = _os.path.join(pf_dir, "Header")
OpenDatabase(header, 0, "Boxlib2D")

# -----------------------------------------------------------------------
# Plot 0: Pseudocolor — the full SDF field as a heat map
#   phi < 0 (inside)  → blue
#   phi = 0 (surface) → white
#   phi > 0 (outside) → red
#
# SetActivePlots(0) is required: AddPlot adds to the list but does NOT
# automatically select the new plot for SetPlotOptions.
# -----------------------------------------------------------------------
AddPlot("Pseudocolor", "sdf")
SetActivePlots(0)
pc = PseudocolorAttributes()
pc.colorTableName   = "RdBu"
pc.invertColorTable = 1       # blue = negative (inside), red = positive
pc.minFlag = 1
pc.maxFlag = 1
pc.min = -1.0
pc.max  =  1.0
SetPlotOptions(pc)

# -----------------------------------------------------------------------
# Plot 1: Contour — zero level set = shape boundary
#
# contourMethod = 1 means "Value list" (not N equally-spaced levels).
# colorType = 0   means ColorBySingleColor.
# -----------------------------------------------------------------------
AddPlot("Contour", "sdf")
SetActivePlots(1)
ct = ContourAttributes()
ct.contourMethod = 1           # 0 = Level, 1 = Value, 2 = Percent
ct.contourValue  = (0.0,)      # SDF = 0 is the surface
ct.colorType     = 0           # 0 = ColorBySingleColor
ct.singleColor   = (0, 0, 0, 255)  # black
ct.lineWidth     = 2
SetPlotOptions(ct)

DrawPlots()

# -----------------------------------------------------------------------
# Save PNG.
# VisIt appends the format extension, so strip .png from the filename.
# outputToCurrentDirectory = 0 means use outputDirectory instead.
# Use forward slashes — VisIt on Windows accepts them and they avoid
# escape issues in Python string literals.
# -----------------------------------------------------------------------
out_dir  = os.path.dirname(os.path.abspath(png_out)).replace("\\", "/")
basename = os.path.splitext(os.path.basename(png_out))[0]

s = SaveWindowAttributes()
s.outputToCurrentDirectory = 0
s.outputDirectory = out_dir
s.fileName        = basename
s.format          = s.PNG
s.width           = 800
s.height          = 800
s.screenCapture   = 0
s.family          = 0   # 0 = exact filename, 1 = append frame number (e.g. circle0000.png)
SetSaveWindowAttributes(s)
SaveWindow()

sys.exit()
