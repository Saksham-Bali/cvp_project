"""cvp: code for the CVP blur-3DGS project (RealX3D, Stage 1 geometry + Stage 2 densification).

Modules
  data     RealX3D scene loading: cameras in the metric laser ("world") frame, images, GT, depth
  geom     projection, depth fusion, visibility, per-stratum accuracy/completeness/F-score
  strata   curvature and proximity strata on the laser ground truth
  viz      figures: overlays, colour maps, coloured PLY export
  published  read RealX3D's published baseline depth maps (min-max normalised) and align them
"""
__version__ = "0.1.0"
