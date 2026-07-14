"""
Galvo -> chamber registration.

One-time calibration mapping the firmware's galvo scan mask (bed
coordinates) onto chamber-camera pixels: rotate the chamber frame 90 deg
CLOCKWISE first (constants.CHAMBER_ROTATE — aligns it with the galvo
orientation), then fisheye/distortion correction, then a homography fit.
Produces the warped part-region mask used as the spatial prior for
detection. Orient the output so the recoat axis matches the Peregrine
convention (streak features transfer axis-aligned).
"""
