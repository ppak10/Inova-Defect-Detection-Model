"""Galvo->chamber registration.

One-time calibration mapping the firmware's galvo scan mask (bed
coordinates) onto chamber-camera pixels: fisheye/distortion correction of
the chamber view, then a homography fit. Produces the warped part-region
mask used as the spatial prior for detection.
"""
