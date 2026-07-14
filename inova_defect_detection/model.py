"""Detection model.

v1: ~4M-param encoder-decoder (SegFormer-B0 or MobileNetV3-encoder U-Net)
over stacked per-layer channels (chamber frame, warped galvo mask,
frame-diff vs previous layer, upsampled bedmatrix), with a per-region
classification head (part vs powder regions from the galvo mask).
v2: dense segmentation head once pixel labels exist.
Export target: ONNX for CPU inference on the recorder host.
"""
