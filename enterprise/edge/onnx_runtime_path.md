# ONNX Runtime Path

ONNX Runtime is a practical route for portable edge inference.

Production path:

1. Export the detector to ONNX.
2. Validate preprocessing and confidence thresholds.
3. Run on edge gateways with CPU, GPU, or NPU acceleration where available.
4. Emit inference metadata, confidence scores, evidence pointers, and model version.
5. Keep cloud control-plane policy and decision records authoritative.
6. Use fleet rollout rings and rollback support for model changes.
