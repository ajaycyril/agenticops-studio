# YOLO Edge Path

Use YOLO for local smoke/fire/person detection on gateway hardware or camera-side compute.

Implementation path:

1. Train or fine-tune a smoke/fire detector.
2. Export to ONNX or vendor runtime format.
3. Package with confidence thresholds, class mapping, and model metadata.
4. Run inference locally and emit metadata, not raw video, by default.
5. Buffer evidence pointers for operator review.
6. Sync model version, device health, and inference metrics to cloud.

Edge and cloud should work together: edge provides low-latency perception and cloud provides governance, policy, audit, and fleet coordination.
