# Qualcomm Edge Path

Qualcomm edge devices can run optimized vision models on local CPU/GPU/NPU targets.

Production path:

1. Convert model to a supported Qualcomm AI runtime format.
2. Validate numerical parity against the source model.
3. Package model, preprocessing, postprocessing, and metadata.
4. Roll out by device cohort.
5. Track latency, thermal behavior, device health, and detection quality.
6. Support offline mode with signed event buffering and later cloud sync.
