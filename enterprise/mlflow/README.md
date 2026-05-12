# MLflow Path

Use MLflow for model registration, model promotion, lineage, metrics, and rollback.

Recommended production flow:

1. Train risk and vision models in controlled pipelines.
2. Register candidates with dataset hash, feature schema, metrics, and explainability artifacts.
3. Gate promotion with precision, recall, false positive rate, calibration, and scenario replay tests.
4. Deploy approved versions to cloud inference and edge packaging jobs.
5. Monitor drift, latency, false alarms, and operator overrides.
