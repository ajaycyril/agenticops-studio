# Enterprise Extension

The Vercel app is the interactive reference demo. The enterprise extension is the production blueprint.

Production deployment would add:

- Event streaming with Kafka or Azure Event Hubs.
- MQTT device ingestion for sensors, cameras, gateways, gates, and drones.
- Edge model deployment with ONNX, YOLO, and Qualcomm NPU paths.
- MLflow model registry and model promotion workflow.
- OPA/Rego service for policy evaluation.
- OpenTelemetry, Grafana, Tempo, and Loki for traces and logs.
- Postgres event store for durable incident state.
- Vector database for SOP retrieval and policy-aware document access.
- Kubernetes for long-running services and internal APIs.
- Enterprise identity, RBAC, break-glass controls, and dual approval.
- Immutable audit logs for decision records and tool execution.
- Real authority integration through governed APIs and legal agreements.
- Real command center workflow with operator queues and escalation playbooks.

The public demo intentionally keeps drone, gate, and authority actions in sandbox mode. A production system would connect those tools through explicit contracts, policy checks, approval workflows, replayable traces, and audited execution ledgers.
