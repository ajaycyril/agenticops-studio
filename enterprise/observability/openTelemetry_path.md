# OpenTelemetry Path

Instrument device ingestion, model inference, agent orchestration, policy checks, approval decisions, tool execution, and decision-record writes.

Recommended spans:

- `device.event.received`
- `vision.inference`
- `risk.prediction`
- `agent.plan`
- `policy.evaluate`
- `approval.resolve`
- `tool.execute`
- `decision_record.write`

Attach incident ID, run ID, model version, policy version, tool name, execution mode, and status. Do not attach secrets or raw images.
