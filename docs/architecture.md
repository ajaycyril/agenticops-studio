# Architecture

```mermaid
sequenceDiagram
  participant Sensor
  participant Vision
  participant ML
  participant Agent
  participant Policy
  participant Human
  participant Tool
  participant Record
  Sensor->>Vision: Camera frame / sample image
  Vision->>Agent: Detections and confidence
  Sensor->>ML: Incident feature vector
  ML->>Agent: Fire probability and risk class
  Agent->>Policy: Proposed actions
  Policy->>Human: Approval-gated decisions
  Human->>Tool: Approval or rejection
  Tool->>Record: Execution status
  Agent->>Record: Decision rationale
```

## Runtime Boundaries

Browser: scenario state, TF.js training, trace viewer, approvals, decision-record local/session storage.

Server routes: OpenAI, Roboflow, report generation, structured logs, safe error handling.

External services: OpenAI Responses API and Roboflow hosted inference when credentials are configured.
