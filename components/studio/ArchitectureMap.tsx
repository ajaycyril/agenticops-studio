"use client";

import { useState } from "react";
import { Background, Controls, MiniMap, ReactFlow, type Edge, type Node } from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { JsonInspector } from "@/components/studio/JsonInspector";

const nodeDetails = {
  smoke: { what: "Smoke sensor emits ppm readings.", input: "Physical smoke concentration", output: "smokePpm", production: "MQTT or event stream with device identity and health." },
  heat: { what: "Heat sensor emits temperature readings.", input: "Thermal reading", output: "temperatureC", production: "Calibrated sensor fleet with drift detection." },
  camera: { what: "Camera captures visual evidence.", input: "Image/video", output: "image evidence pointer", production: "On-device privacy filtering and retention policy." },
  edge: { what: "Edge vision model detects smoke/fire near the camera.", input: "Image frame", output: "detections/confidence", production: "YOLO/ONNX runtime on gateway or NPU." },
  state: { what: "Incident state normalizes all signals.", input: "Sensors, vision, operator state", output: "typed IncidentState", production: "Event-sourced incident aggregate." },
  rule: { what: "Rule engine applies deterministic thresholds.", input: "smokePpm, temperatureC", output: "RuleResult", production: "Versioned rules with replay tests." },
  ml: { what: "ML risk model predicts fire probability.", input: "feature vector", output: "MLResult", production: "Registry, monitoring, and drift alerts." },
  agent: { what: "Agentic control plane proposes governed actions.", input: "evidence, SOP, tools, policy", output: "AgenticResult", production: "Evaluated multi-agent runtime with traceability." },
  sop: { what: "SOP retrieval provides procedures.", input: "incident context", output: "SOP references", production: "Vector DB with access-controlled docs." },
  policy: { what: "Policy engine blocks or approval-gates actions.", input: "proposed action", output: "PolicyDecision", production: "OPA/Rego service with signed policies." },
  human: { what: "Operator approves high-risk actions.", input: "proposal, evidence, policy", output: "approval decision", production: "RBAC and command-center workflow." },
  tools: { what: "Tool executor performs real or sandbox actions.", input: "approved tool call", output: "execution result", production: "Governed APIs for drone, gate, notifications." },
  drone: { what: "Drone reconnaissance collects new evidence.", input: "dispatch command", output: "image/video", production: "Flight safety envelope and airspace policy." },
  gate: { what: "Smart gate controls site access.", input: "unlock command", output: "access state", production: "Physical access API with dual approval." },
  operator: { what: "Operator notification alerts response staff.", input: "incident report", output: "operator alert", production: "Pager/command-center integration." },
  authority: { what: "Authority notification escalates critical incidents.", input: "approved alert", output: "authority event", production: "Official governed API integration." },
  trace: { what: "Trace store records execution telemetry.", input: "events", output: "timeline", production: "OpenTelemetry, Tempo, Loki, SIEM." },
  record: { what: "Decision record preserves the audit trail.", input: "run artifacts", output: "DecisionRecord", production: "Immutable event store and retention policy." }
} as const;

export function ArchitectureMap() {
  const [selected, setSelected] = useState<keyof typeof nodeDetails>("agent");
  const nodes: Node[] = [
    ["smoke", "Smoke Sensor", 0, 40],
    ["heat", "Heat Sensor", 0, 140],
    ["camera", "Camera", 0, 240],
    ["edge", "Edge Vision Model", 230, 240],
    ["state", "Incident State", 230, 110],
    ["rule", "Rule Engine", 460, 20],
    ["ml", "ML Risk Model", 460, 110],
    ["agent", "Agentic Control Plane", 460, 210],
    ["sop", "SOP Retrieval", 700, 20],
    ["policy", "Policy Engine", 700, 120],
    ["human", "Human Approval", 700, 220],
    ["tools", "Tool Execution", 930, 120],
    ["drone", "Drone", 1160, 20],
    ["gate", "Gate", 1160, 100],
    ["operator", "Operator Notification", 1160, 180],
    ["authority", "Authority Notification", 1160, 260],
    ["trace", "Trace Store", 930, 300],
    ["record", "Decision Record", 1160, 360]
  ].map(([id, label, x, y]) => ({ id, position: { x, y }, data: { label } }) as Node);
  const edges: Edge[] = [
    ["smoke", "state"], ["heat", "state"], ["camera", "edge"], ["edge", "state"], ["state", "rule"], ["state", "ml"], ["rule", "agent"], ["ml", "agent"], ["agent", "sop"], ["agent", "policy"], ["policy", "human"], ["human", "tools"], ["tools", "drone"], ["tools", "gate"], ["tools", "operator"], ["tools", "authority"], ["agent", "trace"], ["tools", "trace"], ["trace", "record"]
  ].map(([source, target], index) => ({ id: `e-${index}`, source, target, animated: index < 4 }));

  return (
    <div className="grid gap-4 xl:grid-cols-[1.3fr_.7fr]">
      <Card>
        <CardHeader>
          <CardTitle>Physical AI Architecture</CardTitle>
          <CardDescription>Click any node to inspect inputs, outputs, implementation path, failure modes, and governance concern.</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="h-[640px] overflow-hidden rounded-lg border border-white/10 bg-slate-950">
            <ReactFlow nodes={nodes} edges={edges} onNodeClick={(_, node) => setSelected(node.id as keyof typeof nodeDetails)} fitView>
              <Background />
              <MiniMap />
              <Controls />
            </ReactFlow>
          </div>
        </CardContent>
      </Card>
      <JsonInspector title={`Node: ${selected}`} value={nodeDetails[selected]} />
    </div>
  );
}
