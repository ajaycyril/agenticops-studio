"use client";

import { useMemo, useState } from "react";
import type * as tf from "@tensorflow/tfjs";
import { Background, Controls, MiniMap, ReactFlow, type Edge, type Node } from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import { motion } from "framer-motion";
import { Activity, AlertTriangle, BrainCircuit, Camera, Cpu, Download, FileJson, Gauge, GitBranch, ShieldCheck } from "lucide-react";
import { Alert } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Slider } from "@/components/ui/slider";
import { Switch } from "@/components/ui/switch";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { ConfidenceMeter } from "@/components/studio/ConfidenceMeter";
import { EdgeNodeCard } from "@/components/studio/EdgeNodeCard";
import { JsonInspector } from "@/components/studio/JsonInspector";
import { MetricCard } from "@/components/studio/MetricCard";
import { PolicyDecisionCard } from "@/components/studio/PolicyDecisionCard";
import { RiskBadge } from "@/components/studio/RiskBadge";
import { ToolCallCard } from "@/components/studio/ToolCallCard";
import { getToolRegistry } from "@/lib/agent/tool-registry";
import { buildDecisionRecord } from "@/lib/decision-record/build-decision-record";
import { summarizeDecisionRecord } from "@/lib/decision-record/report-summary";
import { DEFAULT_MODEL_VERSION, ENTERPRISE_LINE, PRODUCT_THESIS } from "@/lib/constants";
import { incidentToFeatures } from "@/lib/ml/dataset";
import { approximateFeatureImportance } from "@/lib/ml/feature-importance";
import { loadModelVersion, saveModelVersion } from "@/lib/ml/model-store";
import { heuristicRiskPrediction, riskLevelFromProbability } from "@/lib/ml/predict-risk";
import { trainRiskModel, type TrainingProgress } from "@/lib/ml/train-risk-model";
import { evaluatePoliciesForActions } from "@/lib/policies/policy-evaluator";
import { evaluateRules } from "@/lib/rule/rule-engine";
import { clonePreset, scenarioPresets } from "@/lib/scenario-presets";
import { createTraceEvent } from "@/lib/trace/create-trace-event";
import { saveTrace } from "@/lib/trace/trace-store";
import type { AgenticResult, DecisionRecord, IncidentState, MLResult, PolicyDecision, RuleResult, TraceEvent, VisionResult } from "@/lib/types";

const sampleNames = ["cooking-smoke", "fire-smoke-room", "unclear-camera"];

function pct(value: number) {
  return `${Math.round(value * 100)}%`;
}

export function StudioShell() {
  const [incident, setIncident] = useState<IncidentState>(() => clonePreset(1));
  const [trace, setTrace] = useState<TraceEvent[]>([
    createTraceEvent({
      type: "incident_created",
      actor: "system",
      status: "success",
      output: clonePreset(1),
      explanation: "Loaded default confirmed fire scenario."
    })
  ]);
  const [ruleResult, setRuleResult] = useState<RuleResult>(() => evaluateRules(clonePreset(1)));
  const [mlResult, setMlResult] = useState<MLResult>(() => heuristicRiskPrediction(clonePreset(1)));
  const [visionResult, setVisionResult] = useState<VisionResult | undefined>();
  const [agenticResult, setAgenticResult] = useState<AgenticResult | undefined>();
  const [policyDecisions, setPolicyDecisions] = useState<PolicyDecision[]>([]);
  const [decisionRecord, setDecisionRecord] = useState<DecisionRecord | undefined>();
  const [lossCurve, setLossCurve] = useState<TrainingProgress[]>([]);
  const [training, setTraining] = useState(false);
  const [model, setModel] = useState<tf.LayersModel | undefined>();
  const [modelMetrics, setModelMetrics] = useState<MLResult["metrics"]>();
  const [modelVersion, setModelVersion] = useState(() => loadModelVersion() ?? DEFAULT_MODEL_VERSION);
  const [sampleName, setSampleName] = useState("fire-smoke-room");
  const [uploadedImage, setUploadedImage] = useState<string | undefined>();
  const [message, setMessage] = useState<string | undefined>();

  const tools = useMemo(() => getToolRegistry(), []);

  function appendTrace(event: Omit<TraceEvent, "id" | "timestamp">) {
    setTrace((current) => {
      const next = [...current, createTraceEvent(event)];
      saveTrace(next);
      return next;
    });
  }

  function loadScenario(index: number) {
    const next = clonePreset(index);
    setIncident(next);
    setRuleResult(evaluateRules(next));
    setMlResult(heuristicRiskPrediction(next));
    setVisionResult(undefined);
    setAgenticResult(undefined);
    setPolicyDecisions([]);
    setDecisionRecord(undefined);
    appendTrace({ type: "incident_created", actor: "system", status: "success", output: next, explanation: `Loaded ${next.scenarioName}.` });
  }

  function updateIncident<K extends keyof IncidentState>(key: K, value: IncidentState[K]) {
    setIncident((current) => {
      const next = { ...current, [key]: value };
      setRuleResult(evaluateRules(next));
      setMlResult({ ...heuristicRiskPrediction(next), modelVersion, metrics: modelMetrics });
      return next;
    });
  }

  function runRules() {
    const result = evaluateRules(incident);
    setRuleResult(result);
    appendTrace({ type: "rule_engine_evaluated", actor: "system", input: incident, output: result, status: "success" });
  }

  async function trainModel() {
    setTraining(true);
    setLossCurve([]);
    appendTrace({ type: "ml_model_training_started", actor: "ml_model", status: "pending", input: { size: 260, epochs: 28 } });
    try {
      const trained = await trainRiskModel({
        size: 260,
        epochs: 28,
        falseAlarmBias: incident.historicalFalseAlarmRate,
        onProgress: (point) => setLossCurve((points) => [...points.slice(-36), point])
      });
      setModel(trained.model);
      setModelMetrics(trained.metrics);
      setModelVersion(trained.modelVersion);
      saveModelVersion(trained.modelVersion);
      appendTrace({ type: "ml_model_training_completed", actor: "ml_model", status: "success", output: trained.metrics });
    } finally {
      setTraining(false);
    }
  }

  async function runPrediction() {
    let probability: number;
    if (model) {
      const prediction = model.predict((await import("@tensorflow/tfjs")).tensor2d([incidentToFeatures(incident)])) as tf.Tensor;
      probability = (await prediction.data())[0];
      prediction.dispose();
    } else {
      probability = heuristicRiskPrediction(incident).fireProbability;
    }
    const result: MLResult = {
      fireProbability: Number(probability.toFixed(3)),
      riskLevel: riskLevelFromProbability(probability),
      modelVersion,
      metrics: modelMetrics,
      featureImportance: approximateFeatureImportance(incidentToFeatures(incident)),
      explanation: "The ML model predicts risk only. It does not coordinate response or execute physical actions."
    };
    setMlResult(result);
    appendTrace({ type: "ml_model_predicted", actor: "ml_model", input: incident, output: result, status: "success" });
  }

  async function runVision() {
    const start = performance.now();
    const response = await fetch("/api/vision/roboflow", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ incidentId: incident.incidentId, sampleName, imageBase64: uploadedImage })
    });
    const data = await response.json();
    if (!data.ok) {
      setMessage(data.error.message);
      appendTrace({ type: "error", actor: "vision_model", status: "failed", output: data.error });
      return;
    }
    const result = data.result as VisionResult;
    setVisionResult(result);
    setIncident((current) => ({
      ...current,
      cameraSmokeConfidence: result.maxSmokeConfidence,
      cameraFireConfidence: result.maxFireConfidence,
      visionProvider: result.provider,
      visionDetections: result.detections,
      imageUrl: uploadedImage ?? `/sample-images/${sampleName}.svg`
    }));
    appendTrace({
      type: "vision_model_called",
      actor: "vision_model",
      input: { provider: result.provider, sampleName },
      output: result,
      latencyMs: Math.round(performance.now() - start),
      status: "success",
      explanation: result.message
    });
  }

  async function runAgent() {
    const start = performance.now();
    appendTrace({ type: "agent_called", actor: "llm_agent", status: "pending", input: { incident, mlResult, ruleResult } });
    const response = await fetch("/api/agent/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        incident,
        ruleResult,
        mlResult,
        visionResult,
        toolRegistry: tools,
        policySummary: "TypeScript fire response policy evaluator v1"
      })
    });
    const data = await response.json();
    if (!data.ok) {
      setMessage(data.error.message);
      appendTrace({ type: "error", actor: "llm_agent", status: "failed", output: data.error });
      return;
    }
    const result = data.result as AgenticResult;
    const policies = evaluatePoliciesForActions(result.proposedActions.map((proposal) => proposal.action), incident, mlResult);
    setAgenticResult(result);
    setPolicyDecisions(policies);
    appendTrace({ type: "agent_output_validated", actor: "llm_agent", output: result, latencyMs: Math.round(performance.now() - start), status: "success" });
    policies.forEach((policy) =>
      appendTrace({
        type: policy.blocked ? "action_blocked" : policy.requiresHumanApproval ? "human_approval_requested" : "policy_checked",
        actor: "policy",
        output: policy,
        status: policy.blocked ? "blocked" : policy.requiresHumanApproval ? "pending" : "success"
      })
    );
  }

  function resolveApproval(action: "unlockGate" | "dispatchDrone" | "notifyAuthority", approved: boolean) {
    setIncident((current) => ({
      ...current,
      humanApproval: { ...current.humanApproval, [action]: approved }
    }));
    appendTrace({
      type: "human_approval_resolved",
      actor: "human",
      status: approved ? "success" : "blocked",
      output: { action, approved, approvedBy: "demo-operator" }
    });
  }

  function writeDecisionRecord() {
    if (!agenticResult || !visionResult) {
      setMessage("Run vision and agent workflow before writing the decision record.");
      return;
    }
    const record = buildDecisionRecord({
      runId: `RUN-${Date.now()}`,
      incident,
      ruleResult,
      mlResult,
      visionResult,
      agenticResult,
      policyDecisions,
      trace
    });
    setDecisionRecord(record);
    sessionStorage.setItem("agenticops.latestDecisionRecord", JSON.stringify(record));
    appendTrace({ type: "decision_record_written", actor: "system", status: "success", output: { runId: record.runId } });
  }

  function downloadRecord() {
    if (!decisionRecord) return;
    const blob = new Blob([JSON.stringify(decisionRecord, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = `${decisionRecord.incidentId}-${decisionRecord.runId}.json`;
    anchor.click();
    URL.revokeObjectURL(url);
  }

  const graphNodes: Node[] = [
    { id: "sensors", position: { x: 0, y: 80 }, data: { label: "Sensors trigger" } },
    { id: "vision", position: { x: 190, y: 0 }, data: { label: "Edge vision" } },
    { id: "ml", position: { x: 190, y: 150 }, data: { label: "ML risk model" } },
    { id: "agent", position: { x: 410, y: 75 }, data: { label: "Agentic control plane" } },
    { id: "policy", position: { x: 660, y: 0 }, data: { label: "Policy guardrails" } },
    { id: "human", position: { x: 660, y: 150 }, data: { label: "Human approval" } },
    { id: "tools", position: { x: 900, y: 75 }, data: { label: "Sandbox tools + record" } }
  ];
  const graphEdges: Edge[] = [
    { id: "e1", source: "sensors", target: "vision", animated: true },
    { id: "e2", source: "sensors", target: "ml", animated: true },
    { id: "e3", source: "vision", target: "agent" },
    { id: "e4", source: "ml", target: "agent" },
    { id: "e5", source: "agent", target: "policy" },
    { id: "e6", source: "policy", target: "human" },
    { id: "e7", source: "human", target: "tools" }
  ];

  return (
    <div className="space-y-6">
      <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }}>
        <Card className="glass-grid">
          <CardHeader>
            <Badge className="w-fit border-cyan-300/30 bg-cyan-300/10 text-cyan-100">Hassantuk-inspired incident workflow</Badge>
            <CardTitle className="text-2xl md:text-4xl">AgenticOps Studio Control Tower</CardTitle>
            <CardDescription className="max-w-4xl text-base">
              {PRODUCT_THESIS} {ENTERPRISE_LINE}
            </CardDescription>
          </CardHeader>
          <CardContent className="grid gap-3 sm:grid-cols-2 lg:grid-cols-5">
            <MetricCard label="Smoke" value={`${incident.smokePpm} ppm`} />
            <MetricCard label="Temperature" value={`${incident.temperatureC} C`} tone={incident.temperatureC > 55 ? "red" : "cyan"} />
            <MetricCard label="ML Fire Probability" value={pct(mlResult.fireProbability)} tone={mlResult.riskLevel === "critical" ? "red" : "amber"} />
            <MetricCard label="Vision Provider" value={incident.visionProvider} />
            <Card className="shadow-none">
              <CardContent className="p-4">
                <div className="text-xs uppercase tracking-wide text-slate-500">Risk</div>
                <div className="mt-3"><RiskBadge level={mlResult.riskLevel} /></div>
              </CardContent>
            </Card>
          </CardContent>
        </Card>
      </motion.div>

      {message ? <Alert>{message}</Alert> : null}

      <Tabs defaultValue="scenario">
        <TabsList>
          {["scenario", "comparison", "vision", "ml", "workflow", "tools", "approval", "trace", "record"].map((tab) => (
            <TabsTrigger key={tab} value={tab}>
              {tab}
            </TabsTrigger>
          ))}
        </TabsList>

        <TabsContent value="scenario">
          <div className="grid gap-4 lg:grid-cols-[.95fr_1.05fr]">
            <Card>
              <CardHeader>
                <CardTitle>Scenario Simulator</CardTitle>
                <CardDescription>Adjust physical-world signals and watch deterministic rules, ML risk, and agent governance diverge.</CardDescription>
              </CardHeader>
              <CardContent className="space-y-5">
                <div className="grid gap-2 sm:grid-cols-2">
                  {scenarioPresets.map((preset, index) => (
                    <Button key={preset.incidentId} variant="secondary" onClick={() => loadScenario(index)}>
                      {preset.scenarioName}
                    </Button>
                  ))}
                </div>
                {[
                  ["smokePpm", "Smoke ppm", 0, 140],
                  ["temperatureC", "Temperature C", 10, 80],
                  ["cameraSmokeConfidence", "Camera smoke", 0, 1],
                  ["cameraFireConfidence", "Camera fire", 0, 1],
                  ["sensorHealth", "Sensor health", 0, 1],
                  ["historicalFalseAlarmRate", "False alarm history", 0, 1],
                  ["windSpeedKmh", "Wind km/h", 0, 45]
                ].map(([key, label, min, max]) => (
                  <div key={String(key)} className="space-y-2">
                    <div className="flex justify-between text-sm text-slate-300">
                      <span>{label}</span>
                      <span>{typeof incident[key as keyof IncidentState] === "number" ? Number(incident[key as keyof IncidentState]).toFixed(max === 1 ? 2 : 0) : ""}</span>
                    </div>
                    <Slider
                      value={[Number(incident[key as keyof IncidentState])]}
                      min={Number(min)}
                      max={Number(max)}
                      step={Number(max) === 1 ? 0.01 : 1}
                      onValueChange={([value]) => updateIncident(key as keyof IncidentState, value as never)}
                    />
                  </div>
                ))}
                <div className="flex flex-wrap gap-4 text-sm text-slate-300">
                  <label className="flex items-center gap-2">Drone available <Switch checked={incident.droneAvailable} onCheckedChange={(value) => updateIncident("droneAvailable", value)} /></label>
                  <label className="flex items-center gap-2">Gate locked <Switch checked={incident.gateLocked} onCheckedChange={(value) => updateIncident("gateLocked", value)} /></label>
                </div>
              </CardContent>
            </Card>
            <JsonInspector title="Incident State" value={incident} />
          </div>
        </TabsContent>

        <TabsContent value="comparison">
          <div className="grid gap-4 lg:grid-cols-3">
            <Card>
              <CardHeader>
                <Cpu className="h-5 w-5 text-cyan-200" />
                <CardTitle>Rule-Based Automation</CardTitle>
                <CardDescription>Inputs: smokePpm, temperatureC. Logic: fixed if/else. Output: alarm/no alarm.</CardDescription>
              </CardHeader>
              <CardContent className="space-y-3">
                <pre className="rounded-md border border-white/10 bg-black/30 p-3 text-xs text-cyan-100">IF smoke_ppm &gt;= 70 THEN raise_alarm</pre>
                <Button onClick={runRules}>Run Rules</Button>
                <RiskBadge level={ruleResult.severity} />
                <JsonInspector title="Rule Result" value={ruleResult} />
              </CardContent>
            </Card>
            <Card>
              <CardHeader>
                <BrainCircuit className="h-5 w-5 text-amber-200" />
                <CardTitle>ML-Based Prediction</CardTitle>
                <CardDescription>Predicts fire probability from fused features, but does not coordinate response.</CardDescription>
              </CardHeader>
              <CardContent className="space-y-3">
                <Button onClick={runPrediction}>Run ML Prediction</Button>
                <ConfidenceMeter label="Fire probability" value={mlResult.fireProbability} />
                <RiskBadge level={mlResult.riskLevel} />
                <JsonInspector title="ML Result" value={mlResult} />
              </CardContent>
            </Card>
            <Card>
              <CardHeader>
                <ShieldCheck className="h-5 w-5 text-emerald-200" />
                <CardTitle>Governed Agentic AI</CardTitle>
                <CardDescription>Uses all context, SOP, tools, policy, approvals, traces, and decision records.</CardDescription>
              </CardHeader>
              <CardContent className="space-y-3">
                <Button onClick={runAgent}>Run Agentic Planner</Button>
                {agenticResult ? <JsonInspector title="Agentic Result" value={agenticResult} /> : <p className="text-sm text-slate-400">Run the planner to see proposed actions.</p>}
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        <TabsContent value="vision">
          <div className="grid gap-4 lg:grid-cols-[.9fr_1.1fr]">
            <Card>
              <CardHeader>
                <Camera className="h-5 w-5 text-cyan-200" />
                <CardTitle>Edge Vision Lab</CardTitle>
                <CardDescription>Run hosted Roboflow inference through a server route, or sample fallback when no key is configured.</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="grid gap-2 sm:grid-cols-3">
                  {sampleNames.map((name) => (
                    <Button key={name} variant={sampleName === name ? "default" : "secondary"} onClick={() => setSampleName(name)}>
                      {name}
                    </Button>
                  ))}
                </div>
                <input
                  type="file"
                  accept="image/*"
                  className="block w-full rounded-md border border-white/10 bg-black/20 p-3 text-sm text-slate-300"
                  onChange={(event) => {
                    const file = event.target.files?.[0];
                    if (!file) return;
                    const reader = new FileReader();
                    reader.onload = () => setUploadedImage(String(reader.result));
                    reader.readAsDataURL(file);
                  }}
                />
                <Button onClick={runVision}>Run Vision Model</Button>
                <div className="relative overflow-hidden rounded-lg border border-white/10 bg-black/30">
                  {/* eslint-disable-next-line @next/next/no-img-element -- supports user-uploaded data URLs and local demo SVGs. */}
                  <img src={uploadedImage ?? `/sample-images/${sampleName}.svg`} alt="Selected sample" className="aspect-video w-full object-cover" />
                  {(visionResult?.detections ?? []).map((detection, index) => (
                    <div
                      key={`${detection.className}-${index}`}
                      className="absolute border-2 border-cyan-300 bg-cyan-300/10 text-xs text-cyan-100"
                      style={{
                        left: `${detection.x ?? 20}%`,
                        top: `${detection.y ?? 20}%`,
                        width: `${detection.width ?? 25}%`,
                        height: `${detection.height ?? 18}%`
                      }}
                    >
                      {detection.className} {pct(detection.confidence)}
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
            <Card>
              <CardHeader>
                <CardTitle>Why Edge AI Matters</CardTitle>
                <CardDescription>
                  Edge AI runs perception and first-level intelligence near the device for lower latency, lower bandwidth, privacy, resilience,
                  local response, and cost control at fleet scale.
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <ConfidenceMeter label="Smoke confidence" value={visionResult?.maxSmokeConfidence ?? incident.cameraSmokeConfidence} />
                <ConfidenceMeter label="Fire confidence" value={visionResult?.maxFireConfidence ?? incident.cameraFireConfidence} />
                <div className="grid gap-3 md:grid-cols-3">
                  <EdgeNodeCard title="Physical layer">Smoke sensor, heat sensor, camera, occupancy signal, drone, smart gate.</EdgeNodeCard>
                  <EdgeNodeCard title="Edge intelligence">Local inference, event filtering, metadata extraction, confidence score, evidence pointer.</EdgeNodeCard>
                  <EdgeNodeCard title="Cloud/control">Incident state, orchestration, policy, human approval, decision record.</EdgeNodeCard>
                </div>
                {visionResult ? <JsonInspector title="Vision Result" value={visionResult} /> : null}
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        <TabsContent value="ml">
          <div className="grid gap-4 lg:grid-cols-[.85fr_1.15fr]">
            <Card>
              <CardHeader>
                <Gauge className="h-5 w-5 text-amber-200" />
                <CardTitle>ML Training Lab</CardTitle>
                <CardDescription>Train a real TensorFlow.js logistic model in the browser and promote the active demo model version.</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <Button onClick={trainModel} disabled={training}>{training ? "Training..." : "Train Browser Model"}</Button>
                <Button variant="secondary" onClick={runPrediction}>Predict Current Incident</Button>
                <div className="grid gap-3 sm:grid-cols-2">
                  <MetricCard label="Model version" value={modelVersion.split(".").slice(-1)[0]} />
                  <MetricCard label="Accuracy" value={modelMetrics ? pct(modelMetrics.accuracy) : "demo"} />
                  <MetricCard label="Recall" value={modelMetrics ? pct(modelMetrics.recall) : "demo"} />
                  <MetricCard label="FPR" value={modelMetrics ? pct(modelMetrics.falsePositiveRate) : "demo"} />
                </div>
              </CardContent>
            </Card>
            <Card>
              <CardHeader>
                <CardTitle>Loss Curve and Feature Importance</CardTitle>
                <CardDescription>ML predicts risk. It does not authorize or execute actions.</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="flex h-44 items-end gap-1 rounded-lg border border-white/10 bg-black/30 p-3">
                  {lossCurve.length ? (
                    lossCurve.map((point) => (
                      <div key={point.epoch} className="flex-1 rounded-t bg-cyan-300/70" style={{ height: `${Math.max(8, 120 * point.loss)}px` }} />
                    ))
                  ) : (
                    <p className="self-center text-sm text-slate-400">Train a model to populate live loss.</p>
                  )}
                </div>
                <div className="grid gap-2 sm:grid-cols-2">
                  {mlResult.featureImportance.map((item) => (
                    <ConfidenceMeter key={item.feature} label={item.feature} value={item.importance} />
                  ))}
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        <TabsContent value="workflow">
          <Card>
            <CardHeader>
              <GitBranch className="h-5 w-5 text-cyan-200" />
              <CardTitle>Agentic Workflow</CardTitle>
              <CardDescription>Logical agents are displayed as nodes even when implemented by one governed structured-output call.</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="h-[420px] overflow-hidden rounded-lg border border-white/10 bg-slate-950">
                <ReactFlow nodes={graphNodes} edges={graphEdges} fitView>
                  <Background />
                  <MiniMap />
                  <Controls />
                </ReactFlow>
              </div>
              <div className="mt-4 grid gap-3 md:grid-cols-3">
                {["Triage Agent", "Vision Context Agent", "Risk Agent", "SOP Agent", "Policy Agent", "Response Planner Agent"].map((node) => (
                  <EdgeNodeCard key={node} title={node}>Reads evidence, emits structured reasoning, and never executes physical actions directly.</EdgeNodeCard>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="tools">
          <div className="grid gap-4 lg:grid-cols-[1fr_.9fr]">
            <div className="grid gap-3 md:grid-cols-2">
              {tools.map((tool) => <ToolCallCard key={tool.name} tool={tool} />)}
            </div>
            <div className="space-y-3">
              <Card>
                <CardHeader>
                  <AlertTriangle className="h-5 w-5 text-amber-200" />
                  <CardTitle>Guardrails</CardTitle>
                  <CardDescription>Schema, policy, evidence sufficiency, approval, permission, and physical safety guardrails are visible.</CardDescription>
                </CardHeader>
              </Card>
              {policyDecisions.map((decision) => <PolicyDecisionCard key={decision.action} decision={decision} />)}
            </div>
          </div>
        </TabsContent>

        <TabsContent value="approval">
          <div className="grid gap-4 md:grid-cols-3">
            {(["unlockGate", "dispatchDrone", "notifyAuthority"] as const).map((key) => {
              const action = key === "unlockGate" ? "unlock_gate" : key === "dispatchDrone" ? "dispatch_drone" : "notify_authority";
              const policy = policyDecisions.find((decision) => decision.action === action);
              return (
                <Card key={key}>
                  <CardHeader>
                    <CardTitle>{action}</CardTitle>
                    <CardDescription>{policy?.reason ?? "Run the agent to calculate policy status."}</CardDescription>
                  </CardHeader>
                  <CardContent className="space-y-3">
                    <Badge>{incident.humanApproval[key] ? "approved" : policy?.requiresHumanApproval ? "pending approval" : "not requested"}</Badge>
                    <div className="flex gap-2">
                      <Button onClick={() => resolveApproval(key, true)}>Approve</Button>
                      <Button variant="danger" onClick={() => resolveApproval(key, false)}>Reject</Button>
                    </div>
                  </CardContent>
                </Card>
              );
            })}
          </div>
        </TabsContent>

        <TabsContent value="trace">
          <div className="grid gap-4 lg:grid-cols-[.85fr_1.15fr]">
            <Card>
              <CardHeader>
                <Activity className="h-5 w-5 text-cyan-200" />
                <CardTitle>Trace Timeline</CardTitle>
                <CardDescription>Every meaningful step emits a structured event for observability and audit reconstruction.</CardDescription>
              </CardHeader>
              <CardContent className="space-y-2">
                {trace.map((event) => (
                  <div key={event.id} className="rounded-md border border-white/10 bg-black/20 p-3 text-xs">
                    <div className="flex flex-wrap items-center justify-between gap-2">
                      <span className="font-medium text-slate-100">{event.type}</span>
                      <Badge>{event.actor}</Badge>
                    </div>
                    <p className="mt-1 text-slate-500">{event.timestamp} · {event.status} {event.latencyMs ? `· ${event.latencyMs}ms` : ""}</p>
                  </div>
                ))}
              </CardContent>
            </Card>
            <JsonInspector title="Trace JSON" value={trace} />
          </div>
        </TabsContent>

        <TabsContent value="record">
          <div className="grid gap-4 lg:grid-cols-[.8fr_1.2fr]">
            <Card>
              <CardHeader>
                <FileJson className="h-5 w-5 text-cyan-200" />
                <CardTitle>Decision Record</CardTitle>
                <CardDescription>Decision records make enterprise AI auditable: inputs, models, policy checks, approvals, executions, and traces.</CardDescription>
              </CardHeader>
              <CardContent className="space-y-3">
                <Button onClick={writeDecisionRecord}>Write Decision Record</Button>
                <Button variant="secondary" onClick={downloadRecord} disabled={!decisionRecord}><Download className="h-4 w-4" /> Download JSON</Button>
                {decisionRecord ? (
                  <div className="space-y-3 text-sm text-slate-300">
                    <p>{summarizeDecisionRecord(decisionRecord, "executive")}</p>
                    <p>{summarizeDecisionRecord(decisionRecord, "technical")}</p>
                  </div>
                ) : null}
              </CardContent>
            </Card>
            <JsonInspector title="Decision Record JSON" value={decisionRecord ?? { message: "Run vision, agent, and write a record." }} />
          </div>
        </TabsContent>
      </Tabs>
    </div>
  );
}
