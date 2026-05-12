"use client";

import { useEffect, useMemo, useState } from "react";
import type * as tf from "@tensorflow/tfjs";
import { Background, Controls, MiniMap, ReactFlow, type Edge, type Node } from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import { motion } from "framer-motion";
import {
  Activity,
  AlertTriangle,
  BrainCircuit,
  Camera,
  Cpu,
  Download,
  FileJson,
  Gauge,
  GitBranch,
  PlayCircle,
  RadioTower,
  Route,
  Server,
  ShieldCheck,
  Timer,
  Workflow
} from "lucide-react";
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
import { heuristicRiskPrediction } from "@/lib/ml/predict-risk";
import { trainRiskModel, type TrainingProgress } from "@/lib/ml/train-risk-model";
import { evaluatePoliciesForActions } from "@/lib/policies/policy-evaluator";
import { evaluateRules } from "@/lib/rule/rule-engine";
import { clonePreset, scenarioPresets } from "@/lib/scenario-presets";
import { createTraceEvent } from "@/lib/trace/create-trace-event";
import { saveTrace } from "@/lib/trace/trace-store";
import type { AgenticResult, DecisionRecord, IncidentState, MLResult, PolicyDecision, RuleResult, TraceEvent, VisionResult } from "@/lib/types";

const sampleImageMap = {
  "cooking-smoke": { file: "cooking-smoke.jpg", label: "Cooking smoke false alarm" },
  "fire-smoke-room": { file: "fire-smoke-room.jpg", label: "Confirmed smoke + flame" },
  "unclear-camera": { file: "unclear-camera.jpg", label: "Low-confidence/unclear frame" }
} as const;
const sampleNames = Object.keys(sampleImageMap) as Array<keyof typeof sampleImageMap>;
const studioTabs = ["journey", "scenario", "physical", "comparison", "vision", "ml", "workflow", "tools", "approval", "trace", "record"] as const;
type StudioTab = (typeof studioTabs)[number];
const tabLabels: Record<StudioTab, string> = {
  journey: "0 Runbook",
  scenario: "1 Scenario",
  physical: "2 Physical AI",
  comparison: "3 Compare",
  vision: "4 Vision",
  ml: "5 ML",
  workflow: "6 Agentic",
  tools: "7 Tools",
  approval: "8 Approval",
  trace: "9 Trace",
  record: "10 Record"
};

type HealthStatus = {
  status: "ok";
  app: string;
  openaiConfigured: boolean;
  roboflowConfigured: boolean;
  openaiModel: string;
  openaiMaxOutputTokens: number;
  openaiTimeoutMs: number;
  openaiMaxAgentCallsPerRun: number;
  timestamp: string;
};

type VisionRunOutcome = {
  ok: boolean;
  result?: VisionResult;
  incident?: IncidentState;
};

type AgentRunOutcome = {
  ok: boolean;
  result?: AgenticResult;
  policies?: PolicyDecision[];
  provider?: "openai" | "sample";
  message?: string;
};

function pct(value: number) {
  return `${Math.round(value * 100)}%`;
}

function tunedRiskLevel(probability: number, threshold: number) {
  if (probability >= Math.min(0.95, threshold + 0.22)) return "critical";
  if (probability >= threshold) return "high";
  if (probability >= Math.max(0.2, threshold * 0.6)) return "medium";
  return "low";
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
  const [sampleName, setSampleName] = useState<keyof typeof sampleImageMap>("fire-smoke-room");
  const [uploadedImage, setUploadedImage] = useState<string | undefined>();
  const [message, setMessage] = useState<string | undefined>();
  const [health, setHealth] = useState<HealthStatus | undefined>();
  const [agentProvider, setAgentProvider] = useState<"openai" | "sample" | "not-run">("not-run");
  const [visionProvider, setVisionProvider] = useState<"roboflow" | "sample" | "not-run">("not-run");
  const [trainingSize, setTrainingSize] = useState(260);
  const [trainingEpochs, setTrainingEpochs] = useState(28);
  const [learningRate, setLearningRate] = useState(0.08);
  const [falseAlarmBias, setFalseAlarmBias] = useState(incident.historicalFalseAlarmRate);
  const [decisionThreshold, setDecisionThreshold] = useState(0.62);
  const [agentControls, setAgentControls] = useState({
    operatingMode: "balanced" as "balanced" | "conservative" | "rapid_response",
    authorityPosture: "critical_only" as "strict" | "approval_gated" | "critical_only",
    operatorInstruction: "Prioritize life safety, evidence sufficiency, and no unsupervised physical action."
  });
  const [guidedRunStatus, setGuidedRunStatus] = useState<
    Array<{ step: string; status: "pending" | "success" | "failed"; detail?: string }>
  >([]);
  const [activeTab, setActiveTab] = useState<StudioTab>("journey");
  const [visionRunning, setVisionRunning] = useState(false);
  const [agentRunning, setAgentRunning] = useState(false);

  const tools = useMemo(() => getToolRegistry(), []);

  useEffect(() => {
    fetch("/api/health")
      .then((response) => response.json())
      .then((data: HealthStatus) => setHealth(data))
      .catch(() => setHealth(undefined));
  }, []);

  const physicalLayers = [
    {
      title: "1. Physical world signals",
      demo: "Smoke ppm, heat, camera frame, occupancy, wind, drone availability, and gate state are editable incident inputs.",
      production: "MQTT/IoT ingestion with device identity, timestamps, calibration, and device-health telemetry.",
      why: "Physical AI starts when software reasons over real-world state, not only text."
    },
    {
      title: "2. Edge perception",
      demo: "The vision route uses Roboflow when configured or sample inference when not configured.",
      production: "YOLO/ONNX on edge gateways or camera-side NPUs with offline buffering and evidence pointers.",
      why: "Edge inference reduces latency, bandwidth, privacy exposure, and cloud dependency."
    },
    {
      title: "3. Risk intelligence",
      demo: "TensorFlow.js trains a browser-side model and predicts fire probability from fused features.",
      production: "Model registry, drift monitoring, replay evaluation, calibration, and promotion gates.",
      why: "ML gives probability. It should not directly execute physical actions."
    },
    {
      title: "4. Agentic control plane",
      demo: "OpenAI Responses API is called server-side when `OPENAI_API_KEY` is configured; otherwise a deterministic governed planner runs.",
      production: "Evaluated agent runtime with tool contracts, SOP retrieval, prompt/version governance, and structured output validation.",
      why: "Agentic AI coordinates models, context, tools, policy, and humans."
    },
    {
      title: "5. Governed execution",
      demo: "Policy evaluator blocks or approval-gates drone, gate, and authority actions; all physical tools stay sandboxed.",
      production: "OPA/Rego, RBAC, dual approval, signed tool requests, immutable audit logs, and real command-center queues.",
      why: "Enterprise agentic AI governs execution before anything touches the physical world."
    }
  ];

  const agentNodes = [
    {
      title: "Triage Agent",
      output: agenticResult?.incidentSummary ?? "Run the agent to summarize severity and gaps.",
      evidence: agenticResult?.riskAssessment.evidence ?? []
    },
    {
      title: "Vision Context Agent",
      output: visionResult
        ? `${visionResult.provider} vision returned smoke ${pct(visionResult.maxSmokeConfidence)} and fire ${pct(visionResult.maxFireConfidence)}.`
        : "Run Edge Vision Lab to attach camera evidence.",
      evidence: visionResult?.detections.map((detection) => `${detection.className} ${pct(detection.confidence)}`) ?? []
    },
    {
      title: "Risk Agent",
      output: `ML model ${mlResult.modelVersion} predicts ${pct(mlResult.fireProbability)} fire probability as ${mlResult.riskLevel}.`,
      evidence: mlResult.featureImportance.slice(0, 3).map((item) => `${item.feature}: ${pct(item.importance)}`)
    },
    {
      title: "SOP Agent",
      output: agenticResult ? `Retrieved ${agenticResult.sopReferences.join(", ")}.` : "Run agent to retrieve local SOP references.",
      evidence: agenticResult?.sopReferences ?? []
    },
    {
      title: "Policy Agent",
      output: policyDecisions.length
        ? `${policyDecisions.filter((decision) => decision.blocked).length} blocked, ${policyDecisions.filter((decision) => decision.requiresHumanApproval).length} approval-gated.`
        : "Run the agent to evaluate policy over proposed actions.",
      evidence: policyDecisions.map((decision) => `${decision.action}: ${decision.blocked ? "blocked" : decision.requiresHumanApproval ? "approval" : "allowed"}`)
    },
    {
      title: "Response Planner Agent",
      output: agenticResult ? `${agenticResult.proposedActions.length} action proposals created. The model does not execute them.` : "Run agent to generate a governed action plan.",
      evidence: agenticResult?.proposedActions.map((proposal) => proposal.action) ?? []
    }
  ];

  const demoJourneys = [
    {
      id: "executive",
      title: "Executive Journey (3 min)",
      icon: Timer,
      steps: [
        "Open Rule vs ML vs Agentic comparison",
        "Run guided incident",
        "Show policy gates and human approvals",
        "Show decision record and trace"
      ],
      run: async () => {
        loadScenario(1);
        setActiveTab("comparison");
        await runGuidedIncident();
        setActiveTab("record");
      }
    },
    {
      id: "technical",
      title: "Technical Journey (8 min)",
      icon: Workflow,
      steps: [
        "Start with Physical + Edge AI layers",
        "Run live/sampled vision route",
        "Tune and train browser ML",
        "Run agent with runtime controls",
        "Inspect trace and policy outcomes"
      ],
      run: async () => {
        loadScenario(2);
        setActiveTab("physical");
        setMessage("Technical journey primed. Move through Physical -> Vision -> ML -> Workflow -> Trace.");
      }
    },
    {
      id: "live-ops",
      title: "Live Ops Journey",
      icon: PlayCircle,
      steps: [
        "Select incident preset",
        "Run vision, ML, and agent orchestration",
        "Resolve required approvals",
        "Write decision record",
        "Generate incident report"
      ],
      run: async () => {
        loadScenario(4);
        setActiveTab("scenario");
        await runGuidedIncident();
      }
    }
  ];

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
    return result;
  }

  async function trainModel() {
    setTraining(true);
    setLossCurve([]);
    appendTrace({ type: "ml_model_training_started", actor: "ml_model", status: "pending", input: { size: 260, epochs: 28 } });
    try {
      const trained = await trainRiskModel({
        size: trainingSize,
        epochs: trainingEpochs,
        falseAlarmBias,
        learningRate,
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

  async function runPrediction(incidentOverride: IncidentState = incident) {
    let probability: number;
    const features = incidentToFeatures(incidentOverride);
    if (model) {
      const prediction = model.predict((await import("@tensorflow/tfjs")).tensor2d([features])) as tf.Tensor;
      probability = (await prediction.data())[0];
      prediction.dispose();
    } else {
      probability = heuristicRiskPrediction(incidentOverride).fireProbability;
    }
    const result: MLResult = {
      fireProbability: Number(probability.toFixed(3)),
      riskLevel: tunedRiskLevel(probability, decisionThreshold),
      modelVersion,
      metrics: modelMetrics,
      featureImportance: approximateFeatureImportance(features),
      explanation: "The ML model predicts risk only. It does not coordinate response or execute physical actions."
    };
    setMlResult(result);
    appendTrace({ type: "ml_model_predicted", actor: "ml_model", input: incidentOverride, output: result, status: "success" });
    return result;
  }

  async function loadSampleImageAsDataUrl(name: keyof typeof sampleImageMap) {
    const response = await fetch(`/sample-images/${sampleImageMap[name].file}`);
    if (!response.ok) throw new Error(`Failed to load sample image ${name}`);
    const blob = await response.blob();
    return await new Promise<string>((resolve, reject) => {
      const reader = new FileReader();
      reader.onloadend = () => {
        if (typeof reader.result === "string") resolve(reader.result);
        else reject(new Error("Invalid sample image payload"));
      };
      reader.onerror = () => reject(new Error("Failed to read sample image"));
      reader.readAsDataURL(blob);
    });
  }

  async function runVision(): Promise<VisionRunOutcome> {
    const start = performance.now();
    setVisionRunning(true);
    setMessage("Running vision inference...");
    try {
      const imagePayload = uploadedImage ?? (await loadSampleImageAsDataUrl(sampleName));
      const response = await fetch("/api/vision/roboflow", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ incidentId: incident.incidentId, sampleName, imageBase64: imagePayload })
      });
      const data = await response.json();
      if (!data.ok) {
        setMessage(`Vision step failed: ${data.error.message}`);
        appendTrace({ type: "error", actor: "vision_model", status: "failed", output: data.error });
        return { ok: false };
      }
      const result = data.result as VisionResult;
      const nextIncident: IncidentState = {
        ...incident,
        cameraSmokeConfidence: result.maxSmokeConfidence,
        cameraFireConfidence: result.maxFireConfidence,
        visionProvider: result.provider,
        visionDetections: result.detections,
        imageUrl: uploadedImage ?? `/sample-images/${sampleImageMap[sampleName].file}`
      };
      setVisionProvider(result.provider);
      setVisionResult(result);
      setIncident(nextIncident);
      setMessage(
        result.provider === "roboflow"
          ? `Live Roboflow inference completed in ${result.latencyMs} ms.`
          : result.message ?? "Sample vision fallback used."
      );
      appendTrace({
        type: "vision_model_called",
        actor: "vision_model",
        input: { provider: result.provider, sampleName },
        output: result,
        latencyMs: Math.round(performance.now() - start),
        status: "success",
        explanation: result.message
      });
      return { ok: true, result, incident: nextIncident };
    } catch (error) {
      const text = error instanceof Error ? error.message : "Vision step failed unexpectedly.";
      setMessage(`Vision step failed: ${text}`);
      appendTrace({ type: "error", actor: "vision_model", status: "failed", output: { message: text } });
      return { ok: false };
    } finally {
      setVisionRunning(false);
    }
  }

  async function runAgent(overrides: { incident?: IncidentState; mlResult?: MLResult; ruleResult?: RuleResult; visionResult?: VisionResult } = {}): Promise<AgentRunOutcome> {
    const start = performance.now();
    const incidentForRun = overrides.incident ?? incident;
    const ruleForRun = overrides.ruleResult ?? ruleResult;
    const mlForRun = overrides.mlResult ?? mlResult;
    const visionForRun = overrides.visionResult ?? visionResult;
    setAgentRunning(true);
    setMessage("Running agentic planner...");
    appendTrace({ type: "agent_called", actor: "llm_agent", status: "pending", input: { incident: incidentForRun, mlResult: mlForRun, ruleResult: ruleForRun } });
    try {
      const response = await fetch("/api/agent/run", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          incident: incidentForRun,
          ruleResult: ruleForRun,
          mlResult: mlForRun,
          visionResult: visionForRun,
          toolRegistry: tools,
          policySummary: "TypeScript fire response policy evaluator v1",
          agentControls
        })
      });
      const data = await response.json();
      if (!data.ok) {
        setMessage(`Agent step failed: ${data.error.message}`);
        appendTrace({ type: "error", actor: "llm_agent", status: "failed", output: data.error });
        return { ok: false, message: data.error.message };
      }
      const result = data.result as AgenticResult;
      setAgentProvider(data.provider === "openai" ? "openai" : "sample");
      const policies = evaluatePoliciesForActions(result.proposedActions.map((proposal) => proposal.action), incidentForRun, mlForRun);
      setAgenticResult(result);
      setPolicyDecisions(policies);
      if (data.provider === "sample" && data.message) {
        setMessage(data.message);
      } else {
        setMessage("Agentic planner completed.");
      }
      appendTrace({
        type: "agent_output_validated",
        actor: "llm_agent",
        output: result,
        latencyMs: Math.round(performance.now() - start),
        status: "success",
        explanation: data.message
      });
      policies.forEach((policy) =>
        appendTrace({
          type: policy.blocked ? "action_blocked" : policy.requiresHumanApproval ? "human_approval_requested" : "policy_checked",
          actor: "policy",
          output: policy,
          status: policy.blocked ? "blocked" : policy.requiresHumanApproval ? "pending" : "success"
        })
      );
      return { ok: true, result, policies, provider: data.provider, message: data.message };
    } catch (error) {
      const text = error instanceof Error ? error.message : "Agent step failed unexpectedly.";
      setMessage(`Agent step failed: ${text}`);
      appendTrace({ type: "error", actor: "llm_agent", status: "failed", output: { message: text } });
      return { ok: false, message: text };
    } finally {
      setAgentRunning(false);
    }
  }

  async function runGuidedIncident() {
    const ruleForRun = evaluateRules(incident);
    setMessage("Running guided incident end-to-end.");
    setActiveTab("workflow");
    setGuidedRunStatus([
      { step: "Rule engine", status: "pending" },
      { step: "Vision inference", status: "pending" },
      { step: "ML prediction", status: "pending" },
      { step: "Agent planner", status: "pending" },
      { step: "Policy evaluation", status: "pending" },
      { step: "Decision record", status: "pending" }
    ]);
    setRuleResult(ruleForRun);
    appendTrace({ type: "rule_engine_evaluated", actor: "system", input: incident, output: ruleForRun, status: "success" });
    setGuidedRunStatus((items) => items.map((item) => (item.step === "Rule engine" ? { ...item, status: "success" } : item)));
    const visionRun = await runVision();
    if (!visionRun.ok || !visionRun.result || !visionRun.incident) {
      setGuidedRunStatus((items) =>
        items.map((item) =>
          item.step === "Vision inference" ? { ...item, status: "failed", detail: "Check Roboflow key/model or input image format." } : item
        )
      );
      return;
    }
    setGuidedRunStatus((items) => items.map((item) => (item.step === "Vision inference" ? { ...item, status: "success" } : item)));
    const mlForRun = await runPrediction(visionRun.incident);
    setGuidedRunStatus((items) => items.map((item) => (item.step === "ML prediction" ? { ...item, status: "success" } : item)));
    const agentRun = await runAgent({
      incident: visionRun.incident,
      mlResult: mlForRun,
      ruleResult: ruleForRun,
      visionResult: visionRun.result
    });
    if (!agentRun.ok || !agentRun.result || !agentRun.policies) {
      setGuidedRunStatus((items) =>
        items.map((item) =>
          item.step === "Agent planner"
            ? { ...item, status: "failed", detail: "Check OpenAI key, model access, token cap, or increase OPENAI_MAX_AGENT_CALLS_PER_RUN." }
            : item
        )
      );
      return;
    }
    setGuidedRunStatus((items) =>
      items.map((item) =>
        item.step === "Agent planner" || item.step === "Policy evaluation" ? { ...item, status: "success" } : item
      )
    );
    writeDecisionRecord({
      incident: visionRun.incident,
      ruleResult: ruleForRun,
      mlResult: mlForRun,
      visionResult: visionRun.result,
      agenticResult: agentRun.result,
      policyDecisions: agentRun.policies
    });
    setGuidedRunStatus((items) => items.map((item) => (item.step === "Decision record" ? { ...item, status: "success" } : item)));
    setMessage("Guided incident completed: vision, ML, agent plan, policy, and decision record are populated.");
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

  function writeDecisionRecord(
    overrides: {
      incident?: IncidentState;
      ruleResult?: RuleResult;
      mlResult?: MLResult;
      visionResult?: VisionResult;
      agenticResult?: AgenticResult;
      policyDecisions?: PolicyDecision[];
    } = {}
  ) {
    const incidentForRecord = overrides.incident ?? incident;
    const ruleForRecord = overrides.ruleResult ?? ruleResult;
    const mlForRecord = overrides.mlResult ?? mlResult;
    const visionForRecord = overrides.visionResult ?? visionResult;
    const agentForRecord = overrides.agenticResult ?? agenticResult;
    const policiesForRecord = overrides.policyDecisions ?? policyDecisions;
    if (!agentForRecord || !visionForRecord) {
      setMessage("Run vision and agent workflow before writing the decision record.");
      return;
    }
    const record = buildDecisionRecord({
      runId: `RUN-${Date.now()}`,
      incident: incidentForRecord,
      ruleResult: ruleForRecord,
      mlResult: mlForRecord,
      visionResult: visionForRecord,
      agenticResult: agentForRecord,
      policyDecisions: policiesForRecord,
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

  const graphNodeStyle = {
    background: "rgba(2, 6, 23, 0.96)",
    border: "1px solid rgba(34, 211, 238, 0.38)",
    borderRadius: 8,
    color: "#e0f2fe",
    minWidth: 150,
    boxShadow: "0 18px 36px rgba(0,0,0,0.28)"
  };
  const graphNodes: Node[] = [
    {
      id: "sensors",
      position: { x: 0, y: 80 },
      style: graphNodeStyle,
      data: { label: `Physical sensors\n${incident.smokePpm} ppm / ${incident.temperatureC} C` }
    },
    {
      id: "vision",
      position: { x: 210, y: 0 },
      style: graphNodeStyle,
      data: { label: `Edge vision\n${visionProvider === "not-run" ? "waiting" : visionProvider}` }
    },
    {
      id: "ml",
      position: { x: 210, y: 170 },
      style: graphNodeStyle,
      data: { label: `ML risk model\n${pct(mlResult.fireProbability)} ${mlResult.riskLevel}` }
    },
    {
      id: "agent",
      position: { x: 440, y: 85 },
      style: { ...graphNodeStyle, border: "1px solid rgba(16, 185, 129, 0.55)" },
      data: { label: `Agentic planner\n${agentProvider === "not-run" ? "click run" : agentProvider}` }
    },
    {
      id: "policy",
      position: { x: 700, y: 0 },
      style: graphNodeStyle,
      data: { label: `Policy guardrails\n${policyDecisions.length || "waiting"} checks` }
    },
    {
      id: "human",
      position: { x: 700, y: 170 },
      style: graphNodeStyle,
      data: { label: `Human approval\n${policyDecisions.filter((decision) => decision.requiresHumanApproval).length} gated` }
    },
    {
      id: "tools",
      position: { x: 950, y: 85 },
      style: graphNodeStyle,
      data: { label: `Sandbox tools\n${agenticResult?.proposedActions.length ?? 0} proposed` }
    }
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
          <CardContent className="grid gap-3 border-t border-white/10 pt-5 md:grid-cols-4">
            <div className="rounded-md border border-white/10 bg-black/20 p-3 text-sm">
              <div className="flex items-center gap-2 text-slate-100"><Server className="h-4 w-4 text-cyan-200" /> OpenAI API</div>
              <p className="mt-1 text-xs text-slate-400">
                {health?.openaiConfigured
                  ? `Live Responses API enabled. Model ${health.openaiModel}, max ${health.openaiMaxOutputTokens} output tokens, ${health.openaiMaxAgentCallsPerRun} call/run.`
                  : "Fallback planner active. Add OPENAI_API_KEY in Vercel to enable live LLM calls."}
              </p>
            </div>
            <div className="rounded-md border border-white/10 bg-black/20 p-3 text-sm">
              <div className="flex items-center gap-2 text-slate-100"><Camera className="h-4 w-4 text-cyan-200" /> Roboflow API</div>
              <p className="mt-1 text-xs text-slate-400">
                {health?.roboflowConfigured ? "Live hosted vision inference enabled." : "Sample vision active. Add ROBOFLOW_API_KEY in Vercel for live inference."}
              </p>
            </div>
            <div className="rounded-md border border-white/10 bg-black/20 p-3 text-sm">
              <div className="flex items-center gap-2 text-slate-100"><BrainCircuit className="h-4 w-4 text-amber-200" /> Browser ML</div>
              <p className="mt-1 text-xs text-slate-400">Real TensorFlow.js training runs in your browser with no server key required.</p>
            </div>
            <Button onClick={() => void runGuidedIncident()} className="h-full min-h-16">
              <Route className="h-4 w-4" /> Run Guided Incident
            </Button>
          </CardContent>
        </Card>
      </motion.div>

      {message ? <Alert>{message}</Alert> : null}
      {guidedRunStatus.length ? (
        <Card className="shadow-none">
          <CardHeader>
            <CardTitle className="text-sm">Live Execution Status</CardTitle>
            <CardDescription>End-to-end run telemetry for the current guided incident.</CardDescription>
          </CardHeader>
          <CardContent className="grid gap-2 sm:grid-cols-2 xl:grid-cols-3">
            {guidedRunStatus.map((item) => (
              <div key={item.step} className="rounded-md border border-white/10 bg-black/20 p-3 text-xs">
                <div className="flex items-center justify-between gap-3">
                  <span className="text-slate-200">{item.step}</span>
                  <Badge
                    className={
                      item.status === "success"
                        ? "border-emerald-300/30 bg-emerald-300/10 text-emerald-100"
                        : item.status === "failed"
                          ? "border-red-400/30 bg-red-500/10 text-red-100"
                          : "border-amber-300/30 bg-amber-300/10 text-amber-100"
                    }
                  >
                    {item.status}
                  </Badge>
                </div>
                {item.detail ? <p className="mt-2 text-slate-400">{item.detail}</p> : null}
              </div>
            ))}
          </CardContent>
        </Card>
      ) : null}

      <Tabs value={activeTab} onValueChange={(value) => setActiveTab(value as StudioTab)}>
        <TabsList>
          {studioTabs.map((tab) => (
            <TabsTrigger key={tab} value={tab}>
              {tabLabels[tab]}
            </TabsTrigger>
          ))}
        </TabsList>

        <TabsContent value="journey">
          <div className="grid gap-4 lg:grid-cols-3">
            {demoJourneys.map((journey) => (
              <Card key={journey.id}>
                <CardHeader>
                  <journey.icon className="h-5 w-5 text-cyan-200" />
                  <CardTitle>{journey.title}</CardTitle>
                  <CardDescription>Customer-facing runbook with explicit steps and expected outcomes.</CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="space-y-2">
                    {journey.steps.map((step, index) => (
                      <div key={step} className="rounded-md border border-white/10 bg-black/20 px-3 py-2 text-xs text-slate-300">
                        {index + 1}. {step}
                      </div>
                    ))}
                  </div>
                  <Button onClick={() => void journey.run()} className="w-full">
                    <Route className="h-4 w-4" /> Start Journey
                  </Button>
                </CardContent>
              </Card>
            ))}
          </div>
          <Card className="mt-4">
            <CardHeader>
              <CardTitle>Live Integration Signal</CardTitle>
              <CardDescription>These indicators confirm whether the demo is actually running live services or fallbacks.</CardDescription>
            </CardHeader>
            <CardContent className="grid gap-3 md:grid-cols-3">
              <div className="rounded-md border border-white/10 bg-black/20 p-3 text-xs text-slate-300">
                <p className="font-medium text-slate-100">OpenAI Agent Runtime</p>
                <p className="mt-1">{health?.openaiConfigured ? "Live server-side Responses API active." : "Fallback planner active."}</p>
              </div>
              <div className="rounded-md border border-white/10 bg-black/20 p-3 text-xs text-slate-300">
                <p className="font-medium text-slate-100">Roboflow Vision Runtime</p>
                <p className="mt-1">{health?.roboflowConfigured ? "Live hosted inference active." : "Sample vision fallback active."}</p>
              </div>
              <div className="rounded-md border border-white/10 bg-black/20 p-3 text-xs text-slate-300">
                <p className="font-medium text-slate-100">Browser ML Runtime</p>
                <p className="mt-1">TensorFlow.js training and prediction run in-browser with live tuning controls.</p>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

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

        <TabsContent value="physical">
          <div className="grid gap-4 xl:grid-cols-[.9fr_1.1fr]">
            <Card>
              <CardHeader>
                <RadioTower className="h-5 w-5 text-cyan-200" />
                <CardTitle>Physical AI and Edge AI Walkthrough</CardTitle>
                <CardDescription>
                  Physical AI is AI connected to real-world signals and actuators. Edge AI is where perception and first-level intelligence run near the device.
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-3">
                {physicalLayers.map((layer) => (
                  <div key={layer.title} className="rounded-lg border border-white/10 bg-black/20 p-4">
                    <h3 className="text-sm font-semibold text-slate-100">{layer.title}</h3>
                    <p className="mt-2 text-xs leading-5 text-cyan-100">Demo: {layer.demo}</p>
                    <p className="mt-1 text-xs leading-5 text-slate-400">Production: {layer.production}</p>
                    <p className="mt-1 text-xs leading-5 text-amber-100">Why it matters: {layer.why}</p>
                  </div>
                ))}
              </CardContent>
            </Card>
            <Card>
              <CardHeader>
                <CardTitle>Current Physical Incident Inputs</CardTitle>
                <CardDescription>These are the physical-world signals the workflow turns into governed AI decisions.</CardDescription>
              </CardHeader>
              <CardContent className="grid gap-3 sm:grid-cols-2">
                <MetricCard label="Smoke sensor" value={`${incident.smokePpm} ppm`} />
                <MetricCard label="Heat sensor" value={`${incident.temperatureC} C`} />
                <MetricCard label="Camera smoke" value={pct(incident.cameraSmokeConfidence)} />
                <MetricCard label="Camera fire" value={pct(incident.cameraFireConfidence)} />
                <MetricCard label="Occupancy" value={incident.occupancyStatus} />
                <MetricCard label="Drone" value={incident.droneAvailable ? "available" : "unavailable"} />
                <MetricCard label="Gate" value={incident.gateLocked ? "locked" : "unlocked"} />
                <MetricCard label="Sensor health" value={pct(incident.sensorHealth)} />
              </CardContent>
            </Card>
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
                <Button onClick={() => runRules()}>Run Rules</Button>
                <p className="text-xs leading-5 text-slate-400">
                  What actually runs: a deterministic TypeScript rule engine in <code>lib/rule/rule-engine.ts</code>. It intentionally ignores camera, SOP, tools, and approvals.
                </p>
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
                <Button onClick={() => void runPrediction()}>Run ML Prediction</Button>
                <p className="text-xs leading-5 text-slate-400">
                  What actually runs: TensorFlow.js if you trained a browser model; otherwise the deterministic baseline predictor. It outputs probability only.
                </p>
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
                <Button onClick={() => void runAgent()} disabled={agentRunning}>
                  {agentRunning ? "Running Agentic Planner..." : "Run Agentic Planner"}
                </Button>
                <p className="text-xs leading-5 text-slate-400">
                  What actually runs: OpenAI Responses API when configured; otherwise deterministic fallback. In both modes, schema validation, policy checks, approvals, trace events, and decision records still run.
                </p>
                <div className="flex flex-wrap gap-2">
                  <Badge>{agentProvider === "openai" ? "Live OpenAI" : agentProvider === "sample" ? "Fallback planner" : "Not run"}</Badge>
                  <Badge>{policyDecisions.length ? `${policyDecisions.length} policy checks` : "policy waiting"}</Badge>
                </div>
                {agenticResult ? (
                  <div className="space-y-3">
                    <div className="rounded-md border border-emerald-300/20 bg-emerald-500/10 p-3 text-xs leading-5 text-emerald-50">
                      <p className="font-semibold">Planner output</p>
                      <p className="mt-1">{agenticResult.incidentSummary}</p>
                      <p className="mt-1 text-emerald-100/80">{agenticResult.riskAssessment.reason}</p>
                    </div>
                    {agenticResult.proposedActions.map((proposal) => {
                      const policy = policyDecisions.find((decision) => decision.action === proposal.action);
                      return (
                        <div key={proposal.action} className="rounded-md border border-white/10 bg-black/20 p-3 text-xs">
                          <div className="flex items-start justify-between gap-3">
                            <div>
                              <p className="font-semibold text-slate-100">{proposal.action}</p>
                              <p className="mt-1 leading-5 text-slate-400">{proposal.reason}</p>
                            </div>
                            <Badge
                              className={
                                policy?.blocked
                                  ? "border-red-400/30 bg-red-500/10 text-red-100"
                                  : policy?.requiresHumanApproval
                                    ? "border-amber-300/30 bg-amber-300/10 text-amber-100"
                                    : "border-emerald-300/30 bg-emerald-300/10 text-emerald-100"
                              }
                            >
                              {policy?.blocked ? "blocked" : policy?.requiresHumanApproval ? "approval" : "allowed"}
                            </Badge>
                          </div>
                          <p className="mt-2 text-slate-500">Evidence: {proposal.evidenceUsed.join(", ")}</p>
                        </div>
                      );
                    })}
                    <JsonInspector title="Agentic Result JSON" value={agenticResult} />
                  </div>
                ) : (
                  <p className="text-sm text-slate-400">Run the planner to generate structured action proposals, policy checks, approval gates, and trace events.</p>
                )}
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
                    <button
                      key={name}
                      type="button"
                      onClick={() => setSampleName(name)}
                      className={`overflow-hidden rounded-lg border text-left transition ${
                        sampleName === name ? "border-cyan-300/70 bg-cyan-500/10" : "border-white/10 bg-black/20 hover:border-cyan-300/40"
                      }`}
                    >
                      {/* eslint-disable-next-line @next/next/no-img-element -- previewing static demo samples. */}
                      <img src={`/sample-images/${sampleImageMap[name].file}`} alt={sampleImageMap[name].label} className="h-20 w-full object-cover" />
                      <div className="px-3 py-2 text-xs text-slate-200">{sampleImageMap[name].label}</div>
                    </button>
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
                <Button onClick={() => void runVision()} disabled={visionRunning}>
                  {visionRunning ? "Running Vision Inference..." : "Run Vision Model"}
                </Button>
                <Badge>{visionProvider === "roboflow" ? "Live Roboflow" : visionProvider === "sample" ? "Sample fallback" : "Not run"}</Badge>
                <div className="relative overflow-hidden rounded-lg border border-white/10 bg-black/30">
                  {/* eslint-disable-next-line @next/next/no-img-element -- supports user-uploaded data URLs and local demo raster assets. */}
                  <img src={uploadedImage ?? `/sample-images/${sampleImageMap[sampleName].file}`} alt="Selected sample" className="aspect-video w-full object-cover" />
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
                <CardTitle>ML Training Lab: Tune a Real Browser Model</CardTitle>
                <CardDescription>
                  This is not static scoring. These controls change the synthetic dataset, optimizer, training duration, and decision threshold used by TensorFlow.js.
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                {[
                  ["trainingSize", "Training rows", trainingSize, 80, 800, 20],
                  ["trainingEpochs", "Epochs", trainingEpochs, 5, 80, 1],
                  ["learningRate", "Learning rate", learningRate, 0.01, 0.2, 0.01],
                  ["falseAlarmBias", "False alarm bias", falseAlarmBias, 0, 1, 0.01],
                  ["decisionThreshold", "High-risk threshold", decisionThreshold, 0.35, 0.9, 0.01]
                ].map(([key, label, value, min, max, step]) => (
                  <div key={String(key)} className="space-y-2">
                    <div className="flex justify-between text-sm text-slate-300">
                      <span>{label}</span>
                      <span>{Number(value).toFixed(Number(step) < 1 ? 2 : 0)}</span>
                    </div>
                    <Slider
                      value={[Number(value)]}
                      min={Number(min)}
                      max={Number(max)}
                      step={Number(step)}
                      onValueChange={([next]) => {
                        if (key === "trainingSize") setTrainingSize(Math.round(next));
                        if (key === "trainingEpochs") setTrainingEpochs(Math.round(next));
                        if (key === "learningRate") setLearningRate(next);
                        if (key === "falseAlarmBias") setFalseAlarmBias(next);
                        if (key === "decisionThreshold") setDecisionThreshold(next);
                      }}
                    />
                  </div>
                ))}
                <Button onClick={() => void trainModel()} disabled={training}>{training ? "Training..." : "Train Browser Model"}</Button>
                <Button variant="secondary" onClick={() => void runPrediction()}>Predict Current Incident</Button>
                <p className="text-xs leading-5 text-slate-400">
                  After training, prediction uses the in-memory TensorFlow.js model. Refreshing the page clears the model weights but keeps the promoted model version label in session storage.
                </p>
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
          <div className="grid gap-4 xl:grid-cols-[1fr_.9fr]">
            <Card>
              <CardHeader>
                <GitBranch className="h-5 w-5 text-cyan-200" />
                <CardTitle>Agentic Orchestration, Not Direct Control</CardTitle>
                <CardDescription>
                  The model proposes structured actions. TypeScript policy, human approval, sandbox tool permissions, trace events, and decision records govern execution.
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="h-[420px] overflow-hidden rounded-lg border border-white/10 bg-slate-950">
                  <ReactFlow nodes={graphNodes} edges={graphEdges} fitView>
                    <Background />
                    <MiniMap />
                    <Controls />
                  </ReactFlow>
                </div>
              </CardContent>
            </Card>
            <div className="grid gap-3">
              <Card className="shadow-none">
                <CardHeader className="p-4 pb-2">
                  <CardTitle className="text-sm">Agent Runtime Controls</CardTitle>
                  <CardDescription>
                    These controls change planner behavior per run. They are included in `/api/agent/run` payload and visible in traces.
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-3 p-4 pt-0">
                  <div className="space-y-2">
                    <p className="text-xs font-medium text-slate-300">Operating mode</p>
                    <div className="flex flex-wrap gap-2">
                      {(["balanced", "conservative", "rapid_response"] as const).map((mode) => (
                        <Button
                          key={mode}
                          size="sm"
                          variant={agentControls.operatingMode === mode ? "default" : "secondary"}
                          onClick={() => setAgentControls((current) => ({ ...current, operatingMode: mode }))}
                        >
                          {mode}
                        </Button>
                      ))}
                    </div>
                    <p className="text-xs leading-5 text-slate-500">
                      {agentControls.operatingMode === "conservative"
                        ? "Conservative mode asks for stronger visual evidence and more approvals."
                        : agentControls.operatingMode === "rapid_response"
                          ? "Rapid response mode proposes reconnaissance earlier when policy allows it."
                          : "Balanced mode weighs sensor, vision, ML, policy, and operator direction together."}
                    </p>
                  </div>
                  <div className="space-y-2">
                    <p className="text-xs font-medium text-slate-300">Authority posture</p>
                    <div className="flex flex-wrap gap-2">
                      {(["strict", "approval_gated", "critical_only"] as const).map((posture) => (
                        <Button
                          key={posture}
                          size="sm"
                          variant={agentControls.authorityPosture === posture ? "default" : "secondary"}
                          onClick={() => setAgentControls((current) => ({ ...current, authorityPosture: posture }))}
                        >
                          {posture}
                        </Button>
                      ))}
                    </div>
                    <p className="text-xs leading-5 text-slate-500">
                      {agentControls.authorityPosture === "strict"
                        ? "Strict posture suppresses authority notification proposals unless policy leaves no safer route."
                        : agentControls.authorityPosture === "approval_gated"
                          ? "Approval-gated posture routes authority notification through explicit operator approval."
                          : "Critical-only posture proposes authority notification only for critical risk."}
                    </p>
                  </div>
                  <div className="space-y-2">
                    <p className="text-xs font-medium text-slate-300">Operator directive</p>
                    <textarea
                      className="min-h-20 w-full rounded-md border border-white/10 bg-black/20 p-3 text-xs text-slate-200 outline-none focus:border-cyan-300/40"
                      value={agentControls.operatorInstruction}
                      onChange={(event) =>
                        setAgentControls((current) => ({
                          ...current,
                          operatorInstruction: event.target.value.slice(0, 500)
                        }))
                      }
                    />
                  </div>
                  <Button onClick={() => void runAgent()} disabled={agentRunning} className="w-full">
                    {agentRunning ? "Running Planner With Controls..." : "Run Planner With These Controls"}
                  </Button>
                </CardContent>
              </Card>
              {agentNodes.map((node) => (
                <Card key={node.title} className="shadow-none">
                  <CardHeader className="p-4 pb-2">
                    <CardTitle className="text-sm">{node.title}</CardTitle>
                    <CardDescription>{node.output}</CardDescription>
                  </CardHeader>
                  <CardContent className="flex flex-wrap gap-2 p-4 pt-0">
                    {node.evidence.length ? node.evidence.map((item) => <Badge key={item}>{item}</Badge>) : <Badge>waiting for run</Badge>}
                  </CardContent>
                </Card>
              ))}
            </div>
          </div>
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
                <Button onClick={() => writeDecisionRecord()}>Write Decision Record</Button>
                <Button variant="secondary" onClick={() => downloadRecord()} disabled={!decisionRecord}><Download className="h-4 w-4" /> Download JSON</Button>
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
