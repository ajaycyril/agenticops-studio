"use client";

import { useEffect, useMemo, useState } from "react";
import type * as tf from "@tensorflow/tfjs";
import { Background, Controls, Handle, MiniMap, Position, ReactFlow, type Edge, type Node, type NodeProps } from "@xyflow/react";
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
  Route,
  ShieldCheck,
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
import { DEFAULT_MODEL_VERSION } from "@/lib/constants";
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
const studioTabs = ["scenario", "vision", "ml", "comparison", "workflow", "approval", "trace", "record"] as const;
type StudioTab = (typeof studioTabs)[number];
const tabLabels: Record<StudioTab, string> = {
  scenario: "1 Live Incident",
  vision: "2 Vision",
  ml: "3 ML",
  comparison: "4 Compare",
  workflow: "5 Agentic",
  approval: "6 Approval",
  trace: "7 Trace",
  record: "8 Record"
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

type CapabilityStage = {
  id: string;
  title: string;
  tab: StudioTab;
  status: "ready" | "pending" | "complete" | "attention";
  signal: string;
  result: string;
  actionLabel: string;
  action: () => void;
};

type AgentRunStatus = {
  status: "idle" | "running" | "success" | "failed";
  provider: "openai" | "sample" | "not-run";
  runtime: "openai-agents-sdk" | "deterministic-fallback" | "not-run";
  message: string;
  latencyMs?: number;
  actions: number;
  policies: number;
};

type FlowNodeData = {
  title: string;
  detail: string;
  status: string;
  tone?: "cyan" | "emerald" | "amber" | "red";
};

type PlatformSignal = {
  label: string;
  value: string;
  detail: string;
  status: CapabilityStage["status"];
};

type LifecycleStage = {
  title: string;
  owner: string;
  status: CapabilityStage["status"];
  detail: string;
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

function statusClass(status: CapabilityStage["status"]) {
  if (status === "complete") return "border-emerald-300/30 bg-emerald-400/10 text-emerald-100";
  if (status === "attention") return "border-amber-300/30 bg-amber-300/10 text-amber-100";
  if (status === "pending") return "border-cyan-300/30 bg-cyan-300/10 text-cyan-100";
  return "border-white/10 bg-white/5 text-slate-300";
}

function AgenticFlowNode({ data }: NodeProps<Node<FlowNodeData>>) {
  const toneClass =
    data.tone === "emerald"
      ? "border-emerald-300/40 bg-emerald-400/10"
      : data.tone === "amber"
        ? "border-amber-300/40 bg-amber-400/10"
        : data.tone === "red"
          ? "border-red-400/40 bg-red-500/10"
          : "border-cyan-300/35 bg-cyan-400/10";
  return (
    <div className={`min-w-44 rounded-lg border px-3 py-2 text-left shadow-xl shadow-black/30 ${toneClass}`}>
      <Handle type="target" position={Position.Left} className="!border-slate-950 !bg-cyan-200" />
      <div className="text-[11px] uppercase text-slate-500">{data.status}</div>
      <div className="mt-1 text-sm font-semibold text-slate-100">{data.title}</div>
      <div className="mt-1 text-[11px] leading-4 text-slate-300">{data.detail}</div>
      <Handle type="source" position={Position.Right} className="!border-slate-950 !bg-cyan-200" />
    </div>
  );
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
  const [activeTab, setActiveTab] = useState<StudioTab>("scenario");
  const [visionRunning, setVisionRunning] = useState(false);
  const [agentRunning, setAgentRunning] = useState(false);
  const [agentRunStatus, setAgentRunStatus] = useState<AgentRunStatus>({
    status: "idle",
    provider: "not-run",
    runtime: "not-run",
    message: "Planner has not run yet. Click Run Agentic Planner or Run Guided Incident.",
    actions: 0,
    policies: 0
  });

  const tools = useMemo(() => getToolRegistry(), []);
  const nodeTypes = useMemo(() => ({ agentic: AgenticFlowNode }), []);

  useEffect(() => {
    fetch("/api/health")
      .then((response) => response.json())
      .then((data: HealthStatus) => setHealth(data))
      .catch(() => setHealth(undefined));
  }, []);

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
    setAgentProvider("not-run");
    setAgentRunStatus({
      status: "idle",
      provider: "not-run",
      runtime: "not-run",
      message: "Scenario changed. Run the planner again to generate a governed action plan.",
      actions: 0,
      policies: 0
    });
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
    setAgentRunStatus({
      status: "running",
      provider: "not-run",
      runtime: "openai-agents-sdk",
      message: "Sending incident, rule result, ML result, vision evidence, tool registry, policy summary, and runtime controls to /api/agent/run.",
      actions: 0,
      policies: 0
    });
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
        setAgentRunStatus({
          status: "failed",
          provider: "not-run",
          runtime: "not-run",
          message: data.error.message,
          latencyMs: Math.round(performance.now() - start),
          actions: 0,
          policies: 0
        });
        appendTrace({ type: "error", actor: "llm_agent", status: "failed", output: data.error });
        return { ok: false, message: data.error.message };
      }
      const result = data.result as AgenticResult;
      const provider = data.provider === "openai" ? "openai" : "sample";
      const runtime = data.runtime === "openai-agents-sdk" ? "openai-agents-sdk" : "deterministic-fallback";
      setAgentProvider(provider);
      const policies = evaluatePoliciesForActions(result.proposedActions.map((proposal) => proposal.action), incidentForRun, mlForRun);
      setAgenticResult(result);
      setPolicyDecisions(policies);
      setAgentRunStatus({
        status: "success",
        provider,
        runtime,
        message:
          data.message ??
          `${runtime === "openai-agents-sdk" ? "OpenAI Agents SDK" : "Deterministic governed fallback"} produced ${result.proposedActions.length} proposed actions and ${policies.length} policy checks.`,
        latencyMs: Math.round(performance.now() - start),
        actions: result.proposedActions.length,
        policies: policies.length
      });
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
      setAgentRunStatus({
        status: "failed",
        provider: "not-run",
        runtime: "not-run",
        message: text,
        latencyMs: Math.round(performance.now() - start),
        actions: 0,
        policies: 0
      });
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

  const graphNodes: Node<FlowNodeData>[] = [
    {
      id: "sensors",
      type: "agentic",
      position: { x: 0, y: 80 },
      data: { title: "Physical sensors", detail: `${incident.smokePpm} ppm smoke, ${incident.temperatureC} C heat`, status: "physical signal", tone: "cyan" }
    },
    {
      id: "vision",
      type: "agentic",
      position: { x: 210, y: 0 },
      data: {
        title: "Edge vision",
        detail: visionResult ? `${visionProvider}: smoke ${pct(visionResult.maxSmokeConfidence)}, fire ${pct(visionResult.maxFireConfidence)}` : "waiting for inference",
        status: visionRunning ? "running" : visionResult ? "evidence attached" : "not run",
        tone: visionResult ? "emerald" : "cyan"
      }
    },
    {
      id: "ml",
      type: "agentic",
      position: { x: 210, y: 170 },
      data: { title: "ML risk model", detail: `${pct(mlResult.fireProbability)} fire probability, ${mlResult.riskLevel}`, status: model ? "trained tfjs" : "baseline/tfjs-ready", tone: "amber" }
    },
    {
      id: "agent",
      type: "agentic",
      position: { x: 440, y: 85 },
      data: {
        title: "Agentic planner",
        detail: agentRunStatus.status === "success" ? `${agentRunStatus.actions} proposals via ${agentRunStatus.provider}` : agentRunStatus.message,
        status: agentRunStatus.status,
        tone: agentRunStatus.status === "failed" ? "red" : agentRunStatus.status === "running" ? "amber" : agenticResult ? "emerald" : "cyan"
      }
    },
    {
      id: "policy",
      type: "agentic",
      position: { x: 700, y: 0 },
      data: {
        title: "Policy guardrails",
        detail: `${policyDecisions.length || 0} checks, ${policyDecisions.filter((decision) => decision.blocked).length} blocked`,
        status: policyDecisions.length ? "evaluated" : "waiting",
        tone: policyDecisions.some((decision) => decision.blocked) ? "red" : policyDecisions.length ? "emerald" : "cyan"
      }
    },
    {
      id: "human",
      type: "agentic",
      position: { x: 700, y: 170 },
      data: {
        title: "Human approval",
        detail: `${policyDecisions.filter((decision) => decision.requiresHumanApproval).length} approval-gated actions`,
        status: "operator gate",
        tone: policyDecisions.some((decision) => decision.requiresHumanApproval) ? "amber" : "cyan"
      }
    },
    {
      id: "tools",
      type: "agentic",
      position: { x: 950, y: 85 },
      data: {
        title: "Sandbox tools",
        detail: `${agenticResult?.proposedActions.length ?? 0} proposed, no real dispatch`,
        status: "controlled execution",
        tone: agenticResult ? "emerald" : "cyan"
      }
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
  const platformSignals: PlatformSignal[] = [
    {
      label: "LLM runtime",
      value: agentRunStatus.runtime === "not-run" ? (health?.openaiConfigured ? "OpenAI ready" : "Fallback ready") : agentRunStatus.runtime,
      detail: health?.openaiConfigured
        ? `${health.openaiModel}, ${health.openaiMaxOutputTokens} output-token cap, ${health.openaiMaxAgentCallsPerRun} call/run`
        : "Deterministic governed fallback keeps demos running without quota.",
      status: agentRunStatus.status === "running" ? "pending" : health?.openaiConfigured ? "complete" : "attention"
    },
    {
      label: "Schema guardrail",
      value: "Zod validated",
      detail: "Agent output must match AgenticResult before policy or tools consume it.",
      status: agenticResult ? "complete" : "ready"
    },
    {
      label: "Policy engine",
      value: `${policyDecisions.length} checks`,
      detail: `${policyDecisions.filter((decision) => decision.blocked).length} blocked, ${policyDecisions.filter((decision) => decision.requiresHumanApproval).length} approval-gated`,
      status: policyDecisions.length ? (policyDecisions.some((decision) => decision.requiresHumanApproval) ? "attention" : "complete") : "ready"
    },
    {
      label: "Tool boundary",
      value: `${tools.filter((tool) => tool.executionMode === "sandbox").length} sandbox`,
      detail: "Physical tools are contract-defined and sandboxed by default.",
      status: "complete"
    },
    {
      label: "Trace events",
      value: String(trace.length),
      detail: "Every major run step emits structured observability data.",
      status: trace.length > 4 ? "complete" : "ready"
    },
    {
      label: "Decision record",
      value: decisionRecord ? "written" : "not written",
      detail: decisionRecord ? decisionRecord.runId : "Write a record after the planner runs.",
      status: decisionRecord ? "complete" : agenticResult ? "attention" : "ready"
    }
  ];
  const lifecycleStages: LifecycleStage[] = [
    {
      title: "Evidence fusion",
      owner: "Triage + Vision Context",
      status: visionResult ? "complete" : "attention",
      detail: visionResult ? `${visionResult.detections.length} detections joined sensor state.` : "Run vision to attach camera evidence."
    },
    {
      title: "Risk reasoning",
      owner: "Risk Agent",
      status: mlResult ? "complete" : "ready",
      detail: `${pct(mlResult.fireProbability)} probability from ${model ? "trained TensorFlow.js model" : "baseline predictor"}.`
    },
    {
      title: "Structured planning",
      owner: "OpenAI Agents SDK",
      status: agentRunning ? "pending" : agenticResult ? "complete" : "attention",
      detail: agenticResult ? `${agenticResult.proposedActions.length} action proposals with evidence.` : "Run planner to create governed proposals."
    },
    {
      title: "Policy evaluation",
      owner: "TypeScript policy engine",
      status: policyDecisions.length ? "complete" : "ready",
      detail: policyDecisions.length ? `${policyDecisions.length} checks over proposed physical actions.` : "Policy runs after action proposals exist."
    },
    {
      title: "Human approval",
      owner: "Demo operator",
      status: policyDecisions.some((decision) => decision.requiresHumanApproval) ? "attention" : policyDecisions.length ? "complete" : "ready",
      detail: `${policyDecisions.filter((decision) => decision.requiresHumanApproval).length} approval-gated actions.`
    },
    {
      title: "Sandbox execution",
      owner: "Safe tool executor",
      status: agenticResult ? "complete" : "ready",
      detail: "Drone, gate, and authority actions remain simulated in this public deployment."
    },
    {
      title: "Audit closure",
      owner: "Decision record",
      status: decisionRecord ? "complete" : "attention",
      detail: decisionRecord ? `${decisionRecord.trace.length} trace events captured.` : "Write a record to close the incident."
    }
  ];
  return (
    <div className="space-y-6">
      <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }}>
        <Card className="glass-grid">
          <CardHeader>
            <Badge className="w-fit border-cyan-300/30 bg-cyan-300/10 text-cyan-100">Live Physical AI incident console</Badge>
            <CardTitle className="max-w-4xl text-2xl md:text-4xl">Start by changing the physical state.</CardTitle>
            <CardDescription className="max-w-4xl text-base">
              Pick or tune the incident first. Then run vision, ML, and the governed agentic planner to see how the same physical signals become a controlled enterprise response.
            </CardDescription>
          </CardHeader>
          <CardContent className="grid gap-3 border-t border-white/10 pt-5 sm:grid-cols-2 xl:grid-cols-5">
            <MetricCard label="Scenario" value={incident.scenarioName} />
            <MetricCard label="Smoke" value={`${incident.smokePpm} ppm`} />
            <MetricCard label="Heat" value={`${incident.temperatureC} C`} tone={incident.temperatureC > 55 ? "red" : "cyan"} />
            <MetricCard label="ML risk" value={pct(mlResult.fireProbability)} tone={mlResult.riskLevel === "critical" ? "red" : "amber"} />
            <div className="flex gap-2 rounded-lg border border-white/10 bg-black/20 p-3">
              <Button onClick={() => setActiveTab("scenario")} variant="secondary" className="flex-1">Tune</Button>
              <Button onClick={() => void runGuidedIncident()} className="flex-1">Run</Button>
            </div>
          </CardContent>
        </Card>
      </motion.div>

      {message ? <Alert>{message}</Alert> : null}
      {guidedRunStatus.length ? (
        <Card className="shadow-none">
          <CardContent className="grid gap-2 p-4 sm:grid-cols-2 xl:grid-cols-6">
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

        <TabsContent value="scenario">
          <div className="grid gap-4 xl:grid-cols-[1.05fr_.95fr]">
            <Card>
              <CardHeader>
                <Badge className="w-fit border-cyan-300/30 bg-cyan-300/10 text-cyan-100">Step 1</Badge>
                <CardTitle>Set the Physical Incident</CardTitle>
                <CardDescription>
                  This is the part to play with first. Change the sensors and physical constraints, then run the AI pipeline.
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-5">
                <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-3">
                  {scenarioPresets.map((preset, index) => (
                    <button
                      key={preset.incidentId}
                      type="button"
                      onClick={() => loadScenario(index)}
                      className={`rounded-lg border p-3 text-left transition ${
                        incident.incidentId === preset.incidentId ? "border-cyan-300/70 bg-cyan-500/10" : "border-white/10 bg-black/20 hover:border-cyan-300/40"
                      }`}
                    >
                      <div className="flex items-center justify-between gap-3">
                        <span className="text-sm font-semibold text-slate-100">{preset.scenarioName}</span>
                        <Badge>{preset.temperatureC >= 50 ? "hot" : preset.sensorHealth < 0.6 ? "degraded" : "active"}</Badge>
                      </div>
                      <div className="mt-3 grid grid-cols-3 gap-2 text-[11px] text-slate-400">
                        <span>{preset.smokePpm} ppm</span>
                        <span>{preset.temperatureC} C</span>
                        <span>{preset.droneAvailable ? "drone ready" : "no drone"}</span>
                      </div>
                      <p className="mt-2 text-xs leading-5 text-slate-500">
                        Vision smoke {pct(preset.cameraSmokeConfidence)}, fire {pct(preset.cameraFireConfidence)}, occupancy {preset.occupancyStatus}.
                      </p>
                    </button>
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
                <div className="grid gap-2 sm:grid-cols-3">
                  <Button onClick={() => void runVision()} disabled={visionRunning}>
                    {visionRunning ? "Vision Running..." : "Run Vision"}
                  </Button>
                  <Button variant="secondary" onClick={() => void runPrediction()}>
                    Run ML
                  </Button>
                  <Button onClick={() => void runGuidedIncident()}>
                    Run End-to-End
                  </Button>
                </div>
              </CardContent>
            </Card>
            <div className="space-y-4">
              <Card>
                <CardHeader>
                  <CardTitle>What This Physical State Means</CardTitle>
                  <CardDescription>
                    Physical AI is AI connected to real-world signals and actuators. This panel shows how the current state enters the workflow.
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-3">
                  <div className="grid gap-3 sm:grid-cols-2">
                    <MetricCard label="Smoke sensor" value={`${incident.smokePpm} ppm`} />
                    <MetricCard label="Heat sensor" value={`${incident.temperatureC} C`} />
                    <MetricCard label="Camera smoke" value={pct(incident.cameraSmokeConfidence)} />
                    <MetricCard label="Camera fire" value={pct(incident.cameraFireConfidence)} />
                    <MetricCard label="Occupancy" value={incident.occupancyStatus} />
                    <MetricCard label="Actuators" value={`${incident.droneAvailable ? "drone ready" : "no drone"} / ${incident.gateLocked ? "gate locked" : "gate open"}`} />
                  </div>
                  <div className="rounded-lg border border-white/10 bg-black/20 p-4 text-xs leading-5 text-slate-300">
                    <p><span className="font-semibold text-cyan-100">Rules</span> only see smoke and heat.</p>
                    <p className="mt-1"><span className="font-semibold text-amber-100">ML</span> also sees camera, sensor health, false-alarm history, occupancy, and wind.</p>
                    <p className="mt-1"><span className="font-semibold text-emerald-100">Agentic AI</span> uses all evidence plus SOP, tool state, policy, approvals, trace, and records.</p>
                  </div>
                </CardContent>
              </Card>
              <Card>
                <CardHeader>
                  <CardTitle>Current Runtime</CardTitle>
                  <CardDescription>Live services are server-side; physical action tools are sandboxed.</CardDescription>
                </CardHeader>
                <CardContent className="grid gap-3 text-xs text-slate-300">
                  <div className="rounded-md border border-white/10 bg-black/20 p-3">
                    OpenAI: {health?.openaiConfigured ? `${health.openaiModel} configured` : "fallback planner"}
                  </div>
                  <div className="rounded-md border border-white/10 bg-black/20 p-3">
                    Roboflow: {health?.roboflowConfigured ? "live hosted inference configured" : "sample fallback"}
                  </div>
                  <div className="rounded-md border border-white/10 bg-black/20 p-3">
                    Browser ML: TensorFlow.js training and prediction available
                  </div>
                </CardContent>
              </Card>
            </div>
          </div>
        </TabsContent>

        <TabsContent value="comparison">
          <div className="space-y-4">
            <Card className="shadow-none">
              <CardHeader>
                <Badge className="w-fit border-emerald-300/30 bg-emerald-300/10 text-emerald-100">Capability boundary</Badge>
                <CardTitle>Do not confuse detection, prediction, and governed execution</CardTitle>
                <CardDescription>
                  This is the central lesson of the showcase. The same incident flows through three different decision styles, and only the governed agentic path can coordinate tools under policy and human approval.
                </CardDescription>
              </CardHeader>
              <CardContent className="grid gap-3 md:grid-cols-3">
                <div className="rounded-lg border border-cyan-300/20 bg-cyan-400/10 p-4">
                  <p className="text-sm font-semibold text-cyan-100">Rule-based automation detects</p>
                  <p className="mt-2 text-xs leading-5 text-slate-300">Fixed thresholds raise alarms from smoke and heat. It cannot fuse camera, occupancy, drone, policy, or SOP context.</p>
                </div>
                <div className="rounded-lg border border-amber-300/20 bg-amber-400/10 p-4">
                  <p className="text-sm font-semibold text-amber-100">ML-based prediction estimates risk</p>
                  <p className="mt-2 text-xs leading-5 text-slate-300">TensorFlow.js turns fused features into probability. It should not unlock gates, notify authorities, or dispatch drones.</p>
                </div>
                <div className="rounded-lg border border-emerald-300/20 bg-emerald-400/10 p-4">
                  <p className="text-sm font-semibold text-emerald-100">Enterprise agentic AI governs coordination</p>
                  <p className="mt-2 text-xs leading-5 text-slate-300">The agent plans with evidence and tools, while schema, policy, humans, sandbox execution, tracing, and records constrain action.</p>
                </div>
              </CardContent>
            </Card>
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
          <div className="space-y-4">
            <Card className="shadow-none">
              <CardHeader>
                <div className="flex flex-col gap-3 lg:flex-row lg:items-start lg:justify-between">
                  <div>
                    <Badge className="mb-3 w-fit border-emerald-300/30 bg-emerald-300/10 text-emerald-100">Agentic control plane cockpit</Badge>
                    <CardTitle>Reasoning is only one layer of the platform</CardTitle>
                    <CardDescription className="mt-2 max-w-4xl">
                      A production agentic system needs runtime caps, schema validation, policy checks, approval gates, tool contracts,
                      traces, and decision records. This cockpit shows those platform controls as live state.
                    </CardDescription>
                  </div>
                  <div className="grid gap-2 sm:grid-cols-2 lg:min-w-96">
                    <Button onClick={() => void runAgent()} disabled={agentRunning} className="min-h-11">
                      <Workflow className="h-4 w-4" /> {agentRunning ? "Planner Running..." : "Run Agentic Planner"}
                    </Button>
                    <Button variant="secondary" onClick={() => void runGuidedIncident()} className="min-h-11">
                      <Route className="h-4 w-4" /> Run End-to-End
                    </Button>
                  </div>
                </div>
              </CardHeader>
              <CardContent className="grid gap-3 md:grid-cols-2 xl:grid-cols-3">
                {platformSignals.map((signal) => (
                  <div key={signal.label} className="rounded-lg border border-white/10 bg-black/20 p-4">
                    <div className="flex items-start justify-between gap-3">
                      <div>
                        <p className="text-xs uppercase text-slate-500">{signal.label}</p>
                        <p className="mt-1 text-lg font-semibold text-slate-100">{signal.value}</p>
                      </div>
                      <Badge className={statusClass(signal.status)}>{signal.status}</Badge>
                    </div>
                    <p className="mt-2 text-xs leading-5 text-slate-400">{signal.detail}</p>
                  </div>
                ))}
              </CardContent>
            </Card>
            <Card className="shadow-none">
              <CardHeader>
                <CardTitle className="text-base">Agentic Execution Lifecycle</CardTitle>
                <CardDescription>Each step is a control boundary. The model can propose; the platform decides what may proceed.</CardDescription>
              </CardHeader>
              <CardContent className="grid gap-3 xl:grid-cols-7">
                {lifecycleStages.map((stage, index) => (
                  <motion.div
                    key={stage.title}
                    initial={{ opacity: 0, y: 8 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: index * 0.03 }}
                    className="rounded-lg border border-white/10 bg-black/20 p-3"
                  >
                    <div className="flex items-start justify-between gap-2">
                      <span className="text-[11px] uppercase text-slate-500">{index + 1}</span>
                      <Badge className={statusClass(stage.status)}>{stage.status}</Badge>
                    </div>
                    <p className="mt-3 text-sm font-semibold text-slate-100">{stage.title}</p>
                    <p className="mt-1 text-[11px] text-cyan-100">{stage.owner}</p>
                    <p className="mt-2 text-xs leading-5 text-slate-400">{stage.detail}</p>
                  </motion.div>
                ))}
              </CardContent>
            </Card>
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
                  <ReactFlow nodes={graphNodes} edges={graphEdges} nodeTypes={nodeTypes} fitView>
                    <Background />
                    <MiniMap />
                    <Controls />
                  </ReactFlow>
                </div>
                <div className="mt-4 grid gap-3 md:grid-cols-3">
                  <div className="rounded-md border border-white/10 bg-black/20 p-3 text-xs">
                    <p className="font-semibold text-slate-100">Evidence in</p>
                    <p className="mt-1 text-slate-400">Sensors, vision detections, ML probability, SOP references, tool state.</p>
                  </div>
                  <div className="rounded-md border border-white/10 bg-black/20 p-3 text-xs">
                    <p className="font-semibold text-slate-100">Governance in path</p>
                    <p className="mt-1 text-slate-400">Schema validation, TypeScript policy checks, approval gates, sandbox tools.</p>
                  </div>
                  <div className="rounded-md border border-white/10 bg-black/20 p-3 text-xs">
                    <p className="font-semibold text-slate-100">Audit out</p>
                    <p className="mt-1 text-slate-400">Trace events, proposed actions, blocked actions, approvals, decision record.</p>
                  </div>
                </div>
              </CardContent>
            </Card>
            <div className="grid gap-3">
              <Card className="shadow-none">
                <CardHeader className="p-4 pb-2">
                  <div className="flex items-start justify-between gap-3">
                    <div>
                      <CardTitle className="text-sm">Agent Run Console</CardTitle>
                      <CardDescription>Visible proof of what happened when the planner ran.</CardDescription>
                    </div>
                    <Badge className={statusClass(agentRunStatus.status === "idle" ? "ready" : agentRunStatus.status === "running" ? "pending" : agentRunStatus.status === "success" ? "complete" : "attention")}>
                      {agentRunStatus.status}
                    </Badge>
                  </div>
                </CardHeader>
                <CardContent className="space-y-3 p-4 pt-0">
                  <div className="grid gap-2 sm:grid-cols-2">
                    <div className="rounded-md border border-white/10 bg-black/20 p-3 text-xs">
                      <p className="text-slate-500">Runtime</p>
                      <p className="mt-1 font-semibold text-slate-100">{agentRunStatus.runtime}</p>
                    </div>
                    <div className="rounded-md border border-white/10 bg-black/20 p-3 text-xs">
                      <p className="text-slate-500">Agent SDK</p>
                      <p className="mt-1 font-semibold text-slate-100">@openai/agents + Zod output guardrail</p>
                    </div>
                    <div className="rounded-md border border-white/10 bg-black/20 p-3 text-xs">
                      <p className="text-slate-500">Provider</p>
                      <p className="mt-1 font-semibold text-slate-100">{agentRunStatus.provider}</p>
                    </div>
                    <div className="rounded-md border border-white/10 bg-black/20 p-3 text-xs">
                      <p className="text-slate-500">Output</p>
                      <p className="mt-1 font-semibold text-slate-100">
                        {agentRunStatus.actions} actions, {agentRunStatus.policies} policy checks{agentRunStatus.latencyMs ? `, ${agentRunStatus.latencyMs}ms` : ""}
                      </p>
                    </div>
                  </div>
                  <p className="rounded-md border border-white/10 bg-black/20 p-3 text-xs leading-5 text-slate-300">{agentRunStatus.message}</p>
                </CardContent>
              </Card>
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
