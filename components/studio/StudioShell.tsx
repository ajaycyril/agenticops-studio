"use client";

import { useEffect, useMemo, useState } from "react";
import type * as tf from "@tensorflow/tfjs";
import {
  Accordion,
  AccordionDetails,
  AccordionSummary,
  Alert,
  Box,
  Button,
  Chip,
  CssBaseline,
  Divider,
  LinearProgress,
  Paper,
  Stack,
  ThemeProvider,
  ToggleButton,
  ToggleButtonGroup,
  Typography,
  createTheme
} from "@mui/material";
import AccountTreeIcon from "@mui/icons-material/AccountTree";
import AutoAwesomeIcon from "@mui/icons-material/AutoAwesome";
import CameraAltIcon from "@mui/icons-material/CameraAlt";
import ExpandMoreIcon from "@mui/icons-material/ExpandMore";
import FactCheckIcon from "@mui/icons-material/FactCheck";
import MemoryIcon from "@mui/icons-material/Memory";
import ModelTrainingIcon from "@mui/icons-material/ModelTraining";
import PolicyIcon from "@mui/icons-material/Policy";
import SensorsIcon from "@mui/icons-material/Sensors";
import ShieldIcon from "@mui/icons-material/Shield";
import { JsonInspector } from "@/components/studio/JsonInspector";
import { getToolRegistry } from "@/lib/agent/tool-registry";
import { buildDecisionRecord } from "@/lib/decision-record/build-decision-record";
import { DEFAULT_MODEL_VERSION } from "@/lib/constants";
import { incidentToFeatures } from "@/lib/ml/dataset";
import { approximateFeatureImportance } from "@/lib/ml/feature-importance";
import { loadModelVersion, saveModelVersion } from "@/lib/ml/model-store";
import { heuristicRiskPrediction } from "@/lib/ml/predict-risk";
import { trainRiskModel } from "@/lib/ml/train-risk-model";
import { evaluatePoliciesForActions } from "@/lib/policies/policy-evaluator";
import { evaluateRules } from "@/lib/rule/rule-engine";
import { clonePreset, scenarioPresets } from "@/lib/scenario-presets";
import { createTraceEvent } from "@/lib/trace/create-trace-event";
import { saveTrace } from "@/lib/trace/trace-store";
import type { AgenticResult, DecisionRecord, IncidentState, MLResult, PolicyDecision, RiskLevel, RuleResult, TraceEvent, VisionResult } from "@/lib/types";

const materialTheme = createTheme({
  palette: {
    mode: "dark",
    background: { default: "#07111f", paper: "#101827" },
    primary: { main: "#22d3ee" },
    secondary: { main: "#60a5fa" },
    warning: { main: "#f59e0b" },
    error: { main: "#ef4444" },
    success: { main: "#22c55e" },
    text: { primary: "#eef6ff", secondary: "#9fb1c8" }
  },
  shape: { borderRadius: 10 },
  typography: {
    fontFamily: "Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif",
    h3: { letterSpacing: 0, fontWeight: 760 },
    h5: { letterSpacing: 0, fontWeight: 740 },
    h6: { letterSpacing: 0, fontWeight: 740 }
  },
  components: {
    MuiPaper: { styleOverrides: { root: { backgroundImage: "none" } } },
    MuiButton: { styleOverrides: { root: { textTransform: "none", fontWeight: 800 } } },
    MuiChip: { styleOverrides: { root: { fontWeight: 760 } } },
    MuiToggleButton: { styleOverrides: { root: { textTransform: "none", justifyContent: "flex-start", fontWeight: 760 } } }
  }
});

const sampleImageMap = {
  "cooking-smoke": { file: "cooking-smoke.jpg", label: "Cooking smoke" },
  "fire-smoke-room": { file: "fire-smoke-room.jpg", label: "Confirmed fire" },
  "unclear-camera": { file: "unclear-camera.jpg", label: "Unclear camera" }
} as const;

const demoPresets = [
  { presetIndex: 1, sample: "fire-smoke-room", label: "Confirmed fire" },
  { presetIndex: 2, sample: "unclear-camera", label: "Unclear camera" },
  { presetIndex: 0, sample: "cooking-smoke", label: "False alarm" }
] as const;

type HealthStatus = {
  status: "ok";
  openaiConfigured: boolean;
  roboflowConfigured: boolean;
  openaiModel: string;
  openaiMaxOutputTokens: number;
  openaiTimeoutMs: number;
  openaiMaxAgentCallsPerRun: number;
};

type AgentControls = {
  operatingMode: "balanced" | "conservative" | "rapid_response";
  authorityPosture: "strict" | "approval_gated" | "critical_only";
};

type RunStepStatus = "waiting" | "running" | "done" | "fallback" | "failed";
type RunStepId = "signals" | "vision" | "ml" | "agent" | "guardrails" | "actions";
type RunStep = { id: RunStepId; title: string; short: string; detail: string; status: RunStepStatus };

const initialRunSteps: RunStep[] = [
  { id: "signals", title: "Physical signals", short: "Collect", detail: "Waiting for run", status: "waiting" },
  { id: "vision", title: "Edge vision", short: "See", detail: "Waiting for run", status: "waiting" },
  { id: "ml", title: "ML risk", short: "Predict", detail: "Waiting for run", status: "waiting" },
  { id: "agent", title: "Agent plan", short: "Reason", detail: "Waiting for run", status: "waiting" },
  { id: "guardrails", title: "Guardrails", short: "Check", detail: "Waiting for run", status: "waiting" },
  { id: "actions", title: "Actions + audit", short: "Record", detail: "Waiting for run", status: "waiting" }
];

const stepIcons: Record<RunStepId, React.ReactNode> = {
  signals: <SensorsIcon />,
  vision: <CameraAltIcon />,
  ml: <ModelTrainingIcon />,
  agent: <AutoAwesomeIcon />,
  guardrails: <PolicyIcon />,
  actions: <FactCheckIcon />
};

function pct(value: number) {
  return `${Math.round(value * 100)}%`;
}

function riskFromProbability(probability: number): RiskLevel {
  if (probability >= 0.82) return "critical";
  if (probability >= 0.62) return "high";
  if (probability >= 0.36) return "medium";
  return "low";
}

function formatAction(action: string) {
  return action.replaceAll("_", " ");
}

function stepColor(status: RunStepStatus): "default" | "primary" | "success" | "warning" | "error" {
  if (status === "running") return "primary";
  if (status === "done") return "success";
  if (status === "fallback") return "warning";
  if (status === "failed") return "error";
  return "default";
}

function Surface({ children, sx }: { children: React.ReactNode; sx?: object }) {
  return (
    <Paper
      elevation={0}
      sx={{
        border: "1px solid rgba(148, 163, 184, 0.18)",
        bgcolor: "rgba(15, 23, 42, 0.9)",
        p: { xs: 1.5, md: 2 },
        ...sx
      }}
    >
      {children}
    </Paper>
  );
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <Box sx={{ display: "flex", justifyContent: "space-between", gap: 2, py: 0.75 }}>
      <Typography variant="body2" color="text.secondary">
        {label}
      </Typography>
      <Typography variant="body2" sx={{ textAlign: "right", fontWeight: 800 }}>
        {value}
      </Typography>
    </Box>
  );
}

export function StudioShell() {
  const [incident, setIncident] = useState<IncidentState>(() => clonePreset(1));
  const [ruleResult, setRuleResult] = useState<RuleResult>(() => evaluateRules(clonePreset(1)));
  const [mlResult, setMlResult] = useState<MLResult>(() => heuristicRiskPrediction(clonePreset(1)));
  const [visionResult, setVisionResult] = useState<VisionResult | undefined>();
  const [agenticResult, setAgenticResult] = useState<AgenticResult | undefined>();
  const [policyDecisions, setPolicyDecisions] = useState<PolicyDecision[]>([]);
  const [decisionRecord, setDecisionRecord] = useState<DecisionRecord | undefined>();
  const [trace, setTrace] = useState<TraceEvent[]>([
    createTraceEvent({ type: "incident_created", actor: "system", status: "success", output: clonePreset(1) })
  ]);
  const [health, setHealth] = useState<HealthStatus | undefined>();
  const [message, setMessage] = useState<string>();
  const [running, setRunning] = useState(false);
  const [training, setTraining] = useState(false);
  const [model, setModel] = useState<tf.LayersModel>();
  const [modelVersion, setModelVersion] = useState(() => loadModelVersion() ?? DEFAULT_MODEL_VERSION);
  const [modelMetrics, setModelMetrics] = useState<MLResult["metrics"]>();
  const [sampleName, setSampleName] = useState<keyof typeof sampleImageMap>("fire-smoke-room");
  const [agentControls, setAgentControls] = useState<AgentControls>({ operatingMode: "balanced", authorityPosture: "critical_only" });
  const [runSteps, setRunSteps] = useState<RunStep[]>(initialRunSteps);
  const [activeStepId, setActiveStepId] = useState<RunStepId>("signals");
  const [agentRuntime, setAgentRuntime] = useState<{ provider?: string; runtime?: string; message?: string; latencyMs?: number }>({});

  const tools = useMemo(() => getToolRegistry(), []);
  const selectedStep = runSteps.find((step) => step.id === activeStepId) ?? runSteps[0];
  const selectedDemo = demoPresets.find((demo) => scenarioPresets[demo.presetIndex].incidentId === incident.incidentId) ?? demoPresets[0];
  const blockedPolicies = policyDecisions.filter((policy) => policy.blocked);
  const gatedPolicies = policyDecisions.filter((policy) => policy.requiresHumanApproval && !policy.blocked);
  const topFeature = mlResult.featureImportance[0];
  const resultJson = { incident, ruleResult, mlResult, visionResult, agenticResult, policyDecisions, decisionRecord };

  useEffect(() => {
    fetch("/api/health")
      .then((response) => response.json())
      .then((data: HealthStatus) => setHealth(data))
      .catch(() => setHealth(undefined));
  }, []);

  function appendTrace(event: Omit<TraceEvent, "id" | "timestamp">) {
    const created = createTraceEvent(event);
    setTrace((current) => {
      const next = [...current, created];
      saveTrace(next);
      return next;
    });
    return created;
  }

  function updateRunStep(id: RunStepId, status: RunStepStatus, detail: string) {
    setActiveStepId(id);
    setRunSteps((current) => current.map((step) => (step.id === id ? { ...step, status, detail } : step)));
  }

  function resetRunState() {
    setVisionResult(undefined);
    setAgenticResult(undefined);
    setPolicyDecisions([]);
    setDecisionRecord(undefined);
    setRunSteps(initialRunSteps);
    setAgentRuntime({});
    setActiveStepId("signals");
  }

  function loadDemo(presetIndex: number, sample: keyof typeof sampleImageMap) {
    const next = clonePreset(presetIndex);
    setIncident(next);
    setSampleName(sample);
    setRuleResult(evaluateRules(next));
    setMlResult({ ...heuristicRiskPrediction(next), modelVersion, metrics: modelMetrics });
    resetRunState();
    setMessage(undefined);
    appendTrace({ type: "incident_created", actor: "system", status: "success", output: next, explanation: `Loaded ${next.scenarioName}.` });
  }

  function runRules(incidentForRun: IncidentState) {
    const result = evaluateRules(incidentForRun);
    setRuleResult(result);
    appendTrace({ type: "rule_engine_evaluated", actor: "system", input: incidentForRun, output: result, status: "success" });
    return result;
  }

  async function trainBrowserModel(incidentForRun: IncidentState) {
    setTraining(true);
    appendTrace({ type: "ml_model_training_started", actor: "ml_model", input: { scenarioName: incidentForRun.scenarioName }, status: "pending" });
    try {
      const trained = await trainRiskModel({ size: 220, epochs: 18, falseAlarmBias: incidentForRun.historicalFalseAlarmRate, learningRate: 0.08 });
      setModel(trained.model);
      setModelMetrics(trained.metrics);
      setModelVersion(trained.modelVersion);
      saveModelVersion(trained.modelVersion);
      appendTrace({ type: "ml_model_training_completed", actor: "ml_model", status: "success", output: trained.metrics });
      return trained;
    } finally {
      setTraining(false);
    }
  }

  async function runPrediction(incidentForRun: IncidentState, modelOverride?: tf.LayersModel, versionOverride = modelVersion, metricsOverride = modelMetrics) {
    let probability: number;
    const features = incidentToFeatures(incidentForRun);
    const activeModel = modelOverride ?? model;
    if (activeModel) {
      const tfjs = await import("@tensorflow/tfjs");
      const prediction = activeModel.predict(tfjs.tensor2d([features])) as tf.Tensor;
      probability = (await prediction.data())[0];
      prediction.dispose();
    } else {
      probability = heuristicRiskPrediction(incidentForRun).fireProbability;
    }
    const result: MLResult = {
      fireProbability: Number(probability.toFixed(3)),
      riskLevel: riskFromProbability(probability),
      modelVersion: versionOverride,
      metrics: metricsOverride,
      featureImportance: approximateFeatureImportance(features),
      explanation: "ML predicts risk only. It never executes physical actions."
    };
    setMlResult(result);
    appendTrace({ type: "ml_model_predicted", actor: "ml_model", input: incidentForRun, output: result, status: "success" });
    return result;
  }

  async function loadSampleImageAsDataUrl(name: keyof typeof sampleImageMap) {
    const response = await fetch(`/sample-images/${sampleImageMap[name].file}`);
    if (!response.ok) throw new Error(`Failed to load sample image ${name}`);
    const blob = await response.blob();
    return await new Promise<string>((resolve, reject) => {
      const reader = new FileReader();
      reader.onloadend = () => (typeof reader.result === "string" ? resolve(reader.result) : reject(new Error("Invalid image payload")));
      reader.onerror = () => reject(new Error("Failed to read sample image"));
      reader.readAsDataURL(blob);
    });
  }

  async function runVision(incidentForRun: IncidentState) {
    const imagePayload = await loadSampleImageAsDataUrl(sampleName);
    const response = await fetch("/api/vision/roboflow", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ incidentId: incidentForRun.incidentId, sampleName, imageBase64: imagePayload })
    });
    const data = await response.json();
    if (!data.ok) throw new Error(data.error.message);
    const result = data.result as VisionResult;
    const nextIncident: IncidentState = {
      ...incidentForRun,
      cameraSmokeConfidence: result.maxSmokeConfidence,
      cameraFireConfidence: result.maxFireConfidence,
      visionProvider: result.provider,
      visionDetections: result.detections,
      imageUrl: `/sample-images/${sampleImageMap[sampleName].file}`
    };
    setIncident(nextIncident);
    setVisionResult(result);
    appendTrace({ type: "vision_model_called", actor: "vision_model", input: { sampleName }, output: result, status: "success", latencyMs: result.latencyMs });
    return { result, nextIncident };
  }

  async function runComparison() {
    setRunning(true);
    setRunSteps(initialRunSteps);
    setAgentRuntime({});
    setMessage("Running. Watch the horizontal flow; the selected stage explains what is happening.");
    try {
      const incidentForRun = incident;
      updateRunStep("signals", "running", "Reading smoke, heat, camera state, occupancy, drone, and gate context.");
      const rule = runRules(incidentForRun);
      updateRunStep("signals", "done", `Rule mode says ${formatAction(rule.action)} using smoke and heat only.`);

      updateRunStep("vision", "running", `Calling server-side vision route for ${sampleImageMap[sampleName].label}.`);
      const vision = await runVision(incidentForRun);
      updateRunStep("vision", vision.result.provider === "sample" ? "fallback" : "done", `${vision.result.provider}: fire ${pct(vision.result.maxFireConfidence)}, smoke ${pct(vision.result.maxSmokeConfidence)}.`);

      updateRunStep("ml", "running", model ? "Scoring with active browser ML model." : "Training TensorFlow.js model in browser, then scoring.");
      const trained = model ? undefined : await trainBrowserModel(vision.nextIncident);
      const ml = await runPrediction(vision.nextIncident, trained?.model ?? model, trained?.modelVersion ?? modelVersion, trained?.metrics ?? modelMetrics);
      updateRunStep("ml", "done", `ML predicts ${pct(ml.fireProbability)} fire probability. It does not act.`);

      updateRunStep("agent", "running", `${health?.openaiConfigured ? "Calling OpenAI Agents SDK" : "Using fallback planner"} with ${agentControls.operatingMode} posture.`);
      appendTrace({ type: "agent_called", actor: "llm_agent", input: { incidentId: vision.nextIncident.incidentId }, status: "pending" });
      const agentStart = Date.now();
      const response = await fetch("/api/agent/run", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          incident: vision.nextIncident,
          ruleResult: rule,
          mlResult: ml,
          visionResult: vision.result,
          toolRegistry: tools,
          policySummary: "TypeScript fire response policy evaluator v1",
          agentControls: {
            ...agentControls,
            operatorInstruction: "Explain the governed handoff: evidence, proposed tool calls, policy checks, human approval gates, sandbox execution, and audit."
          }
        })
      });
      const data = await response.json();
      if (!data.ok) throw new Error(data.error.message);
      const result = data.result as AgenticResult;
      const latencyMs = Date.now() - agentStart;
      setAgentRuntime({ provider: data.provider, runtime: data.runtime, message: data.message, latencyMs });
      updateRunStep("agent", data.provider === "openai" ? "done" : "fallback", `${data.runtime}: proposed ${result.proposedActions.length} governed actions.`);

      updateRunStep("guardrails", "running", "Checking proposed actions against policy, approval gates, and physical safety.");
      const policies = evaluatePoliciesForActions(result.proposedActions.map((proposal) => proposal.action), vision.nextIncident, ml);
      updateRunStep("guardrails", "done", `${policies.length} checks: ${policies.filter((policy) => policy.blocked).length} blocked, ${policies.filter((policy) => policy.requiresHumanApproval && !policy.blocked).length} approval-gated.`);

      updateRunStep("actions", "running", "Writing auditable decision record. Physical actions remain sandboxed.");
      const validatedEvent = appendTrace({ type: "agent_output_validated", actor: "llm_agent", output: result, status: "success", explanation: data.message });
      const policyEvent = appendTrace({ type: "policy_checked", actor: "policy", output: policies, status: "success" });
      const record = buildDecisionRecord({
        runId: `RUN-${Date.now()}`,
        incident: vision.nextIncident,
        ruleResult: rule,
        mlResult: ml,
        visionResult: vision.result,
        agenticResult: result,
        policyDecisions: policies,
        trace: [...trace, validatedEvent, policyEvent]
      });
      setAgenticResult(result);
      setPolicyDecisions(policies);
      setDecisionRecord(record);
      sessionStorage.setItem("agenticops.latestDecisionRecord", JSON.stringify(record));
      appendTrace({ type: "decision_record_written", actor: "system", output: { runId: record.runId }, status: "success" });
      updateRunStep("actions", "done", `Record ${record.runId} written. Proposed physical actions are sandbox only.`);
      setMessage("Done. Click any stage in the horizontal flow to understand that part of agentic AI.");
    } catch (error) {
      const text = error instanceof Error ? error.message : "Run failed.";
      setMessage(text);
      setRunSteps((current) => current.map((step) => (step.status === "running" ? { ...step, status: "failed", detail: text } : step)));
      appendTrace({ type: "error", actor: "system", output: { message: text }, status: "failed" });
    } finally {
      setRunning(false);
    }
  }

  function stageExplanation(id: RunStepId) {
    if (id === "signals") {
      return {
        title: "Rule-based automation detects",
        body: "Rules are deterministic. They only inspect smoke and heat, then raise or escalate an alarm. They do not understand camera context, SOPs, tools, approvals, or policy.",
        metrics: [
          ["Smoke", `${incident.smokePpm} ppm`],
          ["Heat", `${incident.temperatureC} C`],
          ["Rule output", formatAction(ruleResult.action)],
          ["Rule severity", ruleResult.severity]
        ]
      };
    }
    if (id === "vision") {
      return {
        title: "Edge / vision AI sees",
        body: "Vision adds perception. The browser sends the selected camera frame to a server route. Roboflow runs when configured; sample fallback is clearly labeled when needed.",
        metrics: [
          ["Frame", sampleImageMap[sampleName].label],
          ["Provider", visionResult?.provider ?? (health?.roboflowConfigured ? "roboflow ready" : "sample fallback")],
          ["Fire confidence", visionResult ? pct(visionResult.maxFireConfidence) : pct(incident.cameraFireConfidence)],
          ["Smoke confidence", visionResult ? pct(visionResult.maxSmokeConfidence) : pct(incident.cameraSmokeConfidence)]
        ]
      };
    }
    if (id === "ml") {
      return {
        title: "ML predicts, but does not act",
        body: "The browser trains or uses a TensorFlow.js risk model. It outputs probability and feature importance. It cannot dispatch drones, unlock gates, or notify authorities.",
        metrics: [
          ["Fire probability", pct(mlResult.fireProbability)],
          ["Risk level", mlResult.riskLevel],
          ["Model", mlResult.modelVersion.includes("tfjs") ? "browser trained" : "baseline before run"],
          ["Top feature", topFeature?.feature ?? "after run"]
        ]
      };
    }
    if (id === "agent") {
      return {
        title: "Agentic AI coordinates",
        body: "The planner reasons over rules, vision, ML, SOP context, tool contracts, and your guardrail posture. It proposes structured actions; it does not execute physical actions directly.",
        metrics: [
          ["Provider", agentRuntime.provider ?? (health?.openaiConfigured ? "OpenAI ready" : "fallback")],
          ["Runtime", agentRuntime.runtime ?? "after run"],
          ["Latency", agentRuntime.latencyMs ? `${(agentRuntime.latencyMs / 1000).toFixed(1)}s` : "after run"],
          ["Actions proposed", agenticResult ? String(agenticResult.proposedActions.length) : "after run"]
        ]
      };
    }
    if (id === "guardrails") {
      return {
        title: "Guardrails decide what is allowed",
        body: "Policy checks sit between reasoning and execution. They can block an action, require approval, or allow sandbox execution. This is the enterprise layer most agent demos skip.",
        metrics: [
          ["Agent posture", agentControls.operatingMode],
          ["Authority rule", agentControls.authorityPosture],
          ["Blocked", String(blockedPolicies.length)],
          ["Approval-gated", String(gatedPolicies.length)]
        ]
      };
    }
    return {
      title: "Actions are sandboxed and audited",
      body: "Tools are visible, typed, and governed. Drone, gate, and authority actions are simulated by design. The decision record captures inputs, model outputs, policy decisions, and trace.",
      metrics: [
        ["Physical actions", "sandbox only"],
        ["Decision record", decisionRecord?.runId ?? "after run"],
        ["Trace events", String(trace.length)],
        ["Audit storage", "session/local demo store"]
      ]
    };
  }

  const explanation = stageExplanation(activeStepId);
  const proposedActions = agenticResult?.proposedActions ?? [];
  const agentWork = [
    ["Triage", agenticResult?.incidentSummary ?? "Summarizes severity after run."],
    ["Vision", visionResult ? `${visionResult.provider}: fire ${pct(visionResult.maxFireConfidence)}` : "Interprets camera confidence after vision run."],
    ["Risk", `${pct(mlResult.fireProbability)} fire probability, ${mlResult.riskLevel} risk.`],
    ["Policy", policyDecisions.length ? `${blockedPolicies.length} blocked, ${gatedPolicies.length} approval-gated.` : "Checks proposed actions after planner run."],
    ["Planner", proposedActions.length ? proposedActions.map((item) => formatAction(item.action)).join(", ") : "Produces a governed action plan."]
  ];

  return (
    <ThemeProvider theme={materialTheme}>
      <CssBaseline />
      <Box sx={{ mx: "auto", maxWidth: 1500, px: { xs: 0, md: 1 }, pb: 2 }}>
        <Surface sx={{ minHeight: { lg: "calc(100vh - 104px)" }, display: "grid", gap: 2 }}>
          <Box sx={{ display: "grid", gridTemplateColumns: { xs: "1fr", lg: "1fr auto" }, gap: 2, alignItems: "center" }}>
            <Box>
              <Stack direction="row" spacing={1} sx={{ flexWrap: "wrap", gap: 1 }}>
                <Chip color="primary" icon={<ShieldIcon />} label="Agentic AI control plane" />
                <Chip color={health?.openaiConfigured ? "success" : "warning"} label={`OpenAI ${health?.openaiConfigured ? health.openaiModel : "fallback"}`} />
                <Chip color={health?.roboflowConfigured ? "success" : "warning"} label={`Vision ${health?.roboflowConfigured ? "Roboflow" : "sample"}`} />
                <Chip color="warning" label="Physical actions sandboxed" />
              </Stack>
              <Typography variant="h3" sx={{ mt: 1.5 }}>
                Agentic AI is the governed handoff from evidence to action.
              </Typography>
              <Typography variant="body2" color="text.secondary" sx={{ mt: 0.75, maxWidth: 900 }}>
                Rule-based automation detects. ML predicts. Agentic AI coordinates. Enterprise agentic AI governs execution.
              </Typography>
            </Box>
            <Button size="large" variant="contained" startIcon={<AccountTreeIcon />} onClick={() => void runComparison()} disabled={running || training} sx={{ minWidth: { lg: 310 }, py: 1.5 }}>
              {running || training ? "Running live workflow..." : "Run live agentic workflow"}
            </Button>
          </Box>

          {running || training ? <LinearProgress /> : null}
          {message ? <Alert severity={message.startsWith("Done") ? "success" : message.includes("failed") ? "error" : "info"}>{message}</Alert> : null}

          <Box sx={{ display: "grid", gridTemplateColumns: { xs: "1fr", md: "repeat(6, 1fr)" }, gap: 1 }}>
            {runSteps.map((step, index) => (
              <Paper
                key={step.id}
                component="button"
                onClick={() => setActiveStepId(step.id)}
                elevation={0}
                sx={{
                  p: 1.25,
                  minHeight: 112,
                  textAlign: "left",
                  cursor: "pointer",
                  color: "text.primary",
                  bgcolor: activeStepId === step.id ? "rgba(34, 211, 238, 0.16)" : "rgba(2, 6, 23, 0.6)",
                  border: activeStepId === step.id ? "1px solid rgba(34, 211, 238, 0.62)" : "1px solid rgba(148, 163, 184, 0.16)"
                }}
              >
                <Stack direction="row" sx={{ justifyContent: "space-between", alignItems: "center" }}>
                  <Box sx={{ color: "primary.main", display: "flex" }}>{stepIcons[step.id]}</Box>
                  <Chip size="small" color={stepColor(step.status)} label={step.status} />
                </Stack>
                <Typography variant="caption" color="text.secondary" sx={{ display: "block", mt: 0.75 }}>
                  {index + 1}. {step.short}
                </Typography>
                <Typography variant="subtitle2" sx={{ fontWeight: 900 }}>
                  {step.title}
                </Typography>
              </Paper>
            ))}
          </Box>

          <Box sx={{ display: "grid", gridTemplateColumns: { xs: "1fr", xl: "310px 1fr 390px" }, gap: 2, minHeight: 0 }}>
            <Paper elevation={0} sx={{ p: 2, bgcolor: "rgba(2, 6, 23, 0.58)", border: "1px solid rgba(148, 163, 184, 0.16)" }}>
              <Typography variant="h6">Setup</Typography>
              <Typography variant="caption" color="text.secondary">
                Keep the demo simple: pick one incident and guardrail posture.
              </Typography>
              <ToggleButtonGroup
                exclusive
                fullWidth
                value={selectedDemo.label}
                onChange={(_, value) => {
                  const demo = demoPresets.find((item) => item.label === value);
                  if (demo) loadDemo(demo.presetIndex, demo.sample);
                }}
                sx={{ mt: 1.5, display: "grid", gap: 1 }}
              >
                {demoPresets.map((demo) => (
                  <ToggleButton key={demo.label} value={demo.label}>
                    {demo.label}
                  </ToggleButton>
                ))}
              </ToggleButtonGroup>
              <Divider sx={{ my: 2 }} />
              <Typography variant="subtitle2">Configurable guardrails</Typography>
              <Typography variant="caption" color="text.secondary">
                These values are sent to the planner route.
              </Typography>
              <ToggleButtonGroup
                exclusive
                fullWidth
                value={agentControls.operatingMode}
                onChange={(_, value) => value && setAgentControls((current) => ({ ...current, operatingMode: value }))}
                sx={{ mt: 1, display: "grid", gap: 1 }}
              >
                <ToggleButton value="conservative">Conservative</ToggleButton>
                <ToggleButton value="balanced">Balanced</ToggleButton>
                <ToggleButton value="rapid_response">Rapid response</ToggleButton>
              </ToggleButtonGroup>
              <ToggleButtonGroup
                exclusive
                fullWidth
                value={agentControls.authorityPosture}
                onChange={(_, value) => value && setAgentControls((current) => ({ ...current, authorityPosture: value }))}
                sx={{ mt: 1, display: "grid", gap: 1 }}
              >
                <ToggleButton value="strict">Strict authority</ToggleButton>
                <ToggleButton value="approval_gated">Approval gated</ToggleButton>
                <ToggleButton value="critical_only">Critical only</ToggleButton>
              </ToggleButtonGroup>
              <Divider sx={{ my: 2 }} />
              <Metric label="Smoke" value={`${incident.smokePpm} ppm`} />
              <Metric label="Heat" value={`${incident.temperatureC} C`} />
              <Metric label="Camera frame" value={sampleImageMap[sampleName].label} />
            </Paper>

            <Paper elevation={0} sx={{ p: 2, bgcolor: "rgba(2, 6, 23, 0.5)", border: "1px solid rgba(34, 211, 238, 0.18)" }}>
              <Stack direction={{ xs: "column", md: "row" }} sx={{ justifyContent: "space-between", gap: 1 }}>
                <Box>
                  <Typography variant="h5">{explanation.title}</Typography>
                  <Typography variant="body2" color="text.secondary" sx={{ mt: 0.75, maxWidth: 780 }}>
                    {explanation.body}
                  </Typography>
                </Box>
                <Chip color={stepColor(selectedStep.status)} label={selectedStep.status} sx={{ alignSelf: { xs: "flex-start", md: "center" } }} />
              </Stack>
              <Alert severity="info" sx={{ mt: 2 }}>
                {selectedStep.detail}
              </Alert>
              <Box sx={{ mt: 2, display: "grid", gridTemplateColumns: { xs: "1fr", md: "repeat(4, 1fr)" }, gap: 1.5 }}>
                {explanation.metrics.map(([label, value]) => (
                  <Paper key={label} elevation={0} sx={{ p: 1.5, bgcolor: "rgba(15, 23, 42, 0.8)" }}>
                    <Typography variant="caption" color="text.secondary">
                      {label}
                    </Typography>
                    <Typography variant="subtitle2" sx={{ mt: 0.5, fontWeight: 900 }}>
                      {value}
                    </Typography>
                  </Paper>
                ))}
              </Box>
              <Divider sx={{ my: 2 }} />
              <Typography variant="h6">Logical agents inside the control plane</Typography>
              <Box sx={{ mt: 1, display: "grid", gridTemplateColumns: { xs: "1fr", md: "repeat(5, 1fr)" }, gap: 1 }}>
                {agentWork.map(([agent, output]) => (
                  <Paper key={agent} elevation={0} sx={{ p: 1.25, bgcolor: "rgba(15, 23, 42, 0.72)", minHeight: 116 }}>
                    <Typography variant="caption" sx={{ color: "primary.main", fontWeight: 900 }}>
                      {agent}
                    </Typography>
                    <Typography variant="caption" color="text.secondary" sx={{ display: "block", mt: 0.75, lineHeight: 1.45 }}>
                      {output}
                    </Typography>
                  </Paper>
                ))}
              </Box>
            </Paper>

            <Paper elevation={0} sx={{ p: 2, bgcolor: "rgba(2, 6, 23, 0.58)", border: "1px solid rgba(148, 163, 184, 0.16)" }}>
              <Stack direction="row" sx={{ justifyContent: "space-between", alignItems: "center" }}>
                <Typography variant="h6">Governed output</Typography>
                <Chip color={agentRuntime.provider === "openai" ? "success" : agentRuntime.provider ? "warning" : "default"} label={agentRuntime.provider ?? "waiting"} />
              </Stack>
              <Typography variant="caption" color="text.secondary">
                Actions proposed by the planner, then checked by policy. No physical action is real.
              </Typography>
              <Box sx={{ mt: 1.5, display: "grid", gap: 1.25 }}>
                {proposedActions.length ? (
                  proposedActions.map((proposal) => {
                    const policy = policyDecisions.find((item) => item.action === proposal.action);
                    return (
                      <Paper key={proposal.action} elevation={0} sx={{ p: 1.25, bgcolor: "rgba(15, 23, 42, 0.78)", border: "1px solid rgba(148, 163, 184, 0.14)" }}>
                        <Stack direction="row" sx={{ justifyContent: "space-between", gap: 1, alignItems: "center" }}>
                          <Typography variant="subtitle2" sx={{ fontWeight: 900 }}>
                            {formatAction(proposal.action)}
                          </Typography>
                          <Chip
                            size="small"
                            color={policy?.blocked ? "error" : policy?.requiresHumanApproval ? "warning" : "success"}
                            label={policy?.blocked ? "blocked" : policy?.requiresHumanApproval ? "approval" : "allowed"}
                          />
                        </Stack>
                        <Typography variant="caption" color="text.secondary" sx={{ display: "block", mt: 0.75 }}>
                          {policy?.reason ?? proposal.reason}
                        </Typography>
                      </Paper>
                    );
                  })
                ) : (
                  <Alert severity="info" icon={<AutoAwesomeIcon />}>
                    Press run. The proposed tool calls and guardrail results appear here.
                  </Alert>
                )}
              </Box>
              <Divider sx={{ my: 2 }} />
              <Metric label="Planner runtime" value={agentRuntime.runtime ?? "after run"} />
              <Metric label="Planner latency" value={agentRuntime.latencyMs ? `${(agentRuntime.latencyMs / 1000).toFixed(1)}s` : "after run"} />
              <Metric label="Approval gates" value={String(gatedPolicies.length)} />
              <Metric label="Audit record" value={decisionRecord?.runId ?? "after run"} />
            </Paper>
          </Box>
        </Surface>

        <Accordion sx={{ mt: 2, bgcolor: "rgba(15, 23, 42, 0.88)", border: "1px solid rgba(148, 163, 184, 0.18)" }}>
          <AccordionSummary expandIcon={<ExpandMoreIcon />}>
            <Stack direction="row" spacing={1} sx={{ alignItems: "center" }}>
              <MemoryIcon color="primary" />
              <Typography sx={{ fontWeight: 800 }}>Technical JSON</Typography>
            </Stack>
          </AccordionSummary>
          <AccordionDetails>
            <JsonInspector title="Current runtime payload" value={resultJson} />
          </AccordionDetails>
        </Accordion>
      </Box>
    </ThemeProvider>
  );
}
