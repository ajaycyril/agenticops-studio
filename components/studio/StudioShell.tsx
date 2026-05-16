"use client";

import { useEffect, useMemo, useState } from "react";
import type * as tf from "@tensorflow/tfjs";
import {
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
import AutoAwesomeIcon from "@mui/icons-material/AutoAwesome";
import CheckCircleIcon from "@mui/icons-material/CheckCircle";
import GppMaybeIcon from "@mui/icons-material/GppMaybe";
import PlayArrowIcon from "@mui/icons-material/PlayArrow";
import PsychologyIcon from "@mui/icons-material/Psychology";
import RuleIcon from "@mui/icons-material/Rule";
import SensorsIcon from "@mui/icons-material/Sensors";
import VisibilityIcon from "@mui/icons-material/Visibility";
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
import { clonePreset } from "@/lib/scenario-presets";
import { createTraceEvent } from "@/lib/trace/create-trace-event";
import { saveTrace } from "@/lib/trace/trace-store";
import type { AgenticResult, DecisionRecord, IncidentState, MLResult, PolicyDecision, RiskLevel, RuleResult, TraceEvent, VisionResult } from "@/lib/types";

const theme = createTheme({
  palette: {
    mode: "dark",
    background: { default: "#07111f", paper: "#0f172a" },
    primary: { main: "#22d3ee" },
    secondary: { main: "#60a5fa" },
    warning: { main: "#f59e0b" },
    error: { main: "#ef4444" },
    success: { main: "#22c55e" },
    text: { primary: "#f8fafc", secondary: "#a7b4c8" }
  },
  shape: { borderRadius: 8 },
  typography: {
    fontFamily: "Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif",
    h3: { fontWeight: 760, letterSpacing: 0 },
    h5: { fontWeight: 760, letterSpacing: 0 },
    h6: { fontWeight: 760, letterSpacing: 0 }
  },
  components: {
    MuiPaper: { styleOverrides: { root: { backgroundImage: "none" } } },
    MuiButton: { styleOverrides: { root: { textTransform: "none", fontWeight: 800 } } },
    MuiToggleButton: { styleOverrides: { root: { textTransform: "none", fontWeight: 800 } } }
  }
});

const sampleImageMap = {
  "fire-smoke-room": { file: "fire-smoke-room.jpg", label: "Fire/smoke room" },
  "unclear-camera": { file: "unclear-camera.jpg", label: "Unclear camera" },
  "cooking-smoke": { file: "cooking-smoke.jpg", label: "Cooking smoke" }
} as const;

type HealthStatus = {
  status: "ok";
  openaiConfigured: boolean;
  roboflowConfigured: boolean;
  openaiModel: string;
  openaiMaxOutputTokens: number;
  openaiTimeoutMs: number;
  openaiMaxAgentCallsPerRun: number;
};

type SafetyMode = "strict" | "balanced" | "rapid_response";
type PhaseId = "observe" | "predict" | "reason" | "guard" | "act";
type PhaseStatus = "waiting" | "running" | "done" | "fallback" | "blocked";
type Phase = {
  id: PhaseId;
  label: string;
  verb: string;
  status: PhaseStatus;
  headline: string;
  explanation: string;
  result: string;
};

const initialPhases: Phase[] = [
  {
    id: "observe",
    label: "Observe",
    verb: "Perceive",
    status: "waiting",
    headline: "Physical AI starts with real-world evidence.",
    explanation: "Sensors and camera signals create the incident context. This is more than text input: it is AI connected to physical signals.",
    result: "Waiting for run."
  },
  {
    id: "predict",
    label: "Predict",
    verb: "Model",
    status: "waiting",
    headline: "ML estimates risk, but does not decide what to do.",
    explanation: "The browser trains or uses a TensorFlow.js risk model. It outputs probability only.",
    result: "Waiting for run."
  },
  {
    id: "reason",
    label: "Reason",
    verb: "Plan",
    status: "waiting",
    headline: "Agentic AI turns evidence into a proposed plan.",
    explanation: "The planner reads sensors, vision, ML risk, SOP context, tool contracts, and safety posture. It proposes structured actions.",
    result: "Waiting for run."
  },
  {
    id: "guard",
    label: "Guard",
    verb: "Constrain",
    status: "waiting",
    headline: "Enterprise agentic AI does not let the model act alone.",
    explanation: "Policy checks block unsafe actions, require human approval, or allow sandbox execution.",
    result: "Waiting for run."
  },
  {
    id: "act",
    label: "Record",
    verb: "Audit",
    status: "waiting",
    headline: "Tools act only after governance. Every decision is recorded.",
    explanation: "Drone, gate, and authority tools are sandboxed. The platform writes a decision record so the run is auditable.",
    result: "Waiting for run."
  }
];

function pct(value: number) {
  return `${Math.round(value * 100)}%`;
}

function formatAction(action: string) {
  return action.replaceAll("_", " ");
}

function riskFromProbability(probability: number): RiskLevel {
  if (probability >= 0.82) return "critical";
  if (probability >= 0.62) return "high";
  if (probability >= 0.36) return "medium";
  return "low";
}

function phaseColor(status: PhaseStatus): "default" | "primary" | "success" | "warning" | "error" {
  if (status === "running") return "primary";
  if (status === "done") return "success";
  if (status === "fallback") return "warning";
  if (status === "blocked") return "error";
  return "default";
}

function sleep(ms: number) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function TeachingPanel({ title, body, accent }: { title: string; body: string; accent: string }) {
  return (
    <Paper elevation={0} sx={{ p: 2, bgcolor: "rgba(2, 6, 23, 0.54)", border: `1px solid ${accent}`, height: "100%" }}>
      <Typography variant="h6">{title}</Typography>
      <Typography variant="body2" color="text.secondary" sx={{ mt: 1, lineHeight: 1.55 }}>
        {body}
      </Typography>
    </Paper>
  );
}

export function StudioShell() {
  const [incident, setIncident] = useState<IncidentState>(() => clonePreset(1));
  const [sampleName, setSampleName] = useState<keyof typeof sampleImageMap>("fire-smoke-room");
  const [safetyMode, setSafetyMode] = useState<SafetyMode>("balanced");
  const [phases, setPhases] = useState<Phase[]>(initialPhases);
  const [activePhase, setActivePhase] = useState<PhaseId>("observe");
  const [running, setRunning] = useState(false);
  const [health, setHealth] = useState<HealthStatus>();
  const [message, setMessage] = useState("Press Run. The story will explain what each layer did.");
  const [model, setModel] = useState<tf.LayersModel>();
  const [modelVersion, setModelVersion] = useState(() => loadModelVersion() ?? DEFAULT_MODEL_VERSION);
  const [modelMetrics, setModelMetrics] = useState<MLResult["metrics"]>();
  const [ruleResult, setRuleResult] = useState<RuleResult>(() => evaluateRules(clonePreset(1)));
  const [mlResult, setMlResult] = useState<MLResult>(() => heuristicRiskPrediction(clonePreset(1)));
  const [visionResult, setVisionResult] = useState<VisionResult>();
  const [agenticResult, setAgenticResult] = useState<AgenticResult>();
  const [policyDecisions, setPolicyDecisions] = useState<PolicyDecision[]>([]);
  const [decisionRecord, setDecisionRecord] = useState<DecisionRecord>();
  const [trace, setTrace] = useState<TraceEvent[]>([
    createTraceEvent({ type: "incident_created", actor: "system", status: "success", output: clonePreset(1) })
  ]);
  const [runtime, setRuntime] = useState<{ provider?: string; runtime?: string; latencyMs?: number; message?: string }>({});

  const tools = useMemo(() => getToolRegistry(), []);
  const selectedPhase = phases.find((phase) => phase.id === activePhase) ?? phases[0];
  const blockedPolicies = policyDecisions.filter((policy) => policy.blocked);
  const gatedPolicies = policyDecisions.filter((policy) => policy.requiresHumanApproval && !policy.blocked);

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

  function updatePhase(id: PhaseId, status: PhaseStatus, result: string, explanation?: string) {
    setActivePhase(id);
    setPhases((current) =>
      current.map((phase) =>
        phase.id === id ? { ...phase, status, result, explanation: explanation ?? phase.explanation } : phase
      )
    );
  }

  function loadDemo(kind: keyof typeof sampleImageMap) {
    const presetIndex = kind === "cooking-smoke" ? 0 : kind === "unclear-camera" ? 2 : 1;
    const next = clonePreset(presetIndex);
    setIncident(next);
    setSampleName(kind);
    setRuleResult(evaluateRules(next));
    setMlResult(heuristicRiskPrediction(next));
    setVisionResult(undefined);
    setAgenticResult(undefined);
    setPolicyDecisions([]);
    setDecisionRecord(undefined);
    setRuntime({});
    setPhases(initialPhases);
    setActivePhase("observe");
    setMessage("Press Run. The story will explain what each layer did.");
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

  async function runPrediction(incidentForRun: IncidentState, modelOverride?: tf.LayersModel, versionOverride = modelVersion, metricsOverride = modelMetrics) {
    const features = incidentToFeatures(incidentForRun);
    let probability: number;
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
      explanation: "ML predicts probability only. It does not select or execute tools."
    };
    setMlResult(result);
    appendTrace({ type: "ml_model_predicted", actor: "ml_model", input: incidentForRun, output: result, status: "success" });
    return result;
  }

  async function runDemo() {
    setRunning(true);
    setMessage("Running live. Watch the sentence in the center: it explains the agentic handoff.");
    setPhases(initialPhases);
    setPolicyDecisions([]);
    setAgenticResult(undefined);
    setDecisionRecord(undefined);
    setRuntime({});

    try {
      updatePhase("observe", "running", "Reading physical signals and camera evidence...");
      await sleep(500);
      const rule = evaluateRules(incident);
      setRuleResult(rule);
      appendTrace({ type: "rule_engine_evaluated", actor: "system", input: incident, output: rule, status: "success" });
      const vision = await runVision(incident);
      updatePhase(
        "observe",
        vision.result.provider === "sample" ? "fallback" : "done",
        `Signals say ${rule.severity}; vision says fire ${pct(vision.result.maxFireConfidence)}, smoke ${pct(vision.result.maxSmokeConfidence)}.`,
        "This is Physical AI: the workflow starts from real-world signals, not a chat prompt."
      );

      updatePhase("predict", "running", "Training/scoring a browser ML risk model...");
      await sleep(450);
      const trained = model
        ? undefined
        : await trainRiskModel({ size: 220, epochs: 18, falseAlarmBias: vision.nextIncident.historicalFalseAlarmRate, learningRate: 0.08 });
      if (trained) {
        setModel(trained.model);
        setModelMetrics(trained.metrics);
        setModelVersion(trained.modelVersion);
        saveModelVersion(trained.modelVersion);
      }
      const ml = await runPrediction(vision.nextIncident, trained?.model ?? model, trained?.modelVersion ?? modelVersion, trained?.metrics ?? modelMetrics);
      updatePhase(
        "predict",
        "done",
        `ML predicts ${pct(ml.fireProbability)} fire probability (${ml.riskLevel}).`,
        "This is not yet agentic. ML gives a probability; it does not decide which tools to call."
      );

      updatePhase("reason", "running", `${health?.openaiConfigured ? "Calling OpenAI Agents SDK" : "Using fallback planner"} to propose a plan...`);
      await sleep(350);
      appendTrace({ type: "agent_called", actor: "llm_agent", input: { incidentId: vision.nextIncident.incidentId }, status: "pending" });
      const started = Date.now();
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
            operatingMode: safetyMode === "rapid_response" ? "rapid_response" : safetyMode === "strict" ? "conservative" : "balanced",
            authorityPosture: safetyMode === "strict" ? "strict" : safetyMode === "rapid_response" ? "approval_gated" : "critical_only",
            operatorInstruction:
              "Be explicit: explain evidence, proposed tool calls, policy checks, human approval gates, sandboxed execution, and audit."
          }
        })
      });
      const data = await response.json();
      if (!data.ok) throw new Error(data.error.message);
      const agent = data.result as AgenticResult;
      const latencyMs = Date.now() - started;
      setRuntime({ provider: data.provider, runtime: data.runtime, latencyMs, message: data.message });
      setAgenticResult(agent);
      updatePhase(
        "reason",
        data.provider === "openai" ? "done" : "fallback",
        `Agent proposed ${agent.proposedActions.length} tool calls using ${data.runtime}.`,
        "This is the agentic part: the planner converts evidence and a goal into a structured action plan."
      );

      updatePhase("guard", "running", "Checking whether the proposed tool calls are allowed...");
      await sleep(450);
      const policies = evaluatePoliciesForActions(agent.proposedActions.map((proposal) => proposal.action), vision.nextIncident, ml);
      setPolicyDecisions(policies);
      updatePhase(
        "guard",
        policies.some((policy) => policy.blocked) ? "blocked" : "done",
        `${policies.filter((policy) => policy.blocked).length} blocked, ${policies.filter((policy) => policy.requiresHumanApproval && !policy.blocked).length} require approval.`,
        "This is enterprise agentic AI: the model reasons, but policy constrains and humans approve risky execution."
      );

      updatePhase("act", "running", "Writing decision record. Physical actions remain sandboxed...");
      await sleep(450);
      const validatedEvent = appendTrace({ type: "agent_output_validated", actor: "llm_agent", output: agent, status: "success", explanation: data.message });
      const policyEvent = appendTrace({ type: "policy_checked", actor: "policy", output: policies, status: "success" });
      const record = buildDecisionRecord({
        runId: `RUN-${Date.now()}`,
        incident: vision.nextIncident,
        ruleResult: rule,
        mlResult: ml,
        visionResult: vision.result,
        agenticResult: agent,
        policyDecisions: policies,
        trace: [...trace, validatedEvent, policyEvent]
      });
      setDecisionRecord(record);
      sessionStorage.setItem("agenticops.latestDecisionRecord", JSON.stringify(record));
      appendTrace({ type: "decision_record_written", actor: "system", output: { runId: record.runId }, status: "success" });
      updatePhase(
        "act",
        "done",
        `Decision record ${record.runId} written. Drone/gate/authority actions are sandboxed.`,
        "This is production readiness: every proposed action, approval gate, model result, and policy decision is auditable."
      );
      setMessage("Done. The center panel explains the agentic handoff; the right panel shows the governed tool plan.");
    } catch (error) {
      const text = error instanceof Error ? error.message : "Run failed.";
      setMessage(text);
      setPhases((current) => current.map((phase) => (phase.status === "running" ? { ...phase, status: "blocked", result: text } : phase)));
    } finally {
      setRunning(false);
    }
  }

  const actionRows = agenticResult?.proposedActions ?? [];
  const ruleVsMlVsAgent = [
    {
      title: "Rule-based",
      icon: <RuleIcon />,
      sentence: `${formatAction(ruleResult.action)} from smoke/heat thresholds.`,
      limitation: "Detects only."
    },
    {
      title: "ML-based",
      icon: <PsychologyIcon />,
      sentence: `${pct(mlResult.fireProbability)} fire probability.`,
      limitation: "Predicts only."
    },
    {
      title: "Agentic",
      icon: <AutoAwesomeIcon />,
      sentence: agenticResult ? `${agenticResult.proposedActions.length} proposed tool calls, ${policyDecisions.length} policy checks.` : "Plans after run.",
      limitation: "Coordinates under governance."
    }
  ];

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Box sx={{ mx: "auto", maxWidth: 1440, px: { xs: 0, md: 1 } }}>
        <Paper
          elevation={0}
          sx={{
            minHeight: { lg: "calc(100vh - 96px)" },
            p: { xs: 2, md: 3 },
            bgcolor: "rgba(15, 23, 42, 0.92)",
            border: "1px solid rgba(148, 163, 184, 0.18)"
          }}
        >
          <Box sx={{ display: "grid", gridTemplateColumns: { xs: "1fr", lg: "1fr auto" }, gap: 2, alignItems: "center" }}>
            <Box>
              <Stack direction="row" spacing={1} sx={{ flexWrap: "wrap", gap: 1 }}>
                <Chip color="primary" label="One clear demo" />
                <Chip color={health?.openaiConfigured ? "success" : "warning"} label={`OpenAI ${health?.openaiConfigured ? health.openaiModel : "fallback"}`} />
                <Chip color={health?.roboflowConfigured ? "success" : "warning"} label={`Vision ${health?.roboflowConfigured ? "Roboflow" : "sample"}`} />
                <Chip color="warning" label="Physical actions sandboxed" />
              </Stack>
              <Typography variant="h3" sx={{ mt: 1.5 }}>
                Agentic AI is not the action. It is the governed decision loop.
              </Typography>
              <Typography variant="body2" color="text.secondary" sx={{ mt: 0.75, maxWidth: 880 }}>
                Watch the same fire incident move from evidence to prediction to agent plan to guardrail checks to audit.
              </Typography>
            </Box>
            <Button size="large" variant="contained" startIcon={<PlayArrowIcon />} onClick={() => void runDemo()} disabled={running} sx={{ minWidth: 260, py: 1.45 }}>
              {running ? "Running..." : "Run the agentic demo"}
            </Button>
          </Box>

          {running ? <LinearProgress sx={{ mt: 2 }} /> : null}
          <Alert severity={message.startsWith("Done") ? "success" : message.includes("failed") ? "error" : "info"} sx={{ mt: 2 }}>
            {message}
          </Alert>

          <Box sx={{ mt: 2, display: "grid", gridTemplateColumns: { xs: "1fr", lg: "280px 1fr 380px" }, gap: 2 }}>
            <Paper elevation={0} sx={{ p: 2, bgcolor: "rgba(2, 6, 23, 0.55)", border: "1px solid rgba(148, 163, 184, 0.16)" }}>
              <Typography variant="h6">Inputs</Typography>
              <Typography variant="caption" color="text.secondary">
                No fiddly controls. Pick the story and safety posture.
              </Typography>
              <ToggleButtonGroup
                exclusive
                fullWidth
                value={sampleName}
                onChange={(_, value) => value && loadDemo(value)}
                sx={{ mt: 1.5, display: "grid", gap: 1 }}
              >
                {(Object.keys(sampleImageMap) as (keyof typeof sampleImageMap)[]).map((name) => (
                  <ToggleButton key={name} value={name}>{sampleImageMap[name].label}</ToggleButton>
                ))}
              </ToggleButtonGroup>
              <Divider sx={{ my: 2 }} />
              <Typography variant="h6">Guardrail mode</Typography>
              <ToggleButtonGroup
                exclusive
                fullWidth
                value={safetyMode}
                onChange={(_, value) => value && setSafetyMode(value)}
                sx={{ mt: 1, display: "grid", gap: 1 }}
              >
                <ToggleButton value="strict">Strict</ToggleButton>
                <ToggleButton value="balanced">Balanced</ToggleButton>
                <ToggleButton value="rapid_response">Rapid response</ToggleButton>
              </ToggleButtonGroup>
              <Divider sx={{ my: 2 }} />
              <Typography variant="caption" color="text.secondary">Physical scene</Typography>
              <Typography variant="body2" sx={{ mt: 0.75 }}>Smoke {incident.smokePpm} ppm</Typography>
              <Typography variant="body2">Heat {incident.temperatureC} C</Typography>
              <Typography variant="body2">Camera frame: {sampleImageMap[sampleName].label}</Typography>
            </Paper>

            <Box sx={{ display: "grid", gap: 2 }}>
              <Box sx={{ display: "grid", gridTemplateColumns: { xs: "1fr", md: "repeat(5, 1fr)" }, gap: 1 }}>
                {phases.map((phase) => (
                  <Paper
                    key={phase.id}
                    component="button"
                    onClick={() => setActivePhase(phase.id)}
                    elevation={0}
                    sx={{
                      p: 1.4,
                      textAlign: "left",
                      minHeight: 128,
                      cursor: "pointer",
                      color: "text.primary",
                      bgcolor: activePhase === phase.id ? "rgba(34, 211, 238, 0.15)" : "rgba(2, 6, 23, 0.55)",
                      border: activePhase === phase.id ? "1px solid rgba(34, 211, 238, 0.65)" : "1px solid rgba(148, 163, 184, 0.16)"
                    }}
                  >
                    <Stack direction="row" sx={{ alignItems: "center", justifyContent: "space-between" }}>
                      <Typography variant="caption" color="text.secondary">{phase.verb}</Typography>
                      <Chip size="small" color={phaseColor(phase.status)} label={phase.status} />
                    </Stack>
                    <Typography variant="h6" sx={{ mt: 1 }}>{phase.label}</Typography>
                    <Typography variant="caption" color="text.secondary">{phase.result}</Typography>
                  </Paper>
                ))}
              </Box>

              <Paper elevation={0} sx={{ p: 2.5, bgcolor: "rgba(2, 6, 23, 0.56)", border: "1px solid rgba(34, 211, 238, 0.2)", minHeight: 270 }}>
                <Stack direction="row" spacing={1.2} sx={{ alignItems: "center" }}>
                  <Box sx={{ color: "primary.main" }}>
                    {selectedPhase.id === "observe" ? <SensorsIcon /> : selectedPhase.id === "predict" ? <PsychologyIcon /> : selectedPhase.id === "reason" ? <AutoAwesomeIcon /> : selectedPhase.id === "guard" ? <GppMaybeIcon /> : <CheckCircleIcon />}
                  </Box>
                  <Typography variant="h5">{selectedPhase.headline}</Typography>
                </Stack>
                <Typography variant="body1" sx={{ mt: 2, lineHeight: 1.65 }}>
                  {selectedPhase.explanation}
                </Typography>
                <Alert severity={selectedPhase.status === "blocked" ? "warning" : "info"} sx={{ mt: 2 }}>
                  {selectedPhase.result}
                </Alert>
                {runtime.message && selectedPhase.id === "reason" ? (
                  <Alert severity={runtime.provider === "openai" ? "success" : "warning"} sx={{ mt: 1 }}>
                    {runtime.message}
                  </Alert>
                ) : null}
              </Paper>

              <Box sx={{ display: "grid", gridTemplateColumns: { xs: "1fr", md: "repeat(3, 1fr)" }, gap: 1 }}>
                {ruleVsMlVsAgent.map((item) => (
                  <TeachingPanel
                    key={item.title}
                    title={item.title}
                    body={`${item.sentence} ${item.limitation}`}
                    accent={item.title === "Agentic" ? "rgba(34, 211, 238, 0.35)" : "rgba(148, 163, 184, 0.18)"}
                  />
                ))}
              </Box>
            </Box>

            <Paper elevation={0} sx={{ p: 2, bgcolor: "rgba(2, 6, 23, 0.55)", border: "1px solid rgba(148, 163, 184, 0.16)" }}>
              <Typography variant="h6">Governed tool plan</Typography>
              <Typography variant="caption" color="text.secondary">
                This is where agentic AI becomes enterprise AI: proposed tools are checked before action.
              </Typography>
              <Box sx={{ mt: 1.5, display: "grid", gap: 1 }}>
                {actionRows.length ? (
                  actionRows.map((proposal) => {
                    const policy = policyDecisions.find((item) => item.action === proposal.action);
                    return (
                      <Paper key={proposal.action} elevation={0} sx={{ p: 1.25, bgcolor: "rgba(15, 23, 42, 0.72)", border: "1px solid rgba(148, 163, 184, 0.14)" }}>
                        <Stack direction="row" sx={{ justifyContent: "space-between", gap: 1 }}>
                          <Typography variant="subtitle2" sx={{ fontWeight: 900 }}>{formatAction(proposal.action)}</Typography>
                          <Chip size="small" color={policy?.blocked ? "error" : policy?.requiresHumanApproval ? "warning" : "success"} label={policy?.blocked ? "blocked" : policy?.requiresHumanApproval ? "approval" : "allowed"} />
                        </Stack>
                        <Typography variant="caption" color="text.secondary" sx={{ display: "block", mt: 0.75 }}>
                          {policy?.reason ?? proposal.reason}
                        </Typography>
                      </Paper>
                    );
                  })
                ) : (
                  <Alert severity="info" icon={<VisibilityIcon />}>Run the demo to see proposed tool calls and guardrail results.</Alert>
                )}
              </Box>
              <Divider sx={{ my: 2 }} />
              <Typography variant="caption" color="text.secondary">Runtime proof</Typography>
              <Typography variant="body2" sx={{ mt: 0.75 }}>Planner: {runtime.provider ?? "after run"}</Typography>
              <Typography variant="body2">Vision: {visionResult ? `${visionResult.provider} fire ${pct(visionResult.maxFireConfidence)}` : "after run"}</Typography>
              <Typography variant="body2">Latency: {runtime.latencyMs ? `${(runtime.latencyMs / 1000).toFixed(1)}s` : "after run"}</Typography>
              <Typography variant="body2">Blocked: {blockedPolicies.length}</Typography>
              <Typography variant="body2">Approval-gated: {gatedPolicies.length}</Typography>
              <Typography variant="body2">Audit: {decisionRecord?.runId ?? "after run"}</Typography>
            </Paper>
          </Box>
        </Paper>
      </Box>
    </ThemeProvider>
  );
}
