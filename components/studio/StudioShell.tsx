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
  Slider,
  Stack,
  Switch,
  TextField,
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
import LocalFireDepartmentIcon from "@mui/icons-material/LocalFireDepartment";
import MemoryIcon from "@mui/icons-material/Memory";
import ModelTrainingIcon from "@mui/icons-material/ModelTraining";
import PolicyIcon from "@mui/icons-material/Policy";
import SensorsIcon from "@mui/icons-material/Sensors";
import ShieldIcon from "@mui/icons-material/Shield";
import TimelineIcon from "@mui/icons-material/Timeline";
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
    background: {
      default: "#08111f",
      paper: "#101827"
    },
    primary: {
      main: "#22d3ee"
    },
    secondary: {
      main: "#60a5fa"
    },
    warning: {
      main: "#f59e0b"
    },
    error: {
      main: "#ef4444"
    },
    success: {
      main: "#22c55e"
    },
    text: {
      primary: "#eef6ff",
      secondary: "#9fb1c8"
    }
  },
  shape: {
    borderRadius: 10
  },
  typography: {
    fontFamily: "Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif",
    h3: {
      letterSpacing: 0,
      fontWeight: 720
    },
    h5: {
      letterSpacing: 0,
      fontWeight: 700
    },
    h6: {
      letterSpacing: 0,
      fontWeight: 700
    }
  },
  components: {
    MuiPaper: {
      styleOverrides: {
        root: {
          backgroundImage: "none"
        }
      }
    },
    MuiButton: {
      styleOverrides: {
        root: {
          textTransform: "none",
          fontWeight: 700
        }
      }
    },
    MuiChip: {
      styleOverrides: {
        root: {
          fontWeight: 700
        }
      }
    }
  }
});

const sampleImageMap = {
  "cooking-smoke": { file: "cooking-smoke.jpg", label: "Cooking smoke" },
  "fire-smoke-room": { file: "fire-smoke-room.jpg", label: "Confirmed fire" },
  "unclear-camera": { file: "unclear-camera.jpg", label: "Unclear camera" }
} as const;

const visiblePresets = [1, 2, 4];

const physicalControls: {
  key: "smokePpm" | "temperatureC" | "cameraFireConfidence";
  label: string;
  min: number;
  max: number;
  step: number;
  suffix: string;
}[] = [
  { key: "smokePpm", label: "Smoke sensor", min: 0, max: 140, step: 1, suffix: " ppm" },
  { key: "temperatureC", label: "Heat sensor", min: 10, max: 80, step: 1, suffix: " C" },
  { key: "cameraFireConfidence", label: "Camera fire signal", min: 0, max: 1, step: 0.01, suffix: "" }
];

type HealthStatus = {
  status: "ok";
  openaiConfigured: boolean;
  roboflowConfigured: boolean;
  openaiModel: string;
  openaiMaxAgentCallsPerRun: number;
};

type AgentControls = {
  operatingMode: "balanced" | "conservative" | "rapid_response";
  authorityPosture: "strict" | "approval_gated" | "critical_only";
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

function signalValue(incident: IncidentState, key: (typeof physicalControls)[number]["key"]) {
  const value = incident[key];
  return key === "cameraFireConfidence" ? pct(value) : `${value}${physicalControls.find((item) => item.key === key)?.suffix ?? ""}`;
}

function Surface({ children, sx }: { children: React.ReactNode; sx?: object }) {
  return (
    <Paper
      elevation={0}
      sx={{
        border: "1px solid rgba(148, 163, 184, 0.18)",
        bgcolor: "rgba(15, 23, 42, 0.88)",
        p: { xs: 2, md: 3 },
        ...sx
      }}
    >
      {children}
    </Paper>
  );
}

function StatusChip({ label, value, color = "default" }: { label: string; value: string; color?: "default" | "primary" | "success" | "warning" | "error" | "info" }) {
  return (
    <Chip
      color={color}
      variant={color === "default" ? "outlined" : "filled"}
      label={`${label}: ${value}`}
      sx={{ justifyContent: "space-between", maxWidth: "100%" }}
    />
  );
}

function EvidenceLine({ label, value }: { label: string; value: string }) {
  return (
    <Box sx={{ display: "flex", justifyContent: "space-between", gap: 2, py: 0.75 }}>
      <Typography variant="body2" color="text.secondary">
        {label}
      </Typography>
      <Typography variant="body2" sx={{ color: "text.primary", textAlign: "right", fontWeight: 700 }}>
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
    createTraceEvent({
      type: "incident_created",
      actor: "system",
      status: "success",
      output: clonePreset(1),
      explanation: "Loaded confirmed fire scenario."
    })
  ]);
  const [health, setHealth] = useState<HealthStatus | undefined>();
  const [message, setMessage] = useState<string>();
  const [running, setRunning] = useState(false);
  const [training, setTraining] = useState(false);
  const [model, setModel] = useState<tf.LayersModel>();
  const [modelVersion, setModelVersion] = useState(() => loadModelVersion() ?? DEFAULT_MODEL_VERSION);
  const [modelMetrics, setModelMetrics] = useState<MLResult["metrics"]>();
  const [sampleName, setSampleName] = useState<keyof typeof sampleImageMap>("fire-smoke-room");
  const [hasRun, setHasRun] = useState(false);
  const [agentControls, setAgentControls] = useState<AgentControls>({
    operatingMode: "balanced",
    authorityPosture: "critical_only"
  });
  const [plannerNote, setPlannerNote] = useState("Prioritize life safety. Require approvals before any physical-world action.");

  const tools = useMemo(() => getToolRegistry(), []);

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

  function resetRunState() {
    setVisionResult(undefined);
    setAgenticResult(undefined);
    setPolicyDecisions([]);
    setDecisionRecord(undefined);
    setHasRun(false);
  }

  function loadScenario(index: number) {
    const next = clonePreset(index);
    setIncident(next);
    setRuleResult(evaluateRules(next));
    setMlResult({ ...heuristicRiskPrediction(next), modelVersion, metrics: modelMetrics });
    resetRunState();
    setMessage(undefined);
    appendTrace({ type: "incident_created", actor: "system", status: "success", output: next, explanation: `Loaded ${next.scenarioName}.` });
  }

  function updateIncident<K extends keyof IncidentState>(key: K, value: IncidentState[K]) {
    setIncident((current) => {
      const next = { ...current, [key]: value };
      setRuleResult(evaluateRules(next));
      setMlResult({ ...heuristicRiskPrediction(next), modelVersion, metrics: modelMetrics });
      resetRunState();
      return next;
    });
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
      const trained = await trainRiskModel({
        size: 220,
        epochs: 18,
        falseAlarmBias: incidentForRun.historicalFalseAlarmRate,
        learningRate: 0.08
      });
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

  async function runPrediction(
    incidentForRun: IncidentState,
    modelOverride?: tf.LayersModel,
    versionOverride = modelVersion,
    metricsOverride = modelMetrics
  ) {
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
      explanation: "ML predicts fire probability from fused signals. It does not dispatch drones, unlock gates, or notify authorities."
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
      reader.onloadend = () => {
        if (typeof reader.result === "string") resolve(reader.result);
        else reject(new Error("Invalid sample image payload"));
      };
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
    setMessage("Running the governed full-stack incident workflow...");
    try {
      const incidentForRun = incident;
      const rule = runRules(incidentForRun);
      const trained = model ? undefined : await trainBrowserModel(incidentForRun);
      const vision = await runVision(incidentForRun);
      const ml = await runPrediction(
        vision.nextIncident,
        trained?.model ?? model,
        trained?.modelVersion ?? modelVersion,
        trained?.metrics ?? modelMetrics
      );

      appendTrace({ type: "agent_called", actor: "llm_agent", input: { incidentId: vision.nextIncident.incidentId }, status: "pending" });
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
            operatorInstruction: plannerNote
          }
        })
      });
      const data = await response.json();
      if (!data.ok) throw new Error(data.error.message);

      const result = data.result as AgenticResult;
      const policies = evaluatePoliciesForActions(result.proposedActions.map((proposal) => proposal.action), vision.nextIncident, ml);
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
      setHasRun(true);
      sessionStorage.setItem("agenticops.latestDecisionRecord", JSON.stringify(record));
      appendTrace({ type: "decision_record_written", actor: "system", output: { runId: record.runId }, status: "success" });
      setMessage("Run complete. The lanes below show what was real, what was sandboxed, and what the agent proposed.");
    } catch (error) {
      const text = error instanceof Error ? error.message : "Run failed.";
      setMessage(text);
      appendTrace({ type: "error", actor: "system", output: { message: text }, status: "failed" });
    } finally {
      setRunning(false);
    }
  }

  const topFeature = mlResult.featureImportance[0];
  const blockedPolicies = policyDecisions.filter((policy) => policy.blocked);
  const gatedPolicies = policyDecisions.filter((policy) => policy.requiresHumanApproval && !policy.blocked);
  const physicalActions = agenticResult?.proposedActions.filter((proposal) =>
    ["dispatch_drone", "unlock_gate", "notify_authority"].includes(proposal.action)
  );
  const resultJson = { incident, ruleResult, mlResult, visionResult, agenticResult, policyDecisions, decisionRecord };

  const realBoundary = [
    { label: "OpenAI planner", value: health?.openaiConfigured ? `real API (${health.openaiModel})` : "fallback planner", color: health?.openaiConfigured ? "success" : "warning" },
    { label: "Roboflow vision", value: health?.roboflowConfigured ? "real hosted inference" : "sample inference", color: health?.roboflowConfigured ? "success" : "warning" },
    { label: "TensorFlow.js", value: model || hasRun ? "real browser-trained model" : "trains on first run", color: "info" },
    { label: "Policy engine", value: "real TypeScript evaluator", color: "success" },
    { label: "Drone/gate/authority", value: "sandbox only", color: "warning" }
  ] as const;

  const postureCopy = {
    conservative: "asks for stronger evidence and more approvals",
    balanced: "uses normal enterprise thresholds",
    rapid_response: "proposes faster reconnaissance when risk rises"
  } satisfies Record<AgentControls["operatingMode"], string>;

  const authorityCopy = {
    strict: "never proposes authority notification from the planner",
    approval_gated: "requires human approval for authority notification",
    critical_only: "allows authority notification only at critical risk"
  } satisfies Record<AgentControls["authorityPosture"], string>;

  const flowSteps = [
    {
      label: "Physical signals",
      icon: <SensorsIcon />,
      value: `${incident.smokePpm} ppm / ${incident.temperatureC} C`,
      status: "real state"
    },
    {
      label: "Edge vision",
      icon: <CameraAltIcon />,
      value: visionResult ? `fire ${pct(visionResult.maxFireConfidence)}` : sampleImageMap[sampleName].label,
      status: visionResult ? visionResult.provider : "ready"
    },
    {
      label: "ML risk",
      icon: <ModelTrainingIcon />,
      value: `${pct(mlResult.fireProbability)} ${mlResult.riskLevel}`,
      status: model || hasRun ? "tf.js" : "baseline"
    },
    {
      label: "Agent plan",
      icon: <AutoAwesomeIcon />,
      value: agenticResult ? `${agenticResult.proposedActions.length} actions` : "not run",
      status: health?.openaiConfigured ? "OpenAI/fallback" : "fallback"
    },
    {
      label: "Guardrails",
      icon: <PolicyIcon />,
      value: policyDecisions.length ? `${blockedPolicies.length} blocked / ${gatedPolicies.length} gated` : "pending",
      status: "real policy"
    },
    {
      label: "Audit",
      icon: <FactCheckIcon />,
      value: decisionRecord ? decisionRecord.runId : "pending",
      status: "local record"
    }
  ];

  const runtimeLanes = [
    {
      title: "Physical AI",
      icon: <SensorsIcon />,
      status: "Real state model",
      summary: "Smoke, heat, camera confidence, occupancy, drone availability, and access-control state become incident context.",
      lines: [
        ["Smoke", `${incident.smokePpm} ppm`],
        ["Temperature", `${incident.temperatureC} C`],
        ["Drone", incident.droneAvailable ? "available" : "unavailable"],
        ["Gate", incident.gateLocked ? "locked" : "already unlocked"]
      ]
    },
    {
      title: "Edge / Vision AI",
      icon: <CameraAltIcon />,
      status: visionResult ? `${visionResult.provider} result` : "Ready",
      summary: "The selected camera frame is sent to the server route. Keys stay server-side; fallback mode is labeled.",
      lines: [
        ["Frame", sampleImageMap[sampleName].label],
        ["Fire confidence", visionResult ? pct(visionResult.maxFireConfidence) : pct(incident.cameraFireConfidence)],
        ["Smoke confidence", visionResult ? pct(visionResult.maxSmokeConfidence) : pct(incident.cameraSmokeConfidence)],
        ["Latency", visionResult ? `${visionResult.latencyMs} ms` : "after run"]
      ]
    },
    {
      title: "MLOps Risk Model",
      icon: <ModelTrainingIcon />,
      status: mlResult.modelVersion.includes("tfjs") ? "Browser model" : "Baseline",
      summary: "A TensorFlow.js logistic model trains in-browser, versions the active model, and predicts risk without executing actions.",
      lines: [
        ["Fire probability", pct(mlResult.fireProbability)],
        ["Risk", mlResult.riskLevel],
        ["Top feature", topFeature ? topFeature.feature : "not available"],
        ["Accuracy", mlResult.metrics ? pct(mlResult.metrics.accuracy) : "after training"]
      ]
    },
    {
      title: "Guardrails",
      icon: <PolicyIcon />,
      status: policyDecisions.length ? "Evaluated" : "Pending",
      summary: "Schema validation, policy checks, tool permissions, human approval gates, and physical-safety rules sit between reasoning and action.",
      lines: [
        ["Policy checks", String(policyDecisions.length)],
        ["Blocked", String(blockedPolicies.length)],
        ["Approval needed", String(gatedPolicies.length)],
        ["Alarm suppression", "never allowed"]
      ]
    },
    {
      title: "Agentic Orchestration",
      icon: <AutoAwesomeIcon />,
      status: agenticResult ? "Plan generated" : "Not run",
      summary: "The agent reasons over evidence and proposes structured actions. It is not the control system.",
      lines: [
        ["Risk assessment", agenticResult?.riskAssessment.level ?? "after run"],
        ["Actions proposed", agenticResult ? String(agenticResult.proposedActions.length) : "0"],
        ["Physical actions", physicalActions?.length ? String(physicalActions.length) : "0"],
        ["Decision record", decisionRecord ? decisionRecord.runId : "after run"]
      ]
    }
  ];

  return (
    <ThemeProvider theme={materialTheme}>
      <CssBaseline />
      <Box sx={{ mx: "auto", maxWidth: 1440, px: { xs: 0, md: 1 }, pb: 4 }}>
        <Surface sx={{ overflow: "hidden", position: "relative" }}>
          <Box sx={{ display: "grid", gridTemplateColumns: { xs: "1fr", lg: "1.15fr .85fr" }, gap: 3, alignItems: "stretch" }}>
            <Box>
              <Stack direction="row" spacing={1} sx={{ flexWrap: "wrap", gap: 1 }}>
                <Chip icon={<ShieldIcon />} color="primary" label="Governed agentic AI" />
                <Chip variant="outlined" label="Real APIs where configured" />
                <Chip variant="outlined" label="Sandbox physical actions" />
              </Stack>
              <Typography variant="h3" sx={{ mt: 3, maxWidth: 820 }}>
                Tune the control plane. Run the incident. Watch governance change the outcome.
              </Typography>
              <Typography variant="body1" color="text.secondary" sx={{ mt: 2, maxWidth: 760, fontSize: 17, lineHeight: 1.55 }}>
                The useful part is the governed handoff: models reason, tools are constrained, humans approve risky actions, and every decision is recorded.
              </Typography>
              <Box sx={{ mt: 3, display: "grid", gridTemplateColumns: { xs: "1fr", sm: "repeat(3, 1fr)" }, gap: 1.5 }}>
                <Paper elevation={0} sx={{ p: 2, bgcolor: "rgba(245, 158, 11, 0.12)", border: "1px solid rgba(245, 158, 11, 0.25)" }}>
                  <Typography variant="overline" color="warning.main">
                    Rule-based
                  </Typography>
                  <Typography variant="h6">Detects</Typography>
                  <Typography variant="body2" color="text.secondary">
                    Fixed if/else alarm logic. No context fusion.
                  </Typography>
                </Paper>
                <Paper elevation={0} sx={{ p: 2, bgcolor: "rgba(96, 165, 250, 0.12)", border: "1px solid rgba(96, 165, 250, 0.25)" }}>
                  <Typography variant="overline" color="secondary.main">
                    ML-based
                  </Typography>
                  <Typography variant="h6">Predicts</Typography>
                  <Typography variant="body2" color="text.secondary">
                    Learns risk probability. No tool execution.
                  </Typography>
                </Paper>
                <Paper elevation={0} sx={{ p: 2, bgcolor: "rgba(34, 211, 238, 0.12)", border: "1px solid rgba(34, 211, 238, 0.25)" }}>
                  <Typography variant="overline" color="primary.main">
                    Agentic
                  </Typography>
                  <Typography variant="h6">Governs</Typography>
                  <Typography variant="body2" color="text.secondary">
                    Coordinates evidence, tools, policy, approval, and audit.
                  </Typography>
                </Paper>
              </Box>
            </Box>

            <Paper elevation={0} sx={{ p: 2.5, bgcolor: "rgba(2, 6, 23, 0.65)", border: "1px solid rgba(34, 211, 238, 0.22)" }}>
              <Typography variant="h6">Real vs sandbox boundary</Typography>
              <Stack spacing={1.2} sx={{ mt: 2 }}>
                {realBoundary.map((item) => (
                  <StatusChip key={item.label} label={item.label} value={item.value} color={item.color} />
                ))}
              </Stack>
              <Alert severity="info" sx={{ mt: 2 }}>
                The LLM proposes structured JSON. Policy and approval gates decide which sandbox tools may run. No real drone, gate, or authority API is called.
              </Alert>
            </Paper>
          </Box>
        </Surface>

        <Box sx={{ mt: 3, display: "grid", gridTemplateColumns: { xs: "1fr", xl: "420px 1fr" }, gap: 3 }}>
          <Surface>
            <Stack direction="row" spacing={1.2} sx={{ alignItems: "center" }}>
              <LocalFireDepartmentIcon color="warning" />
              <Box>
                <Typography variant="h5">1. Set the physical scene</Typography>
                <Typography variant="body2" color="text.secondary">
                  Pick the incident and camera evidence. Then run once.
                </Typography>
              </Box>
            </Stack>

            <ToggleButtonGroup
              exclusive
              fullWidth
              value={String(visiblePresets.find((index) => scenarioPresets[index].incidentId === incident.incidentId) ?? 1)}
              onChange={(_, value) => {
                if (value) loadScenario(Number(value));
              }}
              sx={{ mt: 2, display: "grid", gridTemplateColumns: "1fr", gap: 1 }}
            >
              {visiblePresets.map((index) => {
                const preset = scenarioPresets[index];
                return (
                  <ToggleButton key={preset.incidentId} value={String(index)} sx={{ justifyContent: "flex-start", py: 1.2 }}>
                    <Stack spacing={0.3} sx={{ alignItems: "flex-start" }}>
                      <Typography variant="body2" sx={{ fontWeight: 800 }}>
                        {preset.scenarioName}
                      </Typography>
                      <Typography variant="caption" color="text.secondary">
                        {preset.smokePpm} ppm, {preset.temperatureC} C, fire camera {pct(preset.cameraFireConfidence)}
                      </Typography>
                    </Stack>
                  </ToggleButton>
                );
              })}
            </ToggleButtonGroup>

            <Divider sx={{ my: 2 }} />

            <Stack spacing={2}>
              {physicalControls.map((control) => (
                <Box key={control.key}>
                  <Stack direction="row" sx={{ alignItems: "center", justifyContent: "space-between" }}>
                    <Typography variant="body2" color="text.secondary">
                      {control.label}
                    </Typography>
                    <Typography variant="body2" sx={{ fontWeight: 800 }}>
                      {signalValue(incident, control.key)}
                    </Typography>
                  </Stack>
                  <Slider
                    value={Number(incident[control.key])}
                    min={control.min}
                    max={control.max}
                    step={control.step}
                    onChange={(_, value) => updateIncident(control.key, value as never)}
                  />
                </Box>
              ))}
            </Stack>

            <Box sx={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 1.5, mt: 1 }}>
              <Paper elevation={0} sx={{ p: 1.5, bgcolor: "rgba(15, 23, 42, 0.72)" }}>
                <Typography variant="caption" color="text.secondary">
                  Drone available
                </Typography>
                <Switch checked={incident.droneAvailable} onChange={(_, value) => updateIncident("droneAvailable", value)} />
              </Paper>
              <Paper elevation={0} sx={{ p: 1.5, bgcolor: "rgba(15, 23, 42, 0.72)" }}>
                <Typography variant="caption" color="text.secondary">
                  Gate locked
                </Typography>
                <Switch checked={incident.gateLocked} onChange={(_, value) => updateIncident("gateLocked", value)} />
              </Paper>
            </Box>

            <Typography variant="subtitle2" sx={{ mt: 2, mb: 1 }}>
              Camera frame for vision API
            </Typography>
            <ToggleButtonGroup
              exclusive
              fullWidth
              value={sampleName}
              onChange={(_, value) => {
                if (value) {
                  setSampleName(value);
                  resetRunState();
                }
              }}
              sx={{ display: "grid", gridTemplateColumns: "1fr", gap: 1 }}
            >
              {(Object.keys(sampleImageMap) as (keyof typeof sampleImageMap)[]).map((name) => (
                <ToggleButton key={name} value={name} sx={{ justifyContent: "flex-start" }}>
                  <CameraAltIcon sx={{ mr: 1 }} fontSize="small" />
                  {sampleImageMap[name].label}
                </ToggleButton>
              ))}
            </ToggleButtonGroup>

            <Divider sx={{ my: 2 }} />

            <Stack direction="row" spacing={1} sx={{ alignItems: "center" }}>
              <PolicyIcon color="primary" fontSize="small" />
              <Typography variant="h6">2. Tune agent and guardrails</Typography>
            </Stack>
            <Typography variant="caption" color="text.secondary">
              These controls are sent into the agent route and change fallback planner behavior too.
            </Typography>

            <Typography variant="subtitle2" sx={{ mt: 1.5 }}>
              Agent posture
            </Typography>
            <ToggleButtonGroup
              exclusive
              fullWidth
              value={agentControls.operatingMode}
              onChange={(_, value) => {
                if (value) setAgentControls((current) => ({ ...current, operatingMode: value }));
              }}
              sx={{ display: "grid", gridTemplateColumns: "1fr", gap: 1 }}
            >
              <ToggleButton value="conservative">Conservative</ToggleButton>
              <ToggleButton value="balanced">Balanced</ToggleButton>
              <ToggleButton value="rapid_response">Rapid response</ToggleButton>
            </ToggleButtonGroup>
            <Typography variant="caption" color="text.secondary">
              {postureCopy[agentControls.operatingMode]}
            </Typography>

            <Typography variant="subtitle2" sx={{ mt: 1.5 }}>
              Authority guardrail
            </Typography>
            <ToggleButtonGroup
              exclusive
              fullWidth
              value={agentControls.authorityPosture}
              onChange={(_, value) => {
                if (value) setAgentControls((current) => ({ ...current, authorityPosture: value }));
              }}
              sx={{ display: "grid", gridTemplateColumns: "1fr", gap: 1 }}
            >
              <ToggleButton value="strict">Strict</ToggleButton>
              <ToggleButton value="approval_gated">Approval gated</ToggleButton>
              <ToggleButton value="critical_only">Critical only</ToggleButton>
            </ToggleButtonGroup>
            <Typography variant="caption" color="text.secondary">
              {authorityCopy[agentControls.authorityPosture]}
            </Typography>

            <TextField
              fullWidth
              multiline
              minRows={2}
              label="Planner instruction"
              value={plannerNote}
              onChange={(event) => setPlannerNote(event.target.value)}
              sx={{ mt: 2 }}
              slotProps={{ htmlInput: { maxLength: 500 } }}
            />

            <Button
              size="large"
              variant="contained"
              fullWidth
              startIcon={<AccountTreeIcon />}
              onClick={() => void runComparison()}
              disabled={running || training}
              sx={{ mt: 2.5, py: 1.4 }}
            >
              {running || training ? "Running full-stack workflow..." : "Run full-stack agentic workflow"}
            </Button>
            {running || training ? <LinearProgress sx={{ mt: 2 }} /> : null}
            {message ? (
              <Alert severity={message.includes("failed") || message.includes("quota") ? "warning" : "success"} sx={{ mt: 2 }}>
                {message}
              </Alert>
            ) : null}
          </Surface>

          <Box sx={{ display: "grid", gap: 2 }}>
            <Surface>
              <Stack direction="row" spacing={1.2} sx={{ alignItems: "center" }}>
                <TimelineIcon color="primary" />
                <Box>
              <Typography variant="h5">3. Runtime flow</Typography>
              <Typography variant="body2" color="text.secondary">
                    Follow the handoff from real-world signal to governed action proposal.
              </Typography>
                </Box>
              </Stack>
              <Box sx={{ mt: 2, display: "grid", gridTemplateColumns: { xs: "1fr", sm: "repeat(2, 1fr)", xl: "repeat(6, 1fr)" }, gap: 1.25 }}>
                {flowSteps.map((step) => (
                  <Paper key={step.label} elevation={0} sx={{ p: 1.5, bgcolor: "rgba(2, 6, 23, 0.58)", border: "1px solid rgba(148, 163, 184, 0.16)" }}>
                    <Stack direction="row" spacing={1} sx={{ alignItems: "center", color: "primary.main" }}>
                      {step.icon}
                      <Typography variant="caption" sx={{ color: "text.secondary", fontWeight: 800 }}>
                        {step.status}
                      </Typography>
                    </Stack>
                    <Typography variant="subtitle2" sx={{ mt: 1, fontWeight: 900 }}>
                      {step.label}
                    </Typography>
                    <Typography variant="body2" sx={{ mt: 0.5 }}>
                      {step.value}
                    </Typography>
                  </Paper>
                ))}
              </Box>

              <Typography variant="h6" sx={{ mt: 2.5 }}>
                Difference in response
              </Typography>
              <Box sx={{ mt: 1, display: "grid", gridTemplateColumns: { xs: "1fr", lg: "repeat(3, 1fr)" }, gap: 2 }}>
                <Paper elevation={0} sx={{ p: 2, border: "1px solid rgba(245, 158, 11, 0.25)", bgcolor: "rgba(245, 158, 11, 0.08)" }}>
                  <Chip size="small" color="warning" label="Rule engine" />
                  <Typography variant="h6" sx={{ mt: 1 }}>
                    Detects alarm conditions
                  </Typography>
                  <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
                    Uses only smoke and heat. It cannot reason over camera evidence, SOP, tools, policy, or approvals.
                  </Typography>
                  <Divider sx={{ my: 1.5 }} />
                  <EvidenceLine label="Output" value={formatAction(ruleResult.action)} />
                  <EvidenceLine label="Severity" value={ruleResult.severity} />
                  <EvidenceLine label="Real system" value="Local TypeScript" />
                  <EvidenceLine label="Executes tools" value="No" />
                </Paper>

                <Paper elevation={0} sx={{ p: 2, border: "1px solid rgba(96, 165, 250, 0.25)", bgcolor: "rgba(96, 165, 250, 0.08)" }}>
                  <Chip size="small" color="info" label="ML model" />
                  <Typography variant="h6" sx={{ mt: 1 }}>
                    Predicts fire probability
                  </Typography>
                  <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
                    Uses fused signals and browser-side TensorFlow.js training. It produces probability, not execution.
                  </Typography>
                  <Divider sx={{ my: 1.5 }} />
                  <EvidenceLine label="Prediction" value={pct(mlResult.fireProbability)} />
                  <EvidenceLine label="Risk" value={mlResult.riskLevel} />
                  <EvidenceLine label="Model" value={model || hasRun ? "TF.js trained" : "Baseline before run"} />
                  <EvidenceLine label="Executes tools" value="No" />
                </Paper>

                <Paper elevation={0} sx={{ p: 2, border: "1px solid rgba(34, 211, 238, 0.28)", bgcolor: "rgba(34, 211, 238, 0.08)" }}>
                  <Chip size="small" color="primary" label="Agentic control plane" />
                  <Typography variant="h6" sx={{ mt: 1 }}>
                    Governs response
                  </Typography>
                  <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
                    Uses all evidence, tool contracts, policy, approval gates, trace, and decision records.
                  </Typography>
                  <Divider sx={{ my: 1.5 }} />
                  <EvidenceLine label="Agent risk" value={agenticResult?.riskAssessment.level ?? "after run"} />
                  <EvidenceLine label="Actions proposed" value={agenticResult ? String(agenticResult.proposedActions.length) : "0"} />
                  <EvidenceLine label="Policy checks" value={String(policyDecisions.length)} />
                  <EvidenceLine label="Physical execution" value="Sandbox only" />
                </Paper>
              </Box>
            </Surface>

            <Box sx={{ display: "grid", gridTemplateColumns: { xs: "1fr", lg: "repeat(5, 1fr)" }, gap: 2 }}>
              {runtimeLanes.map((lane) => (
                <Paper
                  key={lane.title}
                  elevation={0}
                  sx={{
                    p: 2,
                    bgcolor: "rgba(15, 23, 42, 0.88)",
                    border: "1px solid rgba(148, 163, 184, 0.18)",
                    minHeight: 190
                  }}
                >
                  <Stack direction="row" spacing={1} sx={{ alignItems: "center", color: "primary.main" }}>
                    {lane.icon}
                    <Typography variant="subtitle1" sx={{ fontWeight: 800 }}>
                      {lane.title}
                    </Typography>
                  </Stack>
                  <Chip size="small" label={lane.status} sx={{ mt: 1 }} color={lane.title === "Guardrails" ? "warning" : "default"} />
                  <Divider sx={{ my: 1.5 }} />
                  {lane.lines.map(([label, value]) => (
                    <EvidenceLine key={label} label={label} value={value} />
                  ))}
                </Paper>
              ))}
            </Box>

            <Surface>
              <Stack direction={{ xs: "column", md: "row" }} spacing={2} sx={{ alignItems: { xs: "stretch", md: "center" }, justifyContent: "space-between" }}>
                <Box>
                  <Typography variant="h5">3. Action plan, approvals, and audit</Typography>
                  <Typography variant="body2" color="text.secondary">
                    After a run, this section shows exactly what the agent proposed and why policy allowed, blocked, or approval-gated it.
                  </Typography>
                </Box>
                <Stack direction="row" spacing={1} sx={{ flexWrap: "wrap", gap: 1 }}>
                  <Chip icon={<PolicyIcon />} color="warning" label={`${gatedPolicies.length} approval gates`} />
                  <Chip icon={<ShieldIcon />} color={blockedPolicies.length ? "error" : "success"} label={`${blockedPolicies.length} blocked`} />
                  <Chip icon={<FactCheckIcon />} color={decisionRecord ? "success" : "default"} label={decisionRecord ? "audit written" : "audit pending"} />
                </Stack>
              </Stack>

              <Box sx={{ mt: 2, display: "grid", gridTemplateColumns: { xs: "1fr", md: "repeat(2, 1fr)" }, gap: 2 }}>
                {(agenticResult?.proposedActions.length ? agenticResult.proposedActions : []).map((proposal) => {
                  const policy = policyDecisions.find((item) => item.action === proposal.action);
                  return (
                    <Paper key={proposal.action} elevation={0} sx={{ p: 2, bgcolor: "rgba(2, 6, 23, 0.58)", border: "1px solid rgba(148, 163, 184, 0.16)" }}>
                      <Stack direction="row" spacing={2} sx={{ alignItems: "center", justifyContent: "space-between" }}>
                        <Typography variant="subtitle1" sx={{ fontWeight: 800 }}>
                          {formatAction(proposal.action)}
                        </Typography>
                        <Chip
                          size="small"
                          color={policy?.blocked ? "error" : policy?.requiresHumanApproval ? "warning" : "success"}
                          label={policy?.blocked ? "blocked" : policy?.requiresHumanApproval ? "approval required" : "allowed"}
                        />
                      </Stack>
                      <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
                        {policy?.reason ?? proposal.reason}
                      </Typography>
                    </Paper>
                  );
                })}
                {!agenticResult ? (
                  <Alert severity="info" icon={<AutoAwesomeIcon />}>
                    Press the run button to generate the governed plan. OpenAI will be used when quota is available; otherwise the deterministic fallback keeps the demo working.
                  </Alert>
                ) : null}
              </Box>
            </Surface>
          </Box>
        </Box>

        <Accordion sx={{ mt: 3, bgcolor: "rgba(15, 23, 42, 0.88)", border: "1px solid rgba(148, 163, 184, 0.18)" }}>
          <AccordionSummary expandIcon={<ExpandMoreIcon />}>
            <Stack direction="row" spacing={1} sx={{ alignItems: "center" }}>
              <MemoryIcon color="primary" />
              <Typography sx={{ fontWeight: 800 }}>Technical payload: trace, policy, decision record JSON</Typography>
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
