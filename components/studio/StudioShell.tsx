"use client";

import { useEffect, useMemo, useState } from "react";
import type * as tf from "@tensorflow/tfjs";
import { motion } from "framer-motion";
import { BrainCircuit, Camera, Cpu, Gauge, ShieldCheck, Siren } from "lucide-react";
import { Alert } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Slider } from "@/components/ui/slider";
import { Switch } from "@/components/ui/switch";
import { JsonInspector } from "@/components/studio/JsonInspector";
import { RiskBadge } from "@/components/studio/RiskBadge";
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
import type { AgenticResult, DecisionRecord, IncidentState, MLResult, PolicyDecision, RuleResult, TraceEvent, VisionResult } from "@/lib/types";

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

function pct(value: number) {
  return `${Math.round(value * 100)}%`;
}

function riskFromProbability(probability: number) {
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

  function loadScenario(index: number) {
    const next = clonePreset(index);
    setIncident(next);
    setRuleResult(evaluateRules(next));
    setMlResult({ ...heuristicRiskPrediction(next), modelVersion, metrics: modelMetrics });
    setVisionResult(undefined);
    setAgenticResult(undefined);
    setPolicyDecisions([]);
    setDecisionRecord(undefined);
    setHasRun(false);
    setMessage(undefined);
    appendTrace({ type: "incident_created", actor: "system", status: "success", output: next, explanation: `Loaded ${next.scenarioName}.` });
  }

  function updateIncident<K extends keyof IncidentState>(key: K, value: IncidentState[K]) {
    setIncident((current) => {
      const next = { ...current, [key]: value };
      setRuleResult(evaluateRules(next));
      setMlResult({ ...heuristicRiskPrediction(next), modelVersion, metrics: modelMetrics });
      setVisionResult(undefined);
      setAgenticResult(undefined);
      setPolicyDecisions([]);
      setDecisionRecord(undefined);
      setHasRun(false);
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
    setMessage("Running the same incident through rules, ML, vision, policy, and the agentic planner...");
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
            operatingMode: "balanced",
            authorityPosture: "critical_only",
            operatorInstruction: "Compare rule detection, ML prediction, and governed agentic coordination. Never execute physical actions directly."
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
      setMessage("Comparison complete. Read the three response cards from left to right.");
    } catch (error) {
      const text = error instanceof Error ? error.message : "Comparison failed.";
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

  return (
    <div className="mx-auto max-w-7xl space-y-5">
      <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }}>
        <Card className="glass-grid overflow-hidden">
          <CardHeader className="pb-4">
            <div className="flex flex-wrap items-center gap-2">
              <Badge className="border-cyan-300/30 bg-cyan-300/10 text-cyan-100">AgenticOps Studio</Badge>
              <Badge className="border-white/15 bg-white/5 text-slate-200">One incident</Badge>
              <Badge className="border-white/15 bg-white/5 text-slate-200">One button</Badge>
              <Badge className="border-white/15 bg-white/5 text-slate-200">Three responses</Badge>
            </div>
            <CardTitle className="max-w-4xl text-3xl md:text-5xl">See exactly how rule, ML, and agentic AI respond differently.</CardTitle>
            <CardDescription className="max-w-3xl text-base">
              Step 1: choose the physical fire situation. Step 2: run the side-by-side comparison. Step 3: read what detects, what predicts, and what governs action.
            </CardDescription>
          </CardHeader>
          <CardContent className="grid gap-4 lg:grid-cols-[1fr_.8fr]">
            <div className="rounded-lg border border-white/10 bg-black/20 p-4">
              <div className="mb-3 flex items-center gap-2">
                <Siren className="h-5 w-5 text-amber-200" />
                <h2 className="text-lg font-semibold text-slate-100">Step 1. Physical incident</h2>
              </div>
              <div className="grid gap-2 sm:grid-cols-3">
                {visiblePresets.map((index) => {
                  const preset = scenarioPresets[index];
                  const selected = incident.incidentId === preset.incidentId;
                  return (
                    <button
                      key={preset.incidentId}
                      type="button"
                      onClick={() => loadScenario(index)}
                      className={`rounded-md border p-3 text-left text-xs transition ${
                        selected ? "border-cyan-300/70 bg-cyan-400/10" : "border-white/10 bg-black/20 hover:border-cyan-300/40"
                      }`}
                    >
                      <p className="font-semibold text-slate-100">{preset.scenarioName}</p>
                      <p className="mt-1 text-slate-400">
                        {preset.smokePpm} ppm / {preset.temperatureC} C
                      </p>
                    </button>
                  );
                })}
              </div>

              <div className="mt-4 grid gap-4 md:grid-cols-[1fr_.75fr]">
                <div className="space-y-4">
                  {physicalControls.map((control) => (
                    <div key={control.key} className="space-y-2">
                      <div className="flex justify-between text-sm text-slate-300">
                        <span>{control.label}</span>
                        <span>{signalValue(incident, control.key)}</span>
                      </div>
                      <Slider
                        value={[Number(incident[control.key])]}
                        min={control.min}
                        max={control.max}
                        step={control.step}
                        onValueChange={([value]) => updateIncident(control.key, value as never)}
                      />
                    </div>
                  ))}
                  <div className="flex flex-wrap gap-4 text-sm text-slate-300">
                    <label className="flex items-center gap-2">
                      Drone available <Switch checked={incident.droneAvailable} onCheckedChange={(value) => updateIncident("droneAvailable", value)} />
                    </label>
                    <label className="flex items-center gap-2">
                      Gate locked <Switch checked={incident.gateLocked} onCheckedChange={(value) => updateIncident("gateLocked", value)} />
                    </label>
                  </div>
                </div>

                <div className="rounded-md border border-white/10 bg-slate-950/70 p-3">
                  <div className="flex items-center gap-2 text-sm font-semibold text-slate-100">
                    <Camera className="h-4 w-4 text-cyan-200" />
                    Camera frame for vision
                  </div>
                  <div className="mt-3 grid gap-2">
                    {(Object.keys(sampleImageMap) as (keyof typeof sampleImageMap)[]).map((name) => (
                      <button
                        key={name}
                        type="button"
                        onClick={() => {
                          setSampleName(name);
                          setVisionResult(undefined);
                          setAgenticResult(undefined);
                          setDecisionRecord(undefined);
                          setHasRun(false);
                        }}
                        className={`rounded-md border px-3 py-2 text-left text-xs transition ${
                          sampleName === name ? "border-cyan-300/70 bg-cyan-400/10 text-cyan-100" : "border-white/10 bg-black/20 text-slate-300"
                        }`}
                      >
                        {sampleImageMap[name].label}
                      </button>
                    ))}
                  </div>
                  <p className="mt-3 text-xs leading-5 text-slate-400">
                    Roboflow runs server-side when configured. If quota or keys are unavailable, the app returns a labeled sample inference.
                  </p>
                </div>
              </div>
            </div>

            <div className="rounded-lg border border-cyan-300/20 bg-cyan-300/10 p-4">
              <div className="flex items-center gap-2">
                <Gauge className="h-5 w-5 text-cyan-100" />
                <h2 className="text-lg font-semibold text-slate-100">Step 2. Run the comparison</h2>
              </div>
              <p className="mt-2 text-sm leading-6 text-slate-300">
                This single action runs local rules, trains or uses browser ML, calls vision inference, asks the governed planner, checks policy, and writes an audit record.
              </p>
              <Button className="mt-4 h-12 w-full text-base" onClick={() => void runComparison()} disabled={running || training}>
                {running || training ? "Running comparison..." : "Run side-by-side comparison"}
              </Button>
              {message ? <Alert className="mt-4">{message}</Alert> : null}

              <div className="mt-4 grid gap-2 text-xs text-slate-300">
                <div className="flex items-center justify-between rounded-md border border-white/10 bg-black/20 p-3">
                  <span>OpenAI planner</span>
                  <Badge>{health?.openaiConfigured ? `live ${health.openaiModel}` : "fallback"}</Badge>
                </div>
                <div className="flex items-center justify-between rounded-md border border-white/10 bg-black/20 p-3">
                  <span>Roboflow vision</span>
                  <Badge>{health?.roboflowConfigured ? "live server route" : "sample fallback"}</Badge>
                </div>
                <div className="flex items-center justify-between rounded-md border border-white/10 bg-black/20 p-3">
                  <span>Physical actions</span>
                  <Badge>sandboxed</Badge>
                </div>
                <div className="flex items-center justify-between rounded-md border border-white/10 bg-black/20 p-3">
                  <span>Decision record</span>
                  <Badge>{decisionRecord ? decisionRecord.runId : "created after run"}</Badge>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      </motion.div>

      <div className="grid gap-4 xl:grid-cols-3">
        <Card className="border-amber-300/20">
          <CardHeader>
            <div className="flex items-center justify-between gap-3">
              <Badge className="bg-amber-300/15 text-amber-100">1. Detects</Badge>
              <Cpu className="h-5 w-5 text-amber-100" />
            </div>
            <CardTitle>Rule-based automation</CardTitle>
            <CardDescription>Fast fixed logic. It only reads smoke and heat.</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="rounded-md border border-white/10 bg-black/20 p-3 text-sm">
              <p className="text-xs uppercase tracking-wide text-slate-500">Inputs used</p>
              <p className="mt-2 text-slate-100">Smoke ppm and temperature only</p>
            </div>
            <div className="rounded-md border border-white/10 bg-black/20 p-3 text-sm">
              <p className="text-xs uppercase tracking-wide text-slate-500">Decision</p>
              <div className="mt-2 flex flex-wrap items-center gap-2">
                <RiskBadge level={ruleResult.severity} />
                <span className="text-slate-100">{formatAction(ruleResult.action)}</span>
              </div>
              <p className="mt-2 text-xs leading-5 text-slate-400">IF smoke_ppm &gt;= 70 THEN raise_alarm. No camera, SOP, policy, approval, or tools.</p>
            </div>
            <div className="rounded-md border border-white/10 bg-black/20 p-3 text-sm">
              <p className="text-xs uppercase tracking-wide text-slate-500">What is real here</p>
              <p className="mt-2 text-slate-100">Local TypeScript rule engine</p>
              <p className="mt-2 text-xs leading-5 text-amber-100">Limitation: deterministic detection cannot coordinate a governed response.</p>
            </div>
          </CardContent>
        </Card>

        <Card className="border-blue-300/20">
          <CardHeader>
            <div className="flex items-center justify-between gap-3">
              <Badge className="bg-blue-300/15 text-blue-100">2. Predicts</Badge>
              <BrainCircuit className="h-5 w-5 text-blue-100" />
            </div>
            <CardTitle>ML-based prediction</CardTitle>
            <CardDescription>Fuses signals into probability. It still does not act.</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="rounded-md border border-white/10 bg-black/20 p-3 text-sm">
              <p className="text-xs uppercase tracking-wide text-slate-500">Inputs used</p>
              <p className="mt-2 text-slate-100">Sensors, camera confidence, occupancy, sensor health, false-alarm history, wind</p>
            </div>
            <div className="rounded-md border border-white/10 bg-black/20 p-3 text-sm">
              <p className="text-xs uppercase tracking-wide text-slate-500">Prediction</p>
              <div className="mt-2 flex flex-wrap items-center gap-2">
                <RiskBadge level={mlResult.riskLevel} />
                <span className="text-slate-100">{pct(mlResult.fireProbability)} fire probability</span>
              </div>
              <p className="mt-2 text-xs leading-5 text-slate-400">
                Top signal: {topFeature ? `${topFeature.feature} (${pct(topFeature.importance)})` : "not available"}.
              </p>
            </div>
            <div className="rounded-md border border-white/10 bg-black/20 p-3 text-sm">
              <p className="text-xs uppercase tracking-wide text-slate-500">What is real here</p>
              <p className="mt-2 text-slate-100">{model ? "TensorFlow.js model trained in this browser" : hasRun ? "TensorFlow.js trained during comparison" : "Baseline until first run trains TensorFlow.js"}</p>
              <p className="mt-2 text-xs leading-5 text-blue-100">Limitation: ML gives risk. It does not call tools or govern execution.</p>
            </div>
          </CardContent>
        </Card>

        <Card className="border-cyan-300/20">
          <CardHeader>
            <div className="flex items-center justify-between gap-3">
              <Badge className="bg-cyan-300/15 text-cyan-100">3. Governs</Badge>
              <ShieldCheck className="h-5 w-5 text-cyan-100" />
            </div>
            <CardTitle>Governed agentic AI</CardTitle>
            <CardDescription>Coordinates evidence, tools, policy, approval, and audit.</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="rounded-md border border-white/10 bg-black/20 p-3 text-sm">
              <p className="text-xs uppercase tracking-wide text-slate-500">Inputs used</p>
              <p className="mt-2 text-slate-100">All physical state, Roboflow vision, ML risk, rules, SOP, tool registry, policy</p>
            </div>
            <div className="rounded-md border border-white/10 bg-black/20 p-3 text-sm">
              <p className="text-xs uppercase tracking-wide text-slate-500">Plan</p>
              {agenticResult ? (
                <div className="mt-2 space-y-2">
                  <div className="flex flex-wrap items-center gap-2">
                    <RiskBadge level={agenticResult.riskAssessment.level} />
                    <span className="text-slate-100">{agenticResult.proposedActions.length} proposed actions</span>
                  </div>
                  <p className="text-xs leading-5 text-slate-400">{agenticResult.incidentSummary}</p>
                </div>
              ) : (
                <p className="mt-2 text-slate-400">Run the comparison to see the governed action plan.</p>
              )}
            </div>
            <div className="rounded-md border border-white/10 bg-black/20 p-3 text-sm">
              <p className="text-xs uppercase tracking-wide text-slate-500">Action governance</p>
              {agenticResult ? (
                <div className="mt-2 space-y-2">
                  {(physicalActions ?? agenticResult.proposedActions).slice(0, 4).map((proposal) => {
                    const policy = policyDecisions.find((item) => item.action === proposal.action);
                    return (
                      <div key={proposal.action} className="rounded-md border border-white/10 bg-slate-950/70 p-2">
                        <div className="flex flex-wrap items-center justify-between gap-2">
                          <span className="text-slate-100">{formatAction(proposal.action)}</span>
                          <Badge>{policy?.blocked ? "blocked" : policy?.requiresHumanApproval ? "approval" : "allowed"}</Badge>
                        </div>
                        <p className="mt-1 text-xs leading-5 text-slate-400">{policy?.reason ?? proposal.reason}</p>
                      </div>
                    );
                  })}
                </div>
              ) : (
                <p className="mt-2 text-slate-400">No action can execute until policy and approval gates are evaluated.</p>
              )}
            </div>
            <div className="rounded-md border border-white/10 bg-black/20 p-3 text-sm">
              <p className="text-xs uppercase tracking-wide text-slate-500">What is real vs sandbox</p>
              <p className="mt-2 text-slate-100">
                OpenAI {health?.openaiConfigured ? "live" : "fallback"}, Roboflow {visionResult?.provider ?? (health?.roboflowConfigured ? "live" : "sample")}, policy real, audit real, physical actions sandboxed.
              </p>
              <p className="mt-2 text-xs leading-5 text-cyan-100">
                Key point: the LLM is not the control system. It proposes a governed plan; tools act only inside policy and approval boundaries.
              </p>
            </div>
          </CardContent>
        </Card>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>What changed after the run</CardTitle>
          <CardDescription>The output is intentionally simple: detection, prediction, governed coordination, and audit proof.</CardDescription>
        </CardHeader>
        <CardContent className="grid gap-3 md:grid-cols-4">
          <div className="rounded-md border border-white/10 bg-black/20 p-3 text-sm">
            <p className="text-xs uppercase tracking-wide text-slate-500">Vision result</p>
            <p className="mt-2 text-slate-100">
              {visionResult ? `${visionResult.provider}: fire ${pct(visionResult.maxFireConfidence)}, smoke ${pct(visionResult.maxSmokeConfidence)}` : "Not run yet"}
            </p>
          </div>
          <div className="rounded-md border border-white/10 bg-black/20 p-3 text-sm">
            <p className="text-xs uppercase tracking-wide text-slate-500">Policy gates</p>
            <p className="mt-2 text-slate-100">
              {policyDecisions.length ? `${blockedPolicies.length} blocked, ${gatedPolicies.length} need approval` : "Not evaluated yet"}
            </p>
          </div>
          <div className="rounded-md border border-white/10 bg-black/20 p-3 text-sm">
            <p className="text-xs uppercase tracking-wide text-slate-500">Trace</p>
            <p className="mt-2 text-slate-100">{trace.length} structured events</p>
          </div>
          <div className="rounded-md border border-white/10 bg-black/20 p-3 text-sm">
            <p className="text-xs uppercase tracking-wide text-slate-500">Audit record</p>
            <p className="mt-2 text-slate-100">{decisionRecord ? decisionRecord.runId : "Not written yet"}</p>
          </div>
        </CardContent>
      </Card>

      <details className="rounded-lg border border-white/10 bg-black/20 p-4">
        <summary className="cursor-pointer text-sm font-semibold text-slate-100">Technical payload and decision record JSON</summary>
        <div className="mt-4">
          <JsonInspector title="Current runtime payload" value={resultJson} />
        </div>
      </details>
    </div>
  );
}
