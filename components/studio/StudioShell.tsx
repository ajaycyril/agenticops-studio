"use client";

import { useEffect, useMemo, useState } from "react";
import type * as tf from "@tensorflow/tfjs";
import { motion } from "framer-motion";
import { BrainCircuit, Camera, Cpu, ShieldCheck } from "lucide-react";
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

type UseCaseId = "rules" | "ml" | "agentic";

type HealthStatus = {
  status: "ok";
  openaiConfigured: boolean;
  roboflowConfigured: boolean;
  openaiModel: string;
  openaiMaxAgentCallsPerRun: number;
};

type RunSummary = {
  title: string;
  subtitle: string;
  proof: string;
};

const useCases: Array<{
  id: UseCaseId;
  title: string;
  short: string;
  icon: typeof Cpu;
  accent: string;
  inputs: string;
  output: string;
  limitation: string;
}> = [
  {
    id: "rules",
    title: "Rule-based automation",
    short: "Detects",
    icon: Cpu,
    accent: "text-cyan-100 border-cyan-300/30 bg-cyan-400/10",
    inputs: "Smoke ppm and temperature only.",
    output: "Alarm, severity, and fixed escalation.",
    limitation: "It cannot use camera evidence, occupancy, SOP, tools, policy, or approvals."
  },
  {
    id: "ml",
    title: "ML-based prediction",
    short: "Predicts",
    icon: BrainCircuit,
    accent: "text-amber-100 border-amber-300/30 bg-amber-400/10",
    inputs: "Sensors, camera confidence, device health, false-alarm history, occupancy, and wind.",
    output: "Fire probability, risk level, and feature importance.",
    limitation: "It predicts risk. It does not coordinate actions or execute tools."
  },
  {
    id: "agentic",
    title: "Governed agentic AI",
    short: "Coordinates",
    icon: ShieldCheck,
    accent: "text-emerald-100 border-emerald-300/30 bg-emerald-400/10",
    inputs: "All evidence plus SOP, tool registry, policy, human approval state, and audit trail.",
    output: "Structured action plan, policy checks, approval gates, trace, and decision record.",
    limitation: "The LLM is not the control system. Policy and humans govern execution."
  }
];

function pct(value: number) {
  return `${Math.round(value * 100)}%`;
}

function riskFromProbability(probability: number) {
  if (probability >= 0.82) return "critical";
  if (probability >= 0.62) return "high";
  if (probability >= 0.36) return "medium";
  return "low";
}

export function StudioShell() {
  const [incident, setIncident] = useState<IncidentState>(() => clonePreset(1));
  const [activeUseCase, setActiveUseCase] = useState<UseCaseId>("agentic");
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
      explanation: "Loaded default confirmed fire scenario."
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
  const [runSummary, setRunSummary] = useState<RunSummary>({
    title: "Ready",
    subtitle: "Choose a use case and run it against the physical incident.",
    proof: "No run yet."
  });

  const tools = useMemo(() => getToolRegistry(), []);
  const selectedUseCase = useCases.find((item) => item.id === activeUseCase) ?? useCases[2];

  useEffect(() => {
    fetch("/api/health")
      .then((response) => response.json())
      .then((data: HealthStatus) => setHealth(data))
      .catch(() => setHealth(undefined));
  }, []);

  function appendTrace(event: Omit<TraceEvent, "id" | "timestamp">) {
    setTrace((current) => {
      const next = [...current, createTraceEvent(event)];
      saveTrace(next);
      return next;
    });
  }

  function loadScenario(index: number) {
    const next = clonePreset(index);
    const nextRule = evaluateRules(next);
    const nextMl = heuristicRiskPrediction(next);
    setIncident(next);
    setRuleResult(nextRule);
    setMlResult(nextMl);
    setVisionResult(undefined);
    setAgenticResult(undefined);
    setPolicyDecisions([]);
    setDecisionRecord(undefined);
    setRunSummary({
      title: "Physical incident changed",
      subtitle: next.scenarioName,
      proof: "Run a use case to see how it behaves."
    });
    appendTrace({ type: "incident_created", actor: "system", status: "success", output: next });
  }

  function updateIncident<K extends keyof IncidentState>(key: K, value: IncidentState[K]) {
    setIncident((current) => {
      const next = { ...current, [key]: value };
      setRuleResult(evaluateRules(next));
      setMlResult({ ...heuristicRiskPrediction(next), modelVersion, metrics: modelMetrics });
      return next;
    });
  }

  function runRules(incidentForRun = incident) {
    const result = evaluateRules(incidentForRun);
    setRuleResult(result);
    appendTrace({ type: "rule_engine_evaluated", actor: "system", input: incidentForRun, output: result, status: "success" });
    setRunSummary({
      title: "Rule-based automation detected",
      subtitle: result.explanation,
      proof: `${result.rulesEvaluated.filter((rule) => rule.passed).length}/${result.rulesEvaluated.length} rules passed. Action: ${result.action}.`
    });
    return result;
  }

  async function trainBrowserModel() {
    setTraining(true);
    setMessage("Training a small TensorFlow.js model in the browser...");
    try {
      const trained = await trainRiskModel({
        size: 220,
        epochs: 18,
        falseAlarmBias: incident.historicalFalseAlarmRate,
        learningRate: 0.08
      });
      setModel(trained.model);
      setModelMetrics(trained.metrics);
      setModelVersion(trained.modelVersion);
      saveModelVersion(trained.modelVersion);
      appendTrace({ type: "ml_model_training_completed", actor: "ml_model", status: "success", output: trained.metrics });
      setMessage("Browser ML model trained.");
    } finally {
      setTraining(false);
    }
  }

  async function runPrediction(incidentForRun = incident) {
    let probability: number;
    const features = incidentToFeatures(incidentForRun);
    if (model) {
      const tfjs = await import("@tensorflow/tfjs");
      const prediction = model.predict(tfjs.tensor2d([features])) as tf.Tensor;
      probability = (await prediction.data())[0];
      prediction.dispose();
    } else {
      probability = heuristicRiskPrediction(incidentForRun).fireProbability;
    }
    const result: MLResult = {
      fireProbability: Number(probability.toFixed(3)),
      riskLevel: riskFromProbability(probability),
      modelVersion,
      metrics: modelMetrics,
      featureImportance: approximateFeatureImportance(features),
      explanation: "ML predicts fire risk only. It does not coordinate response or execute physical actions."
    };
    setMlResult(result);
    appendTrace({ type: "ml_model_predicted", actor: "ml_model", input: incidentForRun, output: result, status: "success" });
    setRunSummary({
      title: "ML predicted risk",
      subtitle: `${pct(result.fireProbability)} fire probability, ${result.riskLevel} risk.`,
      proof: `Top feature: ${result.featureImportance[0]?.feature ?? "n/a"}. No tool execution.`
    });
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

  async function runVision() {
    const imagePayload = await loadSampleImageAsDataUrl(sampleName);
    const response = await fetch("/api/vision/roboflow", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ incidentId: incident.incidentId, sampleName, imageBase64: imagePayload })
    });
    const data = await response.json();
    if (!data.ok) throw new Error(data.error.message);
    const result = data.result as VisionResult;
    const nextIncident: IncidentState = {
      ...incident,
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

  async function runAgenticFlow() {
    const rule = evaluateRules(incident);
    setRuleResult(rule);
    const vision = await runVision();
    const ml = await runPrediction(vision.nextIncident);
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
          operatorInstruction: "Prioritize life safety and no unsupervised physical action."
        }
      })
    });
    const data = await response.json();
    if (!data.ok) throw new Error(data.error.message);
    const result = data.result as AgenticResult;
    const policies = evaluatePoliciesForActions(result.proposedActions.map((proposal) => proposal.action), vision.nextIncident, ml);
    const record = buildDecisionRecord({
      runId: `RUN-${Date.now()}`,
      incident: vision.nextIncident,
      ruleResult: rule,
      mlResult: ml,
      visionResult: vision.result,
      agenticResult: result,
      policyDecisions: policies,
      trace
    });
    setAgenticResult(result);
    setPolicyDecisions(policies);
    setDecisionRecord(record);
    sessionStorage.setItem("agenticops.latestDecisionRecord", JSON.stringify(record));
    appendTrace({ type: "agent_output_validated", actor: "llm_agent", output: result, status: "success", explanation: data.message });
    appendTrace({ type: "decision_record_written", actor: "system", output: { runId: record.runId }, status: "success" });
    setRunSummary({
      title: "Agentic workflow coordinated",
      subtitle: result.incidentSummary,
      proof: `${result.proposedActions.length} actions proposed, ${policies.length} policy checks, decision record written.`
    });
  }

  async function runSelectedUseCase() {
    setRunning(true);
    setMessage(undefined);
    try {
      if (activeUseCase === "rules") runRules();
      if (activeUseCase === "ml") await runPrediction();
      if (activeUseCase === "agentic") await runAgenticFlow();
    } catch (error) {
      const text = error instanceof Error ? error.message : "Run failed.";
      setMessage(text);
      appendTrace({ type: "error", actor: "system", output: { message: text }, status: "failed" });
    } finally {
      setRunning(false);
    }
  }

  const resultJson =
    activeUseCase === "rules"
      ? ruleResult
      : activeUseCase === "ml"
        ? mlResult
        : {
            agenticResult,
            policyDecisions,
            decisionRecordId: decisionRecord?.runId,
            visionResult
          };

  return (
    <div className="space-y-4">
      <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }}>
        <Card className="glass-grid">
          <CardHeader className="pb-4">
            <Badge className="w-fit border-cyan-300/30 bg-cyan-300/10 text-cyan-100">AgenticOps Studio</Badge>
            <CardTitle className="max-w-4xl text-3xl md:text-5xl">One incident. Three ways to respond.</CardTitle>
            <CardDescription className="max-w-3xl text-base">
              Keep the demo simple: set a physical fire scenario, choose the response style, and run it.
            </CardDescription>
          </CardHeader>
          <CardContent className="grid gap-3 md:grid-cols-3">
            {useCases.map((useCase) => {
              const Icon = useCase.icon;
              const selected = activeUseCase === useCase.id;
              return (
                <button
                  key={useCase.id}
                  type="button"
                  onClick={() => setActiveUseCase(useCase.id)}
                  className={`rounded-lg border p-4 text-left transition ${
                    selected ? useCase.accent : "border-white/10 bg-black/20 hover:border-cyan-300/40"
                  }`}
                >
                  <div className="flex items-center justify-between gap-3">
                    <Icon className="h-5 w-5" />
                    <Badge>{useCase.short}</Badge>
                  </div>
                  <h2 className="mt-4 text-lg font-semibold text-slate-100">{useCase.title}</h2>
                  <p className="mt-2 text-xs leading-5 text-slate-400">{useCase.inputs}</p>
                  <p className="mt-2 text-xs leading-5 text-cyan-100">Output: {useCase.output}</p>
                  <p className="mt-2 text-xs leading-5 text-amber-100">Limit: {useCase.limitation}</p>
                </button>
              );
            })}
          </CardContent>
        </Card>
      </motion.div>

      {message ? <Alert>{message}</Alert> : null}

      <div className="grid gap-4 xl:grid-cols-[.9fr_1.1fr]">
        <Card>
          <CardHeader>
            <Badge className="w-fit">Step 1</Badge>
            <CardTitle>Set the physical state</CardTitle>
            <CardDescription>Pick a scenario, then adjust only the few inputs that matter for the story.</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="grid gap-2 sm:grid-cols-3">
              {[1, 2, 4].map((index) => {
                const preset = scenarioPresets[index];
                return (
                  <button
                    key={preset.incidentId}
                    type="button"
                    onClick={() => loadScenario(index)}
                    className={`rounded-md border p-3 text-left text-xs ${
                      incident.incidentId === preset.incidentId ? "border-cyan-300/70 bg-cyan-400/10" : "border-white/10 bg-black/20"
                    }`}
                  >
                    <p className="font-semibold text-slate-100">{preset.scenarioName}</p>
                    <p className="mt-1 text-slate-400">{preset.smokePpm} ppm / {preset.temperatureC} C</p>
                  </button>
                );
              })}
            </div>

            {[
              ["smokePpm", "Smoke ppm", 0, 140],
              ["temperatureC", "Temperature C", 10, 80],
              ["cameraSmokeConfidence", "Camera smoke confidence", 0, 1],
              ["cameraFireConfidence", "Camera fire confidence", 0, 1]
            ].map(([key, label, min, max]) => (
              <div key={String(key)} className="space-y-2">
                <div className="flex justify-between text-sm text-slate-300">
                  <span>{label}</span>
                  <span>{Number(incident[key as keyof IncidentState]).toFixed(max === 1 ? 2 : 0)}</span>
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
              <label className="flex items-center gap-2">
                Drone available <Switch checked={incident.droneAvailable} onCheckedChange={(value) => updateIncident("droneAvailable", value)} />
              </label>
              <label className="flex items-center gap-2">
                Gate locked <Switch checked={incident.gateLocked} onCheckedChange={(value) => updateIncident("gateLocked", value)} />
              </label>
            </div>

            <div className="rounded-md border border-white/10 bg-black/20 p-3 text-xs leading-5 text-slate-300">
              Physical AI means the software is reasoning over real-world signals and possible actions. This demo keeps all physical actions sandboxed.
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <Badge className="w-fit">Steps 2 and 3</Badge>
            <CardTitle>{selectedUseCase.title}</CardTitle>
            <CardDescription>{selectedUseCase.short}: {selectedUseCase.limitation}</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="grid gap-3 sm:grid-cols-3">
              <div className="rounded-md border border-white/10 bg-black/20 p-3">
                <p className="text-xs text-slate-500">Risk</p>
                <div className="mt-2"><RiskBadge level={mlResult.riskLevel} /></div>
              </div>
              <div className="rounded-md border border-white/10 bg-black/20 p-3 text-sm">
                <p className="text-xs text-slate-500">Vision</p>
                <p className="mt-2 text-slate-100">{visionResult ? `${visionResult.provider} ${pct(visionResult.maxFireConfidence)}` : sampleImageMap[sampleName].label}</p>
              </div>
              <div className="rounded-md border border-white/10 bg-black/20 p-3 text-sm">
                <p className="text-xs text-slate-500">Runtime</p>
                <p className="mt-2 text-slate-100">
                  {activeUseCase === "agentic" ? (health?.openaiConfigured ? health.openaiModel : "fallback") : activeUseCase === "ml" ? "browser ML" : "local rules"}
                </p>
              </div>
            </div>

            <div className="flex flex-wrap gap-2">
              <Button onClick={() => void runSelectedUseCase()} disabled={running}>
                {running ? "Running..." : `Run ${selectedUseCase.short}`}
              </Button>
              {activeUseCase === "ml" ? (
                <Button variant="secondary" onClick={() => void trainBrowserModel()} disabled={training}>
                  {training ? "Training..." : "Train browser model"}
                </Button>
              ) : null}
              {activeUseCase === "agentic" ? (
                <Button variant="secondary" onClick={() => setSampleName(sampleName === "fire-smoke-room" ? "unclear-camera" : "fire-smoke-room")}>
                  <Camera className="h-4 w-4" /> Swap sample image
                </Button>
              ) : null}
            </div>

            <div className="rounded-lg border border-white/10 bg-black/20 p-4">
              <div className="flex items-start justify-between gap-3">
                <div>
                  <p className="text-sm font-semibold text-slate-100">{runSummary.title}</p>
                  <p className="mt-2 text-sm leading-6 text-slate-300">{runSummary.subtitle}</p>
                  <p className="mt-2 text-xs leading-5 text-cyan-100">{runSummary.proof}</p>
                </div>
                {decisionRecord ? <Badge>recorded</Badge> : null}
              </div>
            </div>

            <div className="grid gap-2 text-xs text-slate-400 sm:grid-cols-3">
              <div className="rounded-md border border-white/10 bg-black/20 p-3">OpenAI: {health?.openaiConfigured ? "configured" : "fallback"}</div>
              <div className="rounded-md border border-white/10 bg-black/20 p-3">Roboflow: {health?.roboflowConfigured ? "configured" : "sample"}</div>
              <div className="rounded-md border border-white/10 bg-black/20 p-3">Trace events: {trace.length}</div>
            </div>

            <JsonInspector title="Result" value={resultJson} />
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
