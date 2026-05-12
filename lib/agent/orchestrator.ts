import { Agent, run } from "@openai/agents";
import { env, isOpenAIConfigured } from "@/lib/env";
import { agenticResultSchema } from "@/lib/agent/schemas";
import { agentSystemPrompt, buildAgentUserPrompt } from "@/lib/agent/prompts";
import { AppError } from "@/lib/errors";
import type { AgenticResult, IncidentState, MLResult } from "@/lib/types";

type AgentControls = {
  operatingMode?: "balanced" | "conservative" | "rapid_response";
  authorityPosture?: "strict" | "approval_gated" | "critical_only";
  operatorInstruction?: string;
};

type AgentRunResponse = {
  result: AgenticResult;
  provider: "openai" | "sample";
  runtime: "openai-agents-sdk" | "deterministic-fallback";
  setupRequired: boolean;
  message?: string;
};

export function fallbackAgenticResult(incident: IncidentState, mlResult: Pick<MLResult, "fireProbability" | "riskLevel">, controls: AgentControls = {}): AgenticResult {
  const confidence = Math.max(incident.cameraSmokeConfidence, incident.cameraFireConfidence);
  const conservative = controls.operatingMode === "conservative";
  const rapidResponse = controls.operatingMode === "rapid_response";
  const proposedActions: AgenticResult["proposedActions"] = [
    {
      action: mlResult.riskLevel === "low" ? "generate_report" : "notify_operator",
      riskClass: mlResult.riskLevel,
      reason:
        mlResult.riskLevel === "low"
          ? "Low risk incident can be documented without escalation."
          : "Operator notification keeps a human in the loop for medium or higher fire risk.",
      evidenceUsed: [`ML fire probability ${mlResult.fireProbability}`, `Smoke ${incident.smokePpm} ppm`, `Temperature ${incident.temperatureC} C`],
      confidence: Math.min(0.95, Math.max(0.55, mlResult.fireProbability)),
      requiresPolicyCheck: true,
      requiresHumanApproval: false
    }
  ];

  if (confidence < (conservative ? 0.7 : 0.5)) {
    proposedActions.unshift({
      action: "request_camera_recheck",
      riskClass: "medium",
      reason: conservative
        ? "Conservative operating mode requests visual revalidation unless camera confidence is strong."
        : "Camera confidence is low, so the visual evidence should be revalidated before stronger action.",
      evidenceUsed: [`Max camera confidence ${confidence.toFixed(2)}`],
      confidence: 0.72,
      requiresPolicyCheck: true,
      requiresHumanApproval: false
    });
  }

  if (incident.droneAvailable && (mlResult.riskLevel !== "low" || rapidResponse)) {
    proposedActions.push({
      action: "dispatch_drone",
      riskClass: mlResult.riskLevel,
      reason: "Drone reconnaissance can gather more evidence while keeping physical action sandboxed.",
      evidenceUsed: [`Risk level ${mlResult.riskLevel}`, `Drone available ${incident.droneAvailable}`, `Wind ${incident.windSpeedKmh} km/h`],
      confidence: 0.74,
      requiresPolicyCheck: true,
      requiresHumanApproval: conservative || incident.windSpeedKmh > 25 || confidence < 0.7
    });
  }

  if (mlResult.riskLevel === "high" || mlResult.riskLevel === "critical") {
    proposedActions.push({
      action: "unlock_gate",
      riskClass: mlResult.riskLevel,
      reason: "Emergency access may be needed, but access-control action must be approval-gated.",
      evidenceUsed: [`Risk level ${mlResult.riskLevel}`, `Gate locked ${incident.gateLocked}`],
      confidence: 0.7,
      requiresPolicyCheck: true,
      requiresHumanApproval: true
    });
  }

  if (mlResult.riskLevel === "critical" && controls.authorityPosture !== "strict") {
    proposedActions.push({
      action: "notify_authority",
      riskClass: "critical",
      reason: "Critical fire risk meets authority-notification criteria in sandbox mode.",
      evidenceUsed: [`ML fire probability ${mlResult.fireProbability}`, `Camera fire ${incident.cameraFireConfidence}`],
      confidence: 0.82,
      requiresPolicyCheck: true,
      requiresHumanApproval: false
    });
  }

  proposedActions.push({
    action: "write_decision_record",
    riskClass: "low",
    reason: "Every governed run needs an auditable decision record.",
    evidenceUsed: ["Incident state", "Rule result", "ML result", "Policy decisions"],
    confidence: 1,
    requiresPolicyCheck: false,
    requiresHumanApproval: false
  });

  return {
    incidentSummary: `${incident.scenarioName}: smoke ${incident.smokePpm} ppm, temperature ${incident.temperatureC} C, ML fire probability ${mlResult.fireProbability}.`,
    riskAssessment: {
      level: mlResult.riskLevel,
      reason: "Fallback deterministic planner fused sensor, camera, and ML evidence without calling OpenAI.",
      evidence: [
        `Smoke ${incident.smokePpm} ppm`,
        `Temperature ${incident.temperatureC} C`,
        `Camera smoke ${incident.cameraSmokeConfidence}`,
        `Camera fire ${incident.cameraFireConfidence}`,
        `ML fire probability ${mlResult.fireProbability}`
      ],
      uncertainty: [
        ...(incident.sensorHealth < 0.6 ? ["Sensor health is degraded."] : []),
        ...(incident.occupancyStatus === "unknown" ? ["Occupancy is unknown."] : []),
        ...(confidence < 0.5 ? ["Camera confidence is low."] : [])
      ]
    },
    proposedActions,
    questionsForHuman:
      mlResult.riskLevel === "high" || mlResult.riskLevel === "critical"
        ? ["Approve access-control or drone actions only after checking site context and operator SOP."]
        : ["Confirm whether this appears to be a false alarm or needs monitoring."],
    sopReferences: ["SOP-FIRE-001", "SOP-FIRE-002", "SOP-FIRE-003", "SOP-FIRE-004"],
    explanation:
      `The planner is not the control system. It proposes a governed plan; policy, tool permissions, human approvals, and audit records control execution. Operating mode: ${controls.operatingMode ?? "balanced"}. ${controls.operatorInstruction ? `Operator instruction considered: ${controls.operatorInstruction}` : ""}`
  };
}

export async function runAgenticOrchestrator(payload: unknown): Promise<AgentRunResponse> {
  const parsed = payload as { incident: IncidentState; mlResult: MLResult; agentControls?: AgentControls };
  if (!isOpenAIConfigured()) {
    return {
      result: fallbackAgenticResult(parsed.incident, parsed.mlResult, parsed.agentControls),
      provider: "sample",
      runtime: "deterministic-fallback",
      setupRequired: true,
      message: "OPENAI_API_KEY is missing. Deterministic governed fallback planner used for this run."
    };
  }

  const plannerAgent = new Agent({
    name: "Physical AI Response Planner",
    instructions: agentSystemPrompt,
    model: env.OPENAI_MODEL,
    modelSettings: {
      maxTokens: env.OPENAI_MAX_OUTPUT_TOKENS,
      store: false
    },
    outputType: agenticResultSchema
  });
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), env.OPENAI_TIMEOUT_MS);
  try {
    const sdkRun = await run(plannerAgent, buildAgentUserPrompt(payload), {
      maxTurns: env.OPENAI_MAX_AGENT_CALLS_PER_RUN,
      signal: controller.signal
    });

    const parsedOutput = agenticResultSchema.safeParse(sdkRun.finalOutput);
    if (!parsedOutput.success) {
      throw new AppError({
        code: "AGENT_SCHEMA_VALIDATION_FAILED",
        message: "Agents SDK final output failed schema validation.",
        recoverable: true,
        status: 502
      });
    }

    return {
      result: parsedOutput.data,
      provider: "openai",
      runtime: "openai-agents-sdk",
      setupRequired: false,
      message: "OpenAI Agents SDK planner completed with schema-validated output."
    };
  } catch (error) {
    const status = typeof error === "object" && error && "status" in error ? Number((error as { status?: unknown }).status) : undefined;
    const message = error instanceof Error ? error.message : "Unknown OpenAI Agents SDK error.";
    if (error instanceof AppError && error.code.startsWith("AGENT_")) {
      return {
        result: fallbackAgenticResult(parsed.incident, parsed.mlResult, parsed.agentControls),
        provider: "sample",
        runtime: "deterministic-fallback",
        setupRequired: false,
        message: `${error.message} Deterministic governed fallback planner used for this run.`
      };
    }
    if (error instanceof AppError) throw error;
    if (status === 429 || message.includes("429") || message.toLowerCase().includes("quota")) {
      return {
        result: fallbackAgenticResult(parsed.incident, parsed.mlResult, parsed.agentControls),
        provider: "sample",
        runtime: "deterministic-fallback",
        setupRequired: false,
        message: "OpenAI quota/rate limit reached (429). Deterministic governed fallback planner used for this run."
      };
    }
    if (message.toLowerCase().includes("timeout") || message.toLowerCase().includes("abort")) {
      return {
        result: fallbackAgenticResult(parsed.incident, parsed.mlResult, parsed.agentControls),
        provider: "sample",
        runtime: "deterministic-fallback",
        setupRequired: false,
        message: "OpenAI Agents SDK request timed out. Deterministic governed fallback planner used for this run."
      };
    }
    return {
      result: fallbackAgenticResult(parsed.incident, parsed.mlResult, parsed.agentControls),
      provider: "sample",
      runtime: "deterministic-fallback",
      setupRequired: false,
      message: `OpenAI Agents SDK request failed. Deterministic governed fallback planner used for this run. ${message}`
    };
  } finally {
    clearTimeout(timeout);
  }
}
