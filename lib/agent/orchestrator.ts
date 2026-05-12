import OpenAI from "openai";
import { env, isOpenAIConfigured } from "@/lib/env";
import { agenticJsonSchema, agenticResultSchema } from "@/lib/agent/schemas";
import { agentSystemPrompt, buildAgentUserPrompt } from "@/lib/agent/prompts";
import type { AgenticResult, IncidentState, MLResult } from "@/lib/types";

export function fallbackAgenticResult(incident: IncidentState, mlResult: Pick<MLResult, "fireProbability" | "riskLevel">): AgenticResult {
  const confidence = Math.max(incident.cameraSmokeConfidence, incident.cameraFireConfidence);
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

  if (confidence < 0.5) {
    proposedActions.unshift({
      action: "request_camera_recheck",
      riskClass: "medium",
      reason: "Camera confidence is low, so the visual evidence should be revalidated before stronger action.",
      evidenceUsed: [`Max camera confidence ${confidence.toFixed(2)}`],
      confidence: 0.72,
      requiresPolicyCheck: true,
      requiresHumanApproval: false
    });
  }

  if (incident.droneAvailable && mlResult.riskLevel !== "low") {
    proposedActions.push({
      action: "dispatch_drone",
      riskClass: mlResult.riskLevel,
      reason: "Drone reconnaissance can gather more evidence while keeping physical action sandboxed.",
      evidenceUsed: [`Risk level ${mlResult.riskLevel}`, `Drone available ${incident.droneAvailable}`, `Wind ${incident.windSpeedKmh} km/h`],
      confidence: 0.74,
      requiresPolicyCheck: true,
      requiresHumanApproval: incident.windSpeedKmh > 25 || confidence < 0.7
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

  if (mlResult.riskLevel === "critical") {
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
      "The LLM is not the control system. It proposes a governed plan; policy, tool permissions, human approvals, and audit records control execution."
  };
}

export async function runAgenticOrchestrator(payload: unknown) {
  const parsed = payload as { incident: IncidentState; mlResult: MLResult };
  if (!isOpenAIConfigured()) {
    return { result: fallbackAgenticResult(parsed.incident, parsed.mlResult), provider: "sample" as const, setupRequired: true };
  }

  const client = new OpenAI({ apiKey: env.OPENAI_API_KEY });
  const response = await client.responses.create({
    model: env.OPENAI_MODEL,
    input: [
      { role: "system", content: agentSystemPrompt },
      { role: "user", content: buildAgentUserPrompt(payload) }
    ],
    text: {
      format: {
        type: "json_schema",
        name: "AgenticResult",
        schema: agenticJsonSchema,
        strict: true
      }
    }
  });

  const firstPass = agenticResultSchema.safeParse(JSON.parse(response.output_text));
  if (firstPass.success) return { result: firstPass.data, provider: "openai" as const, setupRequired: false };

  const repair = await client.responses.create({
    model: env.OPENAI_MODEL,
    input: [
      { role: "system", content: `${agentSystemPrompt}\nRepair the previous invalid JSON so it exactly matches the required schema.` },
      { role: "user", content: response.output_text }
    ],
    text: {
      format: {
        type: "json_schema",
        name: "AgenticResult",
        schema: agenticJsonSchema,
        strict: true
      }
    }
  });

  const repaired = agenticResultSchema.parse(JSON.parse(repair.output_text));
  return { result: repaired, provider: "openai" as const, setupRequired: false };
}
