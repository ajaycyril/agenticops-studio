import { z } from "zod";

export const riskLevelSchema = z.enum(["low", "medium", "high", "critical"]);
export const actionNameSchema = z.enum([
  "request_camera_recheck",
  "dispatch_drone",
  "unlock_gate",
  "notify_operator",
  "notify_authority",
  "generate_report",
  "write_decision_record"
]);

export const agenticResultSchema = z.object({
  incidentSummary: z.string(),
  riskAssessment: z.object({
    level: riskLevelSchema,
    reason: z.string(),
    evidence: z.array(z.string()),
    uncertainty: z.array(z.string())
  }),
  proposedActions: z.array(
    z.object({
      action: actionNameSchema,
      riskClass: riskLevelSchema,
      reason: z.string(),
      evidenceUsed: z.array(z.string()),
      confidence: z.number().min(0).max(1),
      requiresPolicyCheck: z.boolean(),
      requiresHumanApproval: z.boolean()
    })
  ),
  questionsForHuman: z.array(z.string()),
  sopReferences: z.array(z.string()),
  explanation: z.string()
});

export const incidentRequestSchema = z.object({
  incident: z.object({
    incidentId: z.string(),
    siteId: z.string(),
    scenarioName: z.string(),
    smokePpm: z.number(),
    temperatureC: z.number(),
    cameraSmokeConfidence: z.number(),
    cameraFireConfidence: z.number(),
    occupancyStatus: z.enum(["none", "detected", "unknown"]),
    droneAvailable: z.boolean(),
    gateLocked: z.boolean(),
    sensorHealth: z.number(),
    historicalFalseAlarmRate: z.number(),
    windSpeedKmh: z.number(),
    imageUrl: z.string().optional(),
    visionProvider: z.enum(["roboflow", "sample", "manual"]),
    visionDetections: z.array(
      z.object({
        className: z.enum(["smoke", "fire", "person", "unknown"]),
        confidence: z.number(),
        x: z.number().optional(),
        y: z.number().optional(),
        width: z.number().optional(),
        height: z.number().optional()
      })
    ),
    humanApproval: z.object({
      unlockGate: z.boolean(),
      dispatchDrone: z.boolean(),
      notifyAuthority: z.boolean()
    })
  }),
  ruleResult: z.unknown(),
  mlResult: z.object({
    fireProbability: z.number(),
    riskLevel: riskLevelSchema,
    modelVersion: z.string(),
    metrics: z
      .object({
        accuracy: z.number(),
        precision: z.number(),
        recall: z.number(),
        falsePositiveRate: z.number()
      })
      .optional(),
    featureImportance: z.array(z.object({ feature: z.string(), importance: z.number() })),
    explanation: z.string()
  }),
  visionResult: z.unknown().optional(),
  policySummary: z.string().optional(),
  agentControls: z
    .object({
      operatingMode: z.enum(["balanced", "conservative", "rapid_response"]).default("balanced"),
      authorityPosture: z.enum(["strict", "approval_gated", "critical_only"]).default("critical_only"),
      operatorInstruction: z.string().max(500).optional()
    })
    .optional(),
  toolRegistry: z.unknown().optional()
});

export const agenticJsonSchema = {
  type: "object",
  additionalProperties: false,
  required: ["incidentSummary", "riskAssessment", "proposedActions", "questionsForHuman", "sopReferences", "explanation"],
  properties: {
    incidentSummary: { type: "string" },
    riskAssessment: {
      type: "object",
      additionalProperties: false,
      required: ["level", "reason", "evidence", "uncertainty"],
      properties: {
        level: { enum: ["low", "medium", "high", "critical"], type: "string" },
        reason: { type: "string" },
        evidence: { type: "array", items: { type: "string" } },
        uncertainty: { type: "array", items: { type: "string" } }
      }
    },
    proposedActions: {
      type: "array",
      items: {
        type: "object",
        additionalProperties: false,
        required: [
          "action",
          "riskClass",
          "reason",
          "evidenceUsed",
          "confidence",
          "requiresPolicyCheck",
          "requiresHumanApproval"
        ],
        properties: {
          action: {
            enum: [
              "request_camera_recheck",
              "dispatch_drone",
              "unlock_gate",
              "notify_operator",
              "notify_authority",
              "generate_report",
              "write_decision_record"
            ],
            type: "string"
          },
          riskClass: { enum: ["low", "medium", "high", "critical"], type: "string" },
          reason: { type: "string" },
          evidenceUsed: { type: "array", items: { type: "string" } },
          confidence: { type: "number" },
          requiresPolicyCheck: { type: "boolean" },
          requiresHumanApproval: { type: "boolean" }
        }
      }
    },
    questionsForHuman: { type: "array", items: { type: "string" } },
    sopReferences: { type: "array", items: { type: "string" } },
    explanation: { type: "string" }
  }
} as const;
