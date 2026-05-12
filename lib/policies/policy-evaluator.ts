import type { ActionName, IncidentState, MLResult, PolicyDecision, RiskLevel } from "@/lib/types";

function isHighOrCritical(level: RiskLevel) {
  return level === "high" || level === "critical";
}

function cameraConfidence(incident: IncidentState) {
  return Math.max(incident.cameraSmokeConfidence, incident.cameraFireConfidence);
}

export function evaluatePolicy(
  action: ActionName,
  incident: IncidentState,
  mlResult: Pick<MLResult, "fireProbability" | "riskLevel">
): PolicyDecision {
  const confidence = cameraConfidence(incident);

  if (action === "dispatch_drone") {
    const evidenceEnough =
      mlResult.fireProbability >= 0.65 ||
      incident.cameraSmokeConfidence >= 0.7 ||
      incident.cameraFireConfidence >= 0.55;
    const allowed = incident.droneAvailable && evidenceEnough && incident.windSpeedKmh <= 35;
    const requiresHumanApproval =
      allowed &&
      (incident.windSpeedKmh > 25 || confidence < 0.7 || (incident.occupancyStatus === "unknown" && isHighOrCritical(mlResult.riskLevel)));

    return {
      action,
      allowed,
      requiresHumanApproval,
      blocked: !allowed,
      reason: allowed
        ? requiresHumanApproval
          ? "Drone dispatch is allowed but approval-gated by operating conditions or uncertainty."
          : "Drone dispatch is allowed in sandbox mode because evidence and weather checks passed."
        : "Drone dispatch is blocked because availability, evidence, or wind-speed policy failed.",
      policyIds: ["POL-DRONE-ALLOW", "POL-DRONE-APPROVAL"]
    };
  }

  if (action === "unlock_gate") {
    const allowed = incident.gateLocked && isHighOrCritical(mlResult.riskLevel);
    return {
      action,
      allowed,
      requiresHumanApproval: true,
      blocked: !allowed,
      reason: !incident.gateLocked
        ? "Gate is already unlocked; no physical action is needed."
        : allowed
          ? "Gate unlock may be proposed for high or critical risk but always needs human approval."
          : "Gate unlock is blocked below high risk.",
      policyIds: ["POL-GATE-APPROVAL"]
    };
  }

  if (action === "notify_authority") {
    const allowed = mlResult.riskLevel === "critical" || mlResult.riskLevel === "high";
    return {
      action,
      allowed,
      requiresHumanApproval: mlResult.riskLevel === "high",
      blocked: !allowed,
      reason:
        mlResult.riskLevel === "critical"
          ? "Critical risk allows authority notification."
          : mlResult.riskLevel === "high"
            ? "High risk authority notification is approval-gated."
            : "Authority notification is blocked for low or medium risk unless explicitly approved.",
      policyIds: ["POL-AUTHORITY"]
    };
  }

  if (action === "request_camera_recheck") {
    const allowed = confidence < 0.5;
    return {
      action,
      allowed,
      requiresHumanApproval: false,
      blocked: !allowed,
      reason: allowed ? "Camera confidence is low, so revalidation is allowed." : "Camera recheck is not required by confidence policy.",
      policyIds: ["POL-RECHECK"]
    };
  }

  if (action === "notify_operator") {
    const allowed = mlResult.riskLevel !== "low";
    return {
      action,
      allowed,
      requiresHumanApproval: false,
      blocked: !allowed,
      reason: allowed ? "Operator notification is allowed for medium, high, or critical risk." : "Operator notification is not required at low risk.",
      policyIds: ["POL-OPERATOR"]
    };
  }

  return {
    action,
    allowed: true,
    requiresHumanApproval: false,
    blocked: false,
    reason: "Reporting and decision-record actions are allowed after an incident run.",
    policyIds: ["POL-REPORTING"]
  };
}

export function evaluatePoliciesForActions(
  actions: ActionName[],
  incident: IncidentState,
  mlResult: Pick<MLResult, "fireProbability" | "riskLevel">
) {
  return actions.map((action) => evaluatePolicy(action, incident, mlResult));
}
