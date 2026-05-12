import { AGENT_VERSION, DEFAULT_MODEL_VERSION, POLICY_VERSION } from "@/lib/constants";
import { env } from "@/lib/env";
import type {
  AgenticResult,
  DecisionRecord,
  IncidentState,
  MLResult,
  PolicyDecision,
  RuleResult,
  TraceEvent,
  VisionResult
} from "@/lib/types";

export function buildDecisionRecord(params: {
  runId: string;
  incident: IncidentState;
  ruleResult: RuleResult;
  mlResult: MLResult;
  visionResult: VisionResult;
  agenticResult: AgenticResult;
  policyDecisions: PolicyDecision[];
  trace: TraceEvent[];
}): DecisionRecord {
  const actionsApproved = params.agenticResult.proposedActions
    .filter((proposal) => {
      if (proposal.action === "unlock_gate") return params.incident.humanApproval.unlockGate;
      if (proposal.action === "dispatch_drone") return params.incident.humanApproval.dispatchDrone;
      if (proposal.action === "notify_authority") return params.incident.humanApproval.notifyAuthority;
      return false;
    })
    .map((proposal) => ({
      action: proposal.action,
      approvedBy: "demo-operator" as const,
      approvedAt: new Date().toISOString()
    }));

  const actionsBlocked = params.policyDecisions
    .filter((decision) => decision.blocked)
    .map((decision) => ({ action: decision.action, reason: decision.reason, policyIds: decision.policyIds }));

  const actionsExecuted = params.policyDecisions.map((decision) => {
    const approvalKey =
      decision.action === "unlock_gate"
        ? "unlockGate"
        : decision.action === "dispatch_drone"
          ? "dispatchDrone"
          : decision.action === "notify_authority"
            ? "notifyAuthority"
            : undefined;
    const hasApproval = approvalKey ? params.incident.humanApproval[approvalKey] : true;
    const status: "success" | "blocked" | "skipped" =
      decision.allowed && (!decision.requiresHumanApproval || hasApproval) ? "success" : decision.blocked ? "blocked" : "skipped";
    return {
      action: decision.action,
      executionMode: decision.action === "generate_report" || decision.action === "write_decision_record" ? ("real" as const) : ("sandbox" as const),
      status,
      result:
        status === "success"
          ? `${decision.action} completed in demo ${decision.action.includes("gate") || decision.action.includes("drone") ? "sandbox" : "workflow"} mode.`
          : decision.reason
    };
  });

  return {
    incidentId: params.incident.incidentId,
    runId: params.runId,
    createdAt: new Date().toISOString(),
    scenarioName: params.incident.scenarioName,
    inputs: params.incident,
    ruleResult: params.ruleResult,
    mlResult: params.mlResult,
    visionResult: params.visionResult,
    agenticResult: params.agenticResult,
    policyDecisions: params.policyDecisions,
    actionsProposed: params.agenticResult.proposedActions,
    actionsApproved,
    actionsBlocked,
    actionsExecuted,
    governance: {
      policyVersion: POLICY_VERSION,
      activeModelVersion: params.mlResult.modelVersion || DEFAULT_MODEL_VERSION,
      visionProvider: params.visionResult.provider,
      llmModel: env.OPENAI_MODEL,
      agentVersion: AGENT_VERSION
    },
    trace: params.trace
  };
}
