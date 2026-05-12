import type { DecisionRecord } from "@/lib/types";

export function summarizeDecisionRecord(record: DecisionRecord, audience: "executive" | "technical") {
  const actions = record.actionsExecuted.map((action) => `${action.action}: ${action.status}`).join(", ");
  if (audience === "executive") {
    return `Incident ${record.incidentId} was assessed as ${record.agenticResult.riskAssessment.level}. The governed control plane proposed ${record.actionsProposed.length} actions, applied ${record.policyDecisions.length} policy checks, and recorded execution outcomes: ${actions}.`;
  }

  return `Run ${record.runId} used policy ${record.governance.policyVersion}, model ${record.governance.activeModelVersion}, vision provider ${record.governance.visionProvider}, and LLM model ${record.governance.llmModel}. Trace event count: ${record.trace.length}. Executions: ${actions}.`;
}
