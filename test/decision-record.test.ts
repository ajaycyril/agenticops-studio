import { describe, expect, it } from "vitest";
import { fallbackAgenticResult } from "@/lib/agent/orchestrator";
import { buildDecisionRecord } from "@/lib/decision-record/build-decision-record";
import { heuristicRiskPrediction } from "@/lib/ml/predict-risk";
import { evaluatePoliciesForActions } from "@/lib/policies/policy-evaluator";
import { evaluateRules } from "@/lib/rule/rule-engine";
import { clonePreset } from "@/lib/scenario-presets";
import { getSamplePrediction } from "@/lib/vision/sample-predictions";

describe("decision record", () => {
  it("builds an auditable run artifact", () => {
    const incident = clonePreset(1);
    const mlResult = heuristicRiskPrediction(incident);
    const agenticResult = fallbackAgenticResult(incident, mlResult);
    const policyDecisions = evaluatePoliciesForActions(agenticResult.proposedActions.map((action) => action.action), incident, mlResult);
    const record = buildDecisionRecord({
      runId: "RUN-TEST",
      incident,
      ruleResult: evaluateRules(incident),
      mlResult,
      visionResult: getSamplePrediction("fire-smoke-room"),
      agenticResult,
      policyDecisions,
      trace: []
    });
    expect(record.runId).toBe("RUN-TEST");
    expect(record.actionsProposed.length).toBeGreaterThan(0);
    expect(record.governance.policyVersion).toContain("fire-response-policy");
  });
});
