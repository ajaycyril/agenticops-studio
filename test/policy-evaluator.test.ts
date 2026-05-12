import { describe, expect, it } from "vitest";
import { evaluatePolicy } from "@/lib/policies/policy-evaluator";
import { clonePreset } from "@/lib/scenario-presets";

describe("policy evaluator", () => {
  it("blocks drone dispatch when drone is unavailable", () => {
    const incident = clonePreset(4);
    const result = evaluatePolicy("dispatch_drone", incident, { fireProbability: 0.9, riskLevel: "critical" });
    expect(result.blocked).toBe(true);
  });

  it("requires approval for gate unlock", () => {
    const incident = clonePreset(1);
    const result = evaluatePolicy("unlock_gate", incident, { fireProbability: 0.88, riskLevel: "critical" });
    expect(result.allowed).toBe(true);
    expect(result.requiresHumanApproval).toBe(true);
  });
});
