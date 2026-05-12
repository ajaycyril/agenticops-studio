import { describe, expect, it } from "vitest";
import { evaluateRules } from "@/lib/rule/rule-engine";
import { clonePreset } from "@/lib/scenario-presets";

describe("rule engine", () => {
  it("marks confirmed fire as critical", () => {
    const result = evaluateRules(clonePreset(1));
    expect(result.triggered).toBe(true);
    expect(result.severity).toBe("critical");
    expect(result.action).toBe("escalate_operator");
  });

  it("does not use camera confidence for deterministic rules", () => {
    const incident = clonePreset(0);
    incident.cameraFireConfidence = 1;
    const result = evaluateRules(incident);
    expect(result.severity).toBe("medium");
  });
});
