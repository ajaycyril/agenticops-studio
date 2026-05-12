import { describe, expect, it } from "vitest";
import { agenticResultSchema } from "@/lib/agent/schemas";
import { fallbackAgenticResult } from "@/lib/agent/orchestrator";
import { heuristicRiskPrediction } from "@/lib/ml/predict-risk";
import { clonePreset } from "@/lib/scenario-presets";

describe("agent schema", () => {
  it("validates fallback structured output", () => {
    const incident = clonePreset(1);
    const output = fallbackAgenticResult(incident, heuristicRiskPrediction(incident));
    expect(agenticResultSchema.safeParse(output).success).toBe(true);
  });
});
