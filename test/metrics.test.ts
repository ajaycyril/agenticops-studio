import { describe, expect, it } from "vitest";
import { classificationMetrics } from "@/lib/ml/metrics";

describe("classification metrics", () => {
  it("calculates accuracy and false positive rate", () => {
    const metrics = classificationMetrics([1, 0, 1, 0], [0.9, 0.8, 0.7, 0.1]);
    expect(metrics.accuracy).toBe(0.75);
    expect(metrics.falsePositiveRate).toBe(0.5);
  });
});
