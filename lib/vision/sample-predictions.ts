import type { VisionResult } from "@/lib/types";

export const sampleVisionPredictions: Record<string, Omit<VisionResult, "latencyMs">> = {
  "cooking-smoke": {
    provider: "sample",
    modelId: "fire-and-smoke-detection-hiwia/2",
    detections: [{ className: "smoke", confidence: 0.35, x: 35, y: 28, width: 32, height: 22 }],
    maxSmokeConfidence: 0.35,
    maxFireConfidence: 0.05,
    setupRequired: true,
    message: "Add ROBOFLOW_API_KEY and ROBOFLOW_MODEL_ID to enable live hosted inference."
  },
  "fire-smoke-room": {
    provider: "sample",
    modelId: "fire-and-smoke-detection-hiwia/2",
    detections: [
      { className: "smoke", confidence: 0.88, x: 22, y: 18, width: 56, height: 38 },
      { className: "fire", confidence: 0.74, x: 45, y: 55, width: 28, height: 24 }
    ],
    maxSmokeConfidence: 0.88,
    maxFireConfidence: 0.74,
    setupRequired: true,
    message: "Add ROBOFLOW_API_KEY and ROBOFLOW_MODEL_ID to enable live hosted inference."
  },
  "unclear-camera": {
    provider: "sample",
    modelId: "fire-and-smoke-detection-hiwia/2",
    detections: [{ className: "unknown", confidence: 0.42, x: 30, y: 30, width: 42, height: 30 }],
    maxSmokeConfidence: 0.42,
    maxFireConfidence: 0.18,
    setupRequired: true,
    message: "Add ROBOFLOW_API_KEY and ROBOFLOW_MODEL_ID to enable live hosted inference."
  }
};

export function getSamplePrediction(sampleName = "unclear-camera", latencyMs = 18): VisionResult {
  return { ...(sampleVisionPredictions[sampleName] ?? sampleVisionPredictions["unclear-camera"]), latencyMs };
}
