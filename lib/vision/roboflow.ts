import { z } from "zod";
import { env, isRoboflowConfigured } from "@/lib/env";
import { AppError } from "@/lib/errors";
import { stripDataUrl } from "@/lib/vision/image-utils";
import { getSamplePrediction } from "@/lib/vision/sample-predictions";
import type { VisionDetection, VisionResult } from "@/lib/types";

export const visionRequestSchema = z.object({
  incidentId: z.string(),
  sampleName: z.string().optional(),
  imageBase64: z.string().optional()
});

type RoboflowPrediction = {
  class?: string;
  confidence?: number;
  x?: number;
  y?: number;
  width?: number;
  height?: number;
};

function normalizeClass(value: string | undefined): VisionDetection["className"] {
  const lower = (value ?? "unknown").toLowerCase();
  if (lower.includes("smoke")) return "smoke";
  if (lower.includes("fire") || lower.includes("flame")) return "fire";
  if (lower.includes("person")) return "person";
  return "unknown";
}

export async function runRoboflowInference(input: z.infer<typeof visionRequestSchema>): Promise<VisionResult> {
  const start = Date.now();
  if (!isRoboflowConfigured() || !input.imageBase64) {
    return getSamplePrediction(input.sampleName, Date.now() - start);
  }

  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 12000);

  try {
    const url = `${env.ROBOFLOW_API_URL}/${env.ROBOFLOW_MODEL_ID}?api_key=${env.ROBOFLOW_API_KEY}`;
    const response = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/x-www-form-urlencoded" },
      body: stripDataUrl(input.imageBase64),
      signal: controller.signal
    });

    if (!response.ok) {
      throw new AppError({
        code: "ROBOFLOW_REQUEST_FAILED",
        message: "Roboflow inference failed. The app is still usable in sample mode.",
        recoverable: true,
        status: 502
      });
    }

    const data = (await response.json()) as { predictions?: RoboflowPrediction[] };
    const detections = (data.predictions ?? []).map((prediction) => ({
      className: normalizeClass(prediction.class),
      confidence: Number(prediction.confidence ?? 0),
      x: prediction.x,
      y: prediction.y,
      width: prediction.width,
      height: prediction.height
    }));

    const maxSmokeConfidence = Math.max(0, ...detections.filter((d) => d.className === "smoke").map((d) => d.confidence));
    const maxFireConfidence = Math.max(0, ...detections.filter((d) => d.className === "fire").map((d) => d.confidence));

    return {
      provider: "roboflow",
      modelId: env.ROBOFLOW_MODEL_ID,
      detections,
      maxSmokeConfidence,
      maxFireConfidence,
      latencyMs: Date.now() - start
    };
  } finally {
    clearTimeout(timeout);
  }
}
