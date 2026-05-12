import { errorResponse } from "@/lib/errors";
import { logError, logInfo, logWarn } from "@/lib/logger";
import { estimateImageMetadata } from "@/lib/vision/image-utils";
import { runRoboflowInference, visionRequestSchema } from "@/lib/vision/roboflow";

export async function POST(request: Request) {
  const start = Date.now();
  const route = "/api/vision/roboflow";
  try {
    logInfo({ route, event: "request_start", status: "pending" });
    const body = await request.json();
    const parsed = visionRequestSchema.parse(body);
    const metadata = estimateImageMetadata(parsed.imageBase64);
    logInfo({ route, incidentId: parsed.incidentId, event: "validation_success", status: "success", metadata });
    const result = await runRoboflowInference(parsed);
    if (result.provider === "sample") {
      logWarn({
        route,
        incidentId: parsed.incidentId,
        event: "fallback_mode",
        status: "success",
        latencyMs: Date.now() - start,
        metadata
      });
    }
    logInfo({
      route,
      incidentId: parsed.incidentId,
      event: "response_success",
      status: "success",
      latencyMs: Date.now() - start,
      metadata: { provider: result.provider, detections: result.detections.length }
    });
    return Response.json({ ok: true, result });
  } catch (error) {
    logError({ route, event: "error", status: "failed", latencyMs: Date.now() - start, error });
    return errorResponse(error);
  }
}
