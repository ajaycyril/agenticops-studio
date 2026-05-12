import { incidentRequestSchema } from "@/lib/agent/schemas";
import { runAgenticOrchestrator } from "@/lib/agent/orchestrator";
import { getToolRegistry } from "@/lib/agent/tool-registry";
import { errorResponse } from "@/lib/errors";
import { logError, logInfo, logWarn } from "@/lib/logger";

export async function POST(request: Request) {
  const route = "/api/agent/run";
  const start = Date.now();
  try {
    logInfo({ route, event: "request_start", status: "pending" });
    const body = await request.json();
    const parsed = incidentRequestSchema.parse({ ...body, toolRegistry: body.toolRegistry ?? getToolRegistry() });
    logInfo({ route, incidentId: parsed.incident.incidentId, event: "validation_success", status: "success" });
    const response = await runAgenticOrchestrator(parsed);
    if (response.provider === "sample") {
      logWarn({
        route,
        incidentId: parsed.incident.incidentId,
        event: "fallback_mode",
        status: "success",
        latencyMs: Date.now() - start
      });
    }
    logInfo({
      route,
      incidentId: parsed.incident.incidentId,
      event: "response_success",
      status: "success",
      latencyMs: Date.now() - start,
      metadata: { provider: response.provider, proposedActions: response.result.proposedActions.length }
    });
    return Response.json({ ok: true, ...response });
  } catch (error) {
    logError({ route, event: "error", status: "failed", latencyMs: Date.now() - start, error });
    return errorResponse(error);
  }
}
