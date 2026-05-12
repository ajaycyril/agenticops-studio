import { z } from "zod";
import { errorResponse } from "@/lib/errors";
import { logError, logInfo } from "@/lib/logger";

const clientEventSchema = z.object({
  incidentId: z.string().optional(),
  event: z.string(),
  status: z.string().optional(),
  metadata: z.record(z.string(), z.unknown()).optional()
});

export async function POST(request: Request) {
  const route = "/api/logs/client-event";
  try {
    const parsed = clientEventSchema.parse(await request.json());
    logInfo({ route, incidentId: parsed.incidentId, event: parsed.event, status: parsed.status ?? "success", metadata: parsed.metadata });
    return Response.json({ ok: true });
  } catch (error) {
    logError({ route, event: "error", status: "failed", error });
    return errorResponse(error);
  }
}
