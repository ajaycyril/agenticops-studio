import OpenAI from "openai";
import { z } from "zod";
import { env, isOpenAIConfigured } from "@/lib/env";
import { errorResponse } from "@/lib/errors";
import { logError, logInfo, logWarn } from "@/lib/logger";

const reportSchema = z.object({
  record: z.unknown(),
  audience: z.enum(["executive", "technical"]).default("executive")
});

function fallbackReport(record: { incidentId?: string; agenticResult?: { riskAssessment?: { level?: string } } }, audience: string) {
  const level = record.agenticResult?.riskAssessment?.level ?? "unknown";
  return audience === "executive"
    ? `Executive report: Incident ${record.incidentId ?? "unknown"} reached ${level} risk. The workflow demonstrates governed execution where models reason, policy constrains, humans approve, and every decision is recorded.`
    : `Technical report: The run includes incident inputs, rule result, ML prediction, vision output, agent proposals, policy decisions, sandbox tool executions, and trace events. No physical action leaves demo sandbox mode.`;
}

export async function POST(request: Request) {
  const route = "/api/agent/report";
  const start = Date.now();
  try {
    logInfo({ route, event: "request_start", status: "pending" });
    const parsed = reportSchema.parse(await request.json());
    const record = parsed.record as { incidentId?: string; agenticResult?: { riskAssessment?: { level?: string } } };
    if (!isOpenAIConfigured()) {
      logWarn({ route, event: "fallback_mode", status: "success", latencyMs: Date.now() - start });
      return Response.json({ ok: true, provider: "sample", setupRequired: true, report: fallbackReport(record, parsed.audience) });
    }

    const client = new OpenAI({ apiKey: env.OPENAI_API_KEY });
    const response = await client.responses.create({
      model: env.OPENAI_MODEL,
      input: `Write a concise ${parsed.audience} incident report for this governed Physical AI workflow. Do not invent real emergency dispatch. Record: ${JSON.stringify(parsed.record).slice(0, 12000)}`
    });
    logInfo({ route, event: "response_success", status: "success", latencyMs: Date.now() - start });
    return Response.json({ ok: true, provider: "openai", setupRequired: false, report: response.output_text });
  } catch (error) {
    logError({ route, event: "error", status: "failed", latencyMs: Date.now() - start, error });
    return errorResponse(error);
  }
}
