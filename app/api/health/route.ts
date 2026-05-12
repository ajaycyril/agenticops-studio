import { APP_NAME } from "@/lib/constants";
import { env, isOpenAIConfigured, isRoboflowConfigured } from "@/lib/env";

export async function GET() {
  return Response.json({
    status: "ok",
    app: APP_NAME,
    openaiConfigured: isOpenAIConfigured(),
    roboflowConfigured: isRoboflowConfigured(),
    openaiModel: env.OPENAI_MODEL,
    openaiMaxOutputTokens: env.OPENAI_MAX_OUTPUT_TOKENS,
    openaiTimeoutMs: env.OPENAI_TIMEOUT_MS,
    openaiMaxAgentCallsPerRun: env.OPENAI_MAX_AGENT_CALLS_PER_RUN,
    timestamp: new Date().toISOString()
  });
}
