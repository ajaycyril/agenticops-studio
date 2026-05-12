import { agenticResultSchema } from "@/lib/agent/schemas";
import type { AgenticResult } from "@/lib/types";

export function normalizeAgentOutput(output: unknown): AgenticResult {
  if (typeof output === "string") {
    return agenticResultSchema.parse(JSON.parse(output));
  }
  return agenticResultSchema.parse(output);
}
