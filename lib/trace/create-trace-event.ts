import type { TraceEvent } from "@/lib/types";

export function createTraceEvent(event: Omit<TraceEvent, "id" | "timestamp">): TraceEvent {
  return {
    id: crypto.randomUUID(),
    timestamp: new Date().toISOString(),
    ...event
  };
}
