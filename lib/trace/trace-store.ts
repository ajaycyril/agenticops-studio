"use client";

import type { TraceEvent } from "@/lib/types";

const STORAGE_KEY = "agenticops.trace.v1";

export function loadTrace(): TraceEvent[] {
  if (typeof window === "undefined") return [];
  const raw = sessionStorage.getItem(STORAGE_KEY);
  return raw ? (JSON.parse(raw) as TraceEvent[]) : [];
}

export function saveTrace(events: TraceEvent[]) {
  if (typeof window === "undefined") return;
  sessionStorage.setItem(STORAGE_KEY, JSON.stringify(events.slice(-200)));
}
