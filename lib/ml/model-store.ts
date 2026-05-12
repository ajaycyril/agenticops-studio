"use client";

const STORAGE_KEY = "agenticops.model.version";

export function saveModelVersion(version: string) {
  sessionStorage.setItem(STORAGE_KEY, version);
}

export function loadModelVersion() {
  if (typeof window === "undefined") return undefined;
  return sessionStorage.getItem(STORAGE_KEY) ?? undefined;
}
