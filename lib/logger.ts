import pino from "pino";

const logger = pino({
  level: process.env.LOG_LEVEL ?? "info",
  redact: {
    paths: ["OPENAI_API_KEY", "ROBOFLOW_API_KEY", "*.apiKey", "*.authorization", "*.base64", "*.imageBase64"],
    censor: "[redacted]"
  }
});

export type LogFields = {
  route: string;
  incidentId?: string;
  runId?: string;
  event: string;
  latencyMs?: number;
  status?: string;
  errorCode?: string;
  metadata?: Record<string, unknown>;
};

export function logInfo(fields: LogFields) {
  logger.info({ timestamp: new Date().toISOString(), level: "info", ...fields });
}

export function logWarn(fields: LogFields) {
  logger.warn({ timestamp: new Date().toISOString(), level: "warn", ...fields });
}

export function logError(fields: LogFields & { error?: unknown }) {
  logger.error({ timestamp: new Date().toISOString(), level: "error", ...fields });
}
