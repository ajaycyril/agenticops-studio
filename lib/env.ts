import { z } from "zod";

const envSchema = z.object({
  OPENAI_API_KEY: z.string().optional(),
  OPENAI_MODEL: z.string().default("gpt-5.5"),
  OPENAI_MAX_OUTPUT_TOKENS: z.coerce.number().int().min(200).max(6000).default(3200),
  OPENAI_TIMEOUT_MS: z.coerce.number().int().min(3000).max(60000).default(45000),
  OPENAI_MAX_AGENT_CALLS_PER_RUN: z.coerce.number().int().min(1).max(6).default(4),
  ROBOFLOW_API_KEY: z.string().optional(),
  ROBOFLOW_MODEL_ID: z.string().default("fire-and-smoke-detection-hiwia/2"),
  ROBOFLOW_API_URL: z.string().url().default("https://serverless.roboflow.com"),
  NEXT_PUBLIC_APP_NAME: z.string().default("AgenticOps Studio"),
  NEXT_PUBLIC_DEMO_MODE: z.string().default("true")
});

export const env = envSchema.parse({
  OPENAI_API_KEY: process.env.OPENAI_API_KEY,
  OPENAI_MODEL: process.env.OPENAI_MODEL,
  OPENAI_MAX_OUTPUT_TOKENS: process.env.OPENAI_MAX_OUTPUT_TOKENS,
  OPENAI_TIMEOUT_MS: process.env.OPENAI_TIMEOUT_MS,
  OPENAI_MAX_AGENT_CALLS_PER_RUN: process.env.OPENAI_MAX_AGENT_CALLS_PER_RUN,
  ROBOFLOW_API_KEY: process.env.ROBOFLOW_API_KEY,
  ROBOFLOW_MODEL_ID: process.env.ROBOFLOW_MODEL_ID,
  ROBOFLOW_API_URL: process.env.ROBOFLOW_API_URL,
  NEXT_PUBLIC_APP_NAME: process.env.NEXT_PUBLIC_APP_NAME,
  NEXT_PUBLIC_DEMO_MODE: process.env.NEXT_PUBLIC_DEMO_MODE
});

export const isOpenAIConfigured = () => Boolean(env.OPENAI_API_KEY);
export const isRoboflowConfigured = () => Boolean(env.ROBOFLOW_API_KEY);
