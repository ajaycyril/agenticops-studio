import { APP_NAME } from "@/lib/constants";
import { isOpenAIConfigured, isRoboflowConfigured } from "@/lib/env";

export async function GET() {
  return Response.json({
    status: "ok",
    app: APP_NAME,
    openaiConfigured: isOpenAIConfigured(),
    roboflowConfigured: isRoboflowConfigured(),
    timestamp: new Date().toISOString()
  });
}
