import { AppShell } from "@/components/layout/AppShell";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";

const items = [
  "Kafka/Event Hubs event streaming",
  "MQTT device ingestion",
  "YOLO/ONNX/Qualcomm edge model deployment",
  "MLflow model registry",
  "OPA/Rego policy service",
  "OpenTelemetry, Grafana, Tempo, Loki",
  "Postgres event store",
  "Vector DB for SOP retrieval",
  "Kubernetes and enterprise identity/RBAC",
  "Immutable audit logs and governed authority APIs"
];

export default function EnterprisePage() {
  return (
    <AppShell>
      <Card>
        <CardHeader>
          <CardTitle className="text-3xl">Enterprise Extension Blueprint</CardTitle>
          <CardDescription>
            The Vercel app is the interactive reference demo. The enterprise extension is the production blueprint for regulated physical-world
            agentic AI deployment.
          </CardDescription>
        </CardHeader>
        <CardContent className="grid gap-3 md:grid-cols-2">
          {items.map((item) => (
            <div key={item} className="rounded-md border border-white/10 bg-black/20 p-4 text-sm text-slate-300">{item}</div>
          ))}
        </CardContent>
      </Card>
    </AppShell>
  );
}
