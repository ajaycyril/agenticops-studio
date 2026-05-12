import { Card, CardContent } from "@/components/ui/card";

export function MetricCard({ label, value, tone = "cyan" }: { label: string; value: string | number; tone?: "cyan" | "amber" | "red" }) {
  const color = tone === "red" ? "text-red-200" : tone === "amber" ? "text-amber-200" : "text-cyan-100";
  return (
    <Card className="shadow-none">
      <CardContent className="p-4">
        <div className="text-xs uppercase tracking-wide text-slate-500">{label}</div>
        <div className={`mt-2 text-2xl font-semibold ${color}`}>{value}</div>
      </CardContent>
    </Card>
  );
}
