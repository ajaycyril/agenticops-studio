import { Progress } from "@/components/ui/progress";

export function ConfidenceMeter({ label, value }: { label: string; value: number }) {
  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between text-xs text-slate-400">
        <span>{label}</span>
        <span>{Math.round(value * 100)}%</span>
      </div>
      <Progress value={value * 100} />
    </div>
  );
}
