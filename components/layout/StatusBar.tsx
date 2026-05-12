import { Badge } from "@/components/ui/badge";

export function StatusBar() {
  return (
    <div className="border-t border-white/10 bg-slate-950/70 px-4 py-2 text-xs text-slate-400">
      <div className="mx-auto flex max-w-7xl flex-wrap items-center justify-between gap-2">
        <span>Demo mode supports missing OpenAI and Roboflow credentials with graceful fallback outputs.</span>
        <div className="flex gap-2">
          <Badge>OpenAI server route</Badge>
          <Badge>Roboflow server route</Badge>
          <Badge>TF.js browser ML</Badge>
        </div>
      </div>
    </div>
  );
}
