import { Button } from "@/components/ui/button";

export function JsonInspector({ value, title = "JSON" }: { value: unknown; title?: string }) {
  const text = JSON.stringify(value, null, 2);
  return (
    <div className="overflow-hidden rounded-lg border border-white/10 bg-slate-950/70">
      <div className="flex items-center justify-between border-b border-white/10 px-4 py-2">
        <span className="text-xs font-medium text-slate-300">{title}</span>
        <Button size="sm" variant="ghost" onClick={() => navigator.clipboard?.writeText(text)}>
          Copy
        </Button>
      </div>
      <pre className="scrollbar-thin max-h-96 overflow-auto p-4 text-xs leading-5 text-slate-300">{text}</pre>
    </div>
  );
}
