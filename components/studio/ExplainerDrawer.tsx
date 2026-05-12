import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

export function ExplainerDrawer({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <Card>
      <CardHeader><CardTitle>{title}</CardTitle></CardHeader>
      <CardContent className="text-sm leading-6 text-slate-300">{children}</CardContent>
    </Card>
  );
}
