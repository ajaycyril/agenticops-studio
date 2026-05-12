import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

export function EdgeNodeCard({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <Card className="shadow-none">
      <CardHeader className="p-4 pb-2">
        <CardTitle className="text-sm">{title}</CardTitle>
      </CardHeader>
      <CardContent className="p-4 pt-0 text-xs leading-5 text-slate-400">{children}</CardContent>
    </Card>
  );
}
