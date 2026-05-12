import * as React from "react";
import { cn } from "@/lib/utils";

export function Alert({ className, ...props }: React.HTMLAttributes<HTMLDivElement>) {
  return <div className={cn("rounded-lg border border-amber-300/20 bg-amber-300/10 p-4 text-sm text-amber-100", className)} {...props} />;
}
