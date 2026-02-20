import { cn } from "@/lib/utils";
import type { LucideIcon } from "lucide-react";

interface MetricCardProps {
  label: string;
  value: number | string;
  unit?: string;
  icon?: LucideIcon;
  trend?: "up" | "down" | "neutral";
  fillPercent?: number;
  variant?: "default" | "compact";
  className?: string;
}

export function MetricCard({
  label,
  value,
  unit,
  icon: Icon,
  fillPercent,
  variant = "default",
  className,
}: MetricCardProps) {
  const isCompact = variant === "compact";

  return (
    <div
      className={cn(
        "relative overflow-hidden rounded-lg border border-slate-800 bg-[#0f1522] transition-all duration-200 hover:border-slate-700 hover:bg-[#121929]",
        isCompact ? "p-3" : "p-4",
        className
      )}
    >
      {/* Background fill indicator */}
      {fillPercent !== undefined && (
        <div
          className="absolute inset-0 bg-gradient-to-r from-primary/5 to-transparent transition-all duration-500"
          style={{ width: `${Math.min(fillPercent, 100)}%` }}
        />
      )}

      <div className="relative z-10">
        <div className="flex items-center gap-2">
          {Icon && (
            <Icon
              className={cn(
                "text-slate-400",
                isCompact ? "h-3.5 w-3.5" : "h-4 w-4"
              )}
            />
          )}
          <p
            className={cn(
              "uppercase tracking-wide text-slate-400",
              isCompact ? "text-[10px]" : "text-xs"
            )}
          >
            {label}
          </p>
        </div>
        <p
          className={cn(
            "font-semibold text-slate-100",
            isCompact ? "mt-1 text-lg" : "mt-2 text-xl"
          )}
        >
          {value}
          {unit && (
            <span
              className={cn(
                "ml-1 font-normal text-slate-400",
                isCompact ? "text-xs" : "text-sm"
              )}
            >
              {unit}
            </span>
          )}
        </p>
      </div>
    </div>
  );
}
