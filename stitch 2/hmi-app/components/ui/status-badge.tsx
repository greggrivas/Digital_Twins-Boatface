import { cn } from "@/lib/utils";

type StatusVariant = "healthy" | "warning" | "critical" | "online" | "offline";

interface StatusBadgeProps {
  status: StatusVariant;
  label?: string;
  showPulse?: boolean;
  size?: "sm" | "md";
  className?: string;
}

const variantStyles = {
  healthy: {
    container: "bg-emerald-500/10 text-emerald-400 ring-1 ring-inset ring-emerald-500/20",
    dot: "bg-emerald-500",
  },
  warning: {
    container: "bg-amber-500/10 text-amber-400 ring-1 ring-inset ring-amber-500/20",
    dot: "bg-amber-500",
  },
  critical: {
    container: "bg-red-500/10 text-red-400 ring-1 ring-inset ring-red-500/20",
    dot: "bg-red-500",
  },
  online: {
    container: "bg-emerald-500/10 text-emerald-400 ring-1 ring-inset ring-emerald-500/20",
    dot: "bg-emerald-500",
  },
  offline: {
    container: "bg-slate-500/10 text-slate-400 ring-1 ring-inset ring-slate-500/20",
    dot: "bg-slate-500",
  },
};

const defaultLabels: Record<StatusVariant, string> = {
  healthy: "Healthy",
  warning: "Warning",
  critical: "Critical",
  online: "Online",
  offline: "Offline",
};

export function StatusBadge({
  status,
  label,
  showPulse = false,
  size = "md",
  className,
}: StatusBadgeProps) {
  const styles = variantStyles[status];
  const displayLabel = label ?? defaultLabels[status];

  return (
    <span
      className={cn(
        "inline-flex items-center gap-1.5 rounded-full font-semibold uppercase",
        styles.container,
        size === "sm" ? "px-2 py-0.5 text-[10px]" : "px-3 py-1 text-xs",
        className
      )}
    >
      <span
        className={cn(
          "rounded-full",
          styles.dot,
          size === "sm" ? "h-1.5 w-1.5" : "h-2 w-2",
          showPulse && "animate-pulse"
        )}
      />
      {displayLabel}
    </span>
  );
}
