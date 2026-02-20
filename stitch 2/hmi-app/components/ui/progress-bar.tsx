import { cn } from "@/lib/utils";

interface ProgressBarProps {
  value: number;
  max?: number;
  variant?: "success" | "warning" | "critical";
  size?: "sm" | "md" | "lg";
  showGlow?: boolean;
  className?: string;
}

const variantStyles = {
  success: {
    gradient: "from-emerald-500 to-emerald-400",
    glow: "shadow-[0_0_10px_rgba(16,185,129,0.5)]",
  },
  warning: {
    gradient: "from-amber-500 to-amber-400",
    glow: "shadow-[0_0_10px_rgba(245,158,11,0.5)]",
  },
  critical: {
    gradient: "from-red-500 to-red-400",
    glow: "shadow-[0_0_10px_rgba(239,68,68,0.5)]",
  },
};

const sizeStyles = {
  sm: "h-1.5",
  md: "h-2",
  lg: "h-3",
};

export function ProgressBar({
  value,
  max = 100,
  variant = "success",
  size = "md",
  showGlow = true,
  className,
}: ProgressBarProps) {
  const percentage = Math.min(Math.max((value / max) * 100, 0), 100);
  const styles = variantStyles[variant];

  return (
    <div
      className={cn(
        "w-full bg-surface-highlight rounded-full overflow-hidden",
        sizeStyles[size],
        className
      )}
    >
      <div
        className={cn(
          "h-full bg-gradient-to-r rounded-full transition-all duration-500",
          styles.gradient,
          showGlow && styles.glow
        )}
        style={{ width: `${percentage}%` }}
      />
    </div>
  );
}
