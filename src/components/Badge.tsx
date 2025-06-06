
import { cn } from "@/lib/utils";

interface BadgeProps extends React.HTMLAttributes<HTMLDivElement> {
  children: React.ReactNode;
  variant?: "default" | "secondary" | "outline" | "destructive";
}

export function Badge({
  className,
  variant = "default",
  ...props
}: BadgeProps) {
  return (
    <div
      className={cn(
        "inline-flex items-center rounded-full border px-2.5 py-0.5 text-xs font-semibold transition-colors focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2",
        variant === "default" &&
          "border-transparent bg-primary text-primary-foreground hover:bg-primary/80",
        variant === "secondary" &&
          "border-transparent bg-secondary text-secondary-foreground hover:bg-secondary/80",
        variant === "outline" &&
          "text-foreground",
        variant === "destructive" &&
          "border-transparent bg-destructive text-destructive-foreground hover:bg-destructive/80",
        className
      )}
      {...props}
    />
  );
}
