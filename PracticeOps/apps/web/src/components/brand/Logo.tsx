import { cn } from "@/lib/utils";

interface LogoProps {
  className?: string;
}

export function Logo({ className }: LogoProps) {
  return (
    <div className={cn("inline-flex flex-col items-start text-neutral-900", className)}>
      <span className="text-xl font-semibold tracking-tight">PracticeOps</span>
      <span className="mt-1 h-px w-full bg-neutral-900" />
    </div>
  );
}
