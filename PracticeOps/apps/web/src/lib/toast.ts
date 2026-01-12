/**
 * Toast Notification Utilities
 *
 * Convenience wrappers for common toast patterns.
 * Uses the shadcn toast system configured in use-toast.ts.
 *
 * Behavior (per prompt requirements):
 * - Desktop: bottom-right
 * - Mobile: bottom-center (handled by ToastViewport)
 * - Auto-dismiss: 4 seconds
 * - Dismissible
 * - Stackable (up to 5)
 */

import { toast } from "@/hooks/use-toast";

interface ToastOptions {
  title?: string;
  description?: string;
}

/**
 * Show a success toast notification.
 * Use for successful operations: "Practice logged!", "Ticket created", etc.
 */
export function toastSuccess(message: string, options?: ToastOptions) {
  return toast({
    title: options?.title || "Success",
    description: message,
    className: "border-green-500/50 bg-green-50 dark:bg-green-950/20",
  });
}

/**
 * Show an error toast notification.
 * Use for failed operations: "Failed to save. Please try again."
 */
export function toastError(message: string, options?: ToastOptions) {
  return toast({
    variant: "destructive",
    title: options?.title || "Error",
    description: message,
  });
}

/**
 * Show a warning toast notification.
 * Use for alerts: "You have 2 blocking tickets due tomorrow"
 */
export function toastWarning(message: string, options?: ToastOptions) {
  return toast({
    title: options?.title || "Warning",
    description: message,
    className: "border-amber-500/50 bg-amber-50 dark:bg-amber-950/20",
  });
}

/**
 * Show an info toast notification.
 * Use for neutral information updates.
 */
export function toastInfo(message: string, options?: ToastOptions) {
  return toast({
    title: options?.title || "Info",
    description: message,
    className: "border-blue-500/50 bg-blue-50 dark:bg-blue-950/20",
  });
}

// Re-export base toast for custom usage
export { toast };

