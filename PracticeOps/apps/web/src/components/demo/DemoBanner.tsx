/**
 * Demo Banner Component
 *
 * Displays a dismissible banner informing users they're in a demo workspace.
 * Shows quick links to key features.
 */

import { useState } from "react";
import { Link } from "react-router-dom";
import { X } from "lucide-react";

export function DemoBanner() {
  const [isDismissed, setIsDismissed] = useState(() => {
    return localStorage.getItem("demo_banner_dismissed") === "true";
  });

  const handleDismiss = () => {
    localStorage.setItem("demo_banner_dismissed", "true");
    setIsDismissed(true);
  };

  if (isDismissed) {
    return null;
  }

  return (
    <div className="bg-amber-50 border-b border-amber-200">
      <div className="container mx-auto px-4 md:px-6 py-3">
        <div className="flex items-center justify-between gap-4">
          <div className="flex-1 flex flex-col sm:flex-row sm:items-center gap-2 sm:gap-4">
            <p className="text-sm text-amber-900 font-medium">
              You're exploring a demo workspace
            </p>
            <div className="flex items-center gap-3 text-sm">
              <Link
                to="/assignments"
                className="text-amber-700 hover:text-amber-900 underline underline-offset-2"
              >
                View assignments
              </Link>
              <Link
                to="/tickets"
                className="text-amber-700 hover:text-amber-900 underline underline-offset-2"
              >
                View tickets
              </Link>
              <Link
                to="/practice"
                className="text-amber-700 hover:text-amber-900 underline underline-offset-2"
              >
                Practice logs
              </Link>
            </div>
          </div>
          <button
            onClick={handleDismiss}
            className="p-1 rounded hover:bg-amber-100 transition-colors"
            aria-label="Dismiss demo banner"
          >
            <X className="h-4 w-4 text-amber-700" />
          </button>
        </div>
      </div>
    </div>
  );
}
