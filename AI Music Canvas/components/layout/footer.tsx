'use client';

import Link from 'next/link';
import { ExternalLink } from 'lucide-react';

export function Footer() {
  return (
    <footer className="site-footer">
      <p className="attribution">
        AI Music Canvas — A portfolio project by <strong>Yash R.</strong>
      </p>

      {/* #9 — Separator dot and links */}
      <div className="flex items-center gap-6">
        <Link href="/architecture" className="footer-link">
          <ExternalLink size={12} strokeWidth={1.5} />
          How it&apos;s built
        </Link>
        <span style={{ color: 'rgba(255,255,255,0.15)' }} aria-hidden="true">·</span>
        <a
          href="https://github.com"
          target="_blank"
          rel="noopener noreferrer"
          className="footer-link"
        >
          <ExternalLink size={12} strokeWidth={1.5} />
          Source
        </a>
      </div>
    </footer>
  );
}
