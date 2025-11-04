import { render } from '../core/reconcile.js';
import { Universe, registerRenderer } from './universe.js';
import { useEffect } from '../core/hooks.js';

export function BranchPortal({ branch, into, intoId, children }) {
  const vnode = Array.isArray(children) ? children[0] : children;

  useEffect(() => {
    const target = into || (intoId ? document.getElementById(intoId) : null);
    if (!target) return;

    const schedule = () => {
      const prev = Universe.current;
      Universe.current = branch;
      render(vnode, target, schedule);  // Pass our wrapped scheduler to maintain branch context
      Universe.current = prev;
    };

    // Initial render
    schedule();

    // Re-render this portal on any multiverse op (fork/checkout/commit/discard)
    const off = registerRenderer(schedule);

    // Cleanup: stop listening, but DON'T clear DOM (let reconciliation handle it)
    return () => {
      if (off) off();
    };
  }, [branch, into, intoId]);

  return null;
}
