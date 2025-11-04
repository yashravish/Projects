// Hook runtime persisted per component instance via vnode._hooks
export let CURRENT_HOOKS = null;

export function beginComponent(hooks){
  CURRENT_HOOKS = hooks;
  CURRENT_HOOKS.i = 0;
  CURRENT_HOOKS.effects = []; // collect this render's effects
}

export function endComponent(){ /* noop */ }

// Optional: called by reconciler when a function component unmounts
export function cleanupHooks(hooks){
  if (!hooks || !hooks.list) return;
  for (let i = 0; i < hooks.list.length; i++){
    const slot = hooks.list[i];
    if (slot && typeof slot.cleanup === 'function') {
      try { slot.cleanup(); } catch {}
      slot.cleanup = null;
    }
  }
}

const is = Object.is || ((a,b) => (a===b) || (a!==a && b!==b)); // NaN-safe compare

export function useState(initial) {
  const hooks = CURRENT_HOOKS;         // capture the right instance
  const idx = hooks.i++;
  
  if (hooks.list[idx] === undefined) {
    hooks.list[idx] = typeof initial==='function' ? initial() : initial;
  }
  
  const setState = (next) => {
    const cur = hooks.list[idx];
    const val = (typeof next === 'function') ? next(cur) : next;
    if (!is(val, cur)) {
      hooks.list[idx] = val;
      hooks.__schedule();              // schedule via this component's root
    }
  };
  
  return [hooks.list[idx], setState];
}

export function useEffect(effect, deps) {
  const hooks = CURRENT_HOOKS;         // capture instance
  const idx = hooks.i++;
  const prev = hooks.list[idx];
  const changed = !prev || !deps || deps.some((d, i) => !is(d, prev.deps && prev.deps[i]));
  
  // Store meta in the slot now
  hooks.list[idx] = { deps, cleanup: prev && prev.cleanup ? prev.cleanup : null };
  
  if (changed) {
    // Queue a runner that closes over *this component's* hooks/slot
    hooks.effects.push(() => {
      try { if (hooks.list[idx].cleanup) hooks.list[idx].cleanup(); } catch {}
      hooks.list[idx].cleanup = effect() || null;
    });
  }
}
