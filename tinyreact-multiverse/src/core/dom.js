const isEvent = k => /^on[A-Z]/.test(k);
const toEventName = k => k.slice(2).toLowerCase();
const isAttr = k => k.startsWith('data-') || k.startsWith('aria-');
const isProp = k => k !== 'children' && !isEvent(k);
const isNil = v => v === null || v === undefined;

export function updateDom(dom, prev, next) {
  prev = prev || {};
  next = next || {};

  // 1) Remove old or changed event listeners
  for (const k in prev) {
    if (isEvent(k)) {
      const oldH = prev[k];
      const newH = next[k];
      if (!newH || oldH !== newH) {
        dom.removeEventListener(toEventName(k), oldH);
      }
    }
  }

  // 2) Remove old props/attrs that are gone now
  for (const k in prev) {
    if (!isProp(k)) continue;
    if (!(k in next)) {
      if (k === 'className' || k === 'class') dom.className = '';
      else if (k === 'style') dom.style.cssText = '';
      else if (k in dom && typeof dom[k] !== 'function') {
        // Known DOM property → clear
        try { dom[k] = typeof dom[k] === 'boolean' ? false : ''; } catch {}
      } else if (isAttr(k)) {
        dom.removeAttribute(k);
      } else {
        dom.removeAttribute(k);
      }
    }
  }

  // 3) Set new/changed props/attrs
  for (const k in next) {
    if (isEvent(k)) continue; // handled below
    if (!isProp(k)) continue;

    const oldV = prev[k];
    const newV = next[k];
    if (oldV === newV) continue;

    if (k === 'class' || k === 'className') {
      dom.className = newV || '';
      continue;
    }

    if (k === 'style') {
      if (isNil(newV)) {
        dom.style.cssText = '';
      } else if (typeof newV === 'string') {
        dom.style.cssText = newV;
      } else {
        // object style: set & cleanup removed keys
        const prevObj = (typeof oldV === 'object' && oldV) || {};
        const nextObj = (typeof newV === 'object' && newV) || {};
        for (const s in prevObj) {
          if (!(s in nextObj)) dom.style[s] = '';
        }
        for (const s in nextObj) {
          dom.style[s] = nextObj[s];
        }
      }
      continue;
    }

    // Input-like special cases: always set as property when available
    if (k === 'value' || k === 'checked' || k in dom) {
      try { dom[k] = newV; } catch { /* fallback to attribute */ }
      if (isNil(newV)) dom.removeAttribute(k);
      continue;
    }

    // data-/aria- and unknown → attributes
    if (isAttr(k) || typeof newV === 'string' || typeof newV === 'number' || typeof newV === 'boolean') {
      if (isNil(newV) || newV === false) dom.removeAttribute(k);
      else dom.setAttribute(k, String(newV));
      continue;
    }

    // Fallback: best effort property set
    try { dom[k] = newV; } catch { /* ignore */ }
  }

  // 4) Add new/changed event listeners
  for (const k in next) {
    if (!isEvent(k)) continue;
    const oldH = prev[k];
    const newH = next[k];
    if (oldH !== newH && typeof newH === 'function') {
      dom.addEventListener(toEventName(k), newH);
    }
  }
}
