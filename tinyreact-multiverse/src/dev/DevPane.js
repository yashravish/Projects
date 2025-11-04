import { onUniverseChange, Universe, fork, checkout, commit, discard } from '../multiverse/universe.js';

export function mountDevPane() {
  // Avoid double-mount
  if (document.getElementById('__mv_pane__')) return;

  const pane = document.createElement('div');
  pane.id = '__mv_pane__';
  pane.setAttribute('role', 'dialog');
  pane.setAttribute('aria-label', 'Multiverse Dev Pane');
  pane.tabIndex = -1;
  pane.style.cssText = `
    position:fixed;right:12px;bottom:12px;z-index:2147483647;
    background:#111;color:#eee;font:12px/1.4 system-ui;
    border:1px solid #333;border-radius:10px;box-shadow:0 8px 24px rgba(0,0,0,.4);
    padding:10px;display:none;min-width:260px;max-width:320px;max-height:60vh;overflow:auto;
  `;
  document.body.appendChild(pane);

  let raf = 0;
  const schedule = () => {
    if (raf) return;
    raf = requestAnimationFrame(() => { raf = 0; renderPane(); });
  };

  function renderPane() {
    pane.innerHTML = `
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;gap:8px">
        <strong>Multiverse</strong>
        <span style="opacity:.7">branch: <code>${Universe.current}</code></span>
      </div>
    `;

    const list = document.createElement('div');

    Object.keys(Universe.branches).forEach(id => {
      const row = document.createElement('div');
      row.style.cssText = 'display:flex;gap:6px;align-items:center;margin:6px 0;flex-wrap:wrap;';
      const label = document.createElement('code'); 
      label.textContent = id;

      const ck = btn('Checkout', () => checkout(id));
      const cm = btn('Commit→main', () => commit(id));
      const rm = btn('Discard', () => discard(id));

      // Button states
      if (id === 'main') { 
        cm.disabled = true; 
        rm.disabled = true; 
        cm.style.opacity = rm.style.opacity = .4; 
      }
      if (id === Universe.current) { 
        ck.disabled = true; 
        ck.style.opacity = .5; 
      }

      // Optional: snapshot export/import
      const ex = btn('Export', () => {
        const payload = JSON.stringify([...Universe.branches[id].entries()]);
        if (navigator.clipboard) navigator.clipboard.writeText(payload).catch(()=>{});
      });
      
      const im = btn('Import', async () => {
        try {
          const txt = await navigator.clipboard.readText();
          const arr = JSON.parse(txt);
          Universe.branches[id] = new Map(arr);
          schedule(); // redraw pane
        } catch {}
      });

      row.append(label, ck, cm, rm, ex, im);
      list.appendChild(row);
    });

    const actions = document.createElement('div');
    actions.style.cssText = 'margin-top:8px;display:flex;gap:6px;flex-wrap:wrap;';
    actions.append(
      btn('Fork from current', () => fork()),
      btn('New named fork…', () => {
        const name = prompt('Branch name? (letters/numbers only)');
        if (!name) return;
        // naive name handling; overwrite if exists
        const base = Universe.branches[Universe.current] || new Map();
        Universe.branches[name] = new Map(base);
        schedule();
      })
    );

    pane.appendChild(list);
    pane.appendChild(actions);
  }

  const off = onUniverseChange(schedule);
  renderPane();

  function btn(txt, fn) {
    const b = document.createElement('button');
    b.textContent = txt;
    b.style.cssText = `
      background:#222;color:#ddd;border:1px solid #444;border-radius:6px;
      padding:4px 6px;cursor:pointer;
    `;
    b.onclick = fn;
    return b;
  }

  // Toggle with Alt+D; focus the pane when opening
  const keyHandler = (e) => {
    if (e.altKey && e.key.toLowerCase() === 'd') {
      const showing = pane.style.display !== 'none';
      pane.style.display = showing ? 'none' : 'block';
      if (!showing) {
        // ensure content current and move focus for accessibility
        schedule();
        setTimeout(() => pane.focus(), 0);
      }
    }
  };
  window.addEventListener('keydown', keyHandler);

  // Return cleanup so callers can unmount (optional)
  return () => {
    if (off) off();
    window.removeEventListener('keydown', keyHandler);
    cancelAnimationFrame(raf);
    pane.remove();
  };
}
