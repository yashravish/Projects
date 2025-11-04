// Multiverse state engine: branchable hook storage
export const Universe = {
  current: 'main',
  branches: { main: new Map() },          // branchId -> Map(hookKey -> state)
  listeners: new Set(),                   // external listeners (DevPane, etc.)
  renderers: new Set(),                   // schedulers from render roots
  _txDepth: 0, 
  _pending: false,
};

// --- subscriptions ---
export function onUniverseChange(fn){
  Universe.listeners.add(fn);
  return () => Universe.listeners.delete(fn);
}

// called by the renderer once per root to re-render on branch changes
export function registerRenderer(schedule){
  Universe.renderers.add(schedule);
  return () => Universe.renderers.delete(schedule);
}

function emit(){
  if (Universe._txDepth) {
    Universe._pending = true;
    return;
  }
  Universe.listeners.forEach(fn => fn());
  Universe.renderers.forEach(sched => sched());     // re-render all roots
}

function tx(fn){
  Universe._txDepth++;
  try { 
    fn(); 
  } finally {
    Universe._txDepth--;
    if (Universe._txDepth === 0 && Universe._pending) {
      Universe._pending = false; 
      emit();
    }
  }
}

// --- utils ---
const uid = () => (typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function'
  ? crypto.randomUUID()
  : 'b' + Math.random().toString(36).slice(2) + Date.now().toString(36));

function shallowCloneMap(m){ 
  return new Map(m || []); 
}

// Swap in a deep clone if you want isolation for object states:
// const deepCloneVal = v => (v && typeof v === 'object' ? JSON.parse(JSON.stringify(v)) : v);
// function deepCloneMap(m){ 
//   const out = new Map(); 
//   m?.forEach((v,k)=>out.set(k, deepCloneVal(v))); 
//   return out; 
// }

// --- API ---
export function fork(id = uid()) {
  const base = Universe.branches[Universe.current] || new Map();
  Universe.branches[id] = shallowCloneMap(base);
  emit();
  return id;
}

export function checkout(id) {
  if (!Universe.branches[id]) Universe.branches[id] = new Map();
  Universe.current = id;
  emit();
}

export function discard(id) {
  if (id === 'main' || !Universe.branches[id]) return;
  tx(() => {
    delete Universe.branches[id];
    if (Universe.current === id) Universe.current = 'main';
  });
  emit();
}

export function commit(id) {
  if (id === 'main') return;
  const src = Universe.branches[id];
  if (!src) return;
  tx(() => {
    Universe.branches.main = shallowCloneMap(src);
    Universe.current = 'main';
  });
  emit();
}

// Optional helpers (nice for DevPane/UX)
export function listBranches(){ 
  return Object.keys(Universe.branches); 
}

export function diff(a='main', b=Universe.current){
  const A = Universe.branches[a] || new Map();
  const B = Universe.branches[b] || new Map();
  const changed = [];
  const keys = new Set([...A.keys(), ...B.keys()]);
  keys.forEach(k => { 
    if (A.get(k) !== B.get(k)) changed.push(k); 
  });
  return changed;
}
