import { updateDom } from './dom.js';
import { beginComponent, endComponent, cleanupHooks } from './hooks.js';
import { registerRenderer } from '../multiverse/universe.js';

const ROOTS = new Map(); // container -> { tree }
let _nextId = 0;

function ensureVnodeId(vnode) {
  if (vnode._id == null) vnode._id = `c${_nextId++}`;
  return vnode._id;
}

export function render(vnode, container, customSchedule) {
  const prev = ROOTS.get(container);
  const schedule = customSchedule || (prev ? prev.schedule : makeScheduler(container));
  const nextTree = prev
    ? reconcile(container, prev.tree, vnode, schedule)
    : mount(container, vnode, schedule);
  ROOTS.set(container, { tree: nextTree, schedule: customSchedule || nextTree._schedule });
  flushEffects(nextTree);
}

function makeScheduler(container) {
  const schedule = () => {
    queueMicrotask(() => {
      const root = ROOTS.get(container);
      if (!root) return;
      const newTree = reconcile(container, root.tree, root.tree._source, schedule);
      ROOTS.set(container, { tree: newTree, schedule });
      flushEffects(newTree);
    });
  };

  // Register once per container with multiverse
  if (!ROOTS.has(container)) {
    registerRenderer(schedule);
  }

  return schedule;
}

function mount(container, vnode, schedule) {
  const m = mountNode(vnode, schedule);
  if (!m) return null;
  m._schedule = schedule;
  container.innerHTML = '';
  if (m._dom) container.appendChild(m._dom);
  return m;
}

function mountNode(vnode, schedule) {
  if (!vnode) return null;

  vnode._source = vnode;

  if (typeof vnode.type === 'function') {
    const hooks = { 
      list: [], 
      i: 0, 
      effects: [], 
      __schedule: schedule,
      componentId: ensureVnodeId(vnode)
    };
    beginComponent(hooks);
    const child = vnode.type(vnode.props || {});
    endComponent();
    const mounted = mountNode(child, schedule);
    vnode._hooks = hooks;
    vnode._child = mounted;
    vnode._dom = mounted ? mounted._dom : null;
    return vnode;
  }
  
  if (vnode.type === 'TEXT') {
    vnode._dom = document.createTextNode(vnode.props.nodeValue);
    return vnode;
  }
  
  const dom = document.createElement(vnode.type);
  updateDom(dom, {}, vnode.props);
  (vnode.props.children || []).forEach(c => {
    const m = mountNode(c, schedule);
    if (m && m._dom) dom.appendChild(m._dom);
  });
  vnode._dom = dom;
  return vnode;
}

function unmount(vnode, container) {
  if (!vnode) return;
  
  if (typeof vnode.type === 'function') {
    // run effect cleanups
    cleanupHooks(vnode._hooks);
    // unmount its rendered child
    unmount(vnode._child, container);
    return;
  }
  
  // element or text
  const dom = vnode._dom;
  // unmount children first
  (vnode.props && vnode.props.children ? vnode.props.children : []).forEach(c => unmount(c, dom));
  if (dom && dom.parentNode) dom.parentNode.removeChild(dom);
}

function reconcile(container, oldV, newV, schedule) {
  if (!oldV) return mountNode(newV, schedule);
  
  // If newV is gone, unmount oldV properly (with cleanups)
  if (!newV) {
    unmount(oldV, container);
    return null;
  }
  
  // Function components (either side)
  if (typeof oldV.type === 'function' || typeof newV.type === 'function') {
    // Replacement: different component function → unmount old, mount new
    if (oldV.type !== newV.type) {
      unmount(oldV, container);
      const m = mountNode(newV, schedule);
      container.appendChild(m._dom);
      return m;
    }
    
    // Same function → reuse hooks instance
    // Preserve the component ID from oldV to newV
    newV._id = oldV._id;

    const hooks = oldV._hooks || {
      list: [],
      i: 0,
      effects: [],
      __schedule: schedule,
      componentId: ensureVnodeId(oldV)
    };
    hooks.__schedule = schedule;
    hooks.componentId = ensureVnodeId(oldV);

    beginComponent(hooks);
    const childOut = newV.type(newV.props || {});
    endComponent();

    const patchedChild = reconcile(container, oldV._child, childOut, schedule);
    newV._hooks = hooks;
    newV._child = patchedChild;
    newV._dom = patchedChild && patchedChild._dom ? patchedChild._dom : oldV._dom;
    newV._source = newV;
    return newV;
  }
  
  // Element/Text nodes
  if (oldV.type !== newV.type) {
    // Replace element/text → unmount old subtree first
    unmount(oldV, container);
    const m = mountNode(newV, schedule);
    container.appendChild(m._dom);
    return m;
  }
  
  const dom = (newV._dom = oldV._dom);
  updateDom(dom, oldV.props, newV.props);
  
  const oldCh = oldV.props.children || [];
  const newCh = (newV.props.children || []);
  const max = Math.max(oldCh.length, newCh.length);
  
  for (let i = 0; i < max; i++) {
    const patched = reconcile(dom, oldCh[i], newCh[i], schedule);
    if (patched) newCh[i] = patched;
  }
  
  newV._source = newV;
  return newV;
}

function flushEffects(vnode) {
  walk(vnode, v => v._hooks && v._hooks.effects ? v._hooks.effects.splice(0).forEach(fn => fn()) : null);
}

function walk(v, fn) {
  if (!v) return;
  fn(v);
  if (v._child) walk(v._child, fn);
  (v.props && v.props.children ? v.props.children : []).forEach(c => walk(c, fn));
}
