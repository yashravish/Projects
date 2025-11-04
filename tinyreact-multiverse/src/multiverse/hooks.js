import { Universe } from './universe.js';
import { CURRENT_HOOKS } from '../core/hooks.js';

const is = Object.is || ((a, b) => (a === b) || (a !== a && b !== b)); // NaN-safe

// useBranchState(userKey, initial)
export function useBranchState(userKey, initial) {
  const hooks = CURRENT_HOOKS;                      // capture correct instance
  const idx = hooks.i++;                            // hook slot
  const componentId = hooks.componentId || 'Component';
  const key = `${componentId}#${idx}::${userKey ?? ''}`;

  const branchId = Universe.current;                // capture the branch at render
  const branch = Universe.branches[branchId] || (Universe.branches[branchId] = new Map());

  if (!branch.has(key)) {
    const init = (typeof initial === 'function') ? initial() : initial;
    branch.set(key, init);
  }

  const schedule = hooks.__schedule;                // capture scheduler

  const set = (next) => {
    // Always read from current branch state
    const currentBranch = Universe.branches[branchId] || (Universe.branches[branchId] = new Map());
    const cur = currentBranch.get(key);
    const val = (typeof next === 'function') ? next(cur) : next;
    if (!is(val, cur)) {
      currentBranch.set(key, val);
      schedule();
    }
  };

  return [branch.get(key), set];
}
