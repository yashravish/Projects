TinyReact Multiverse

A minimal React-like framework featuring a groundbreaking Multiverse State system that enables branching application state for A/B testing, variant experimentation, and non-destructive state exploration.

What Makes This Special

Unlike traditional state management, which operates linearly, Multiverse State allows you to:

Fork state into parallel universes where changes don't affect the main timeline

Commit or discard branches to selectively adopt experiments

Switch between universes instantly without remounting components

Isolate experiments in separate branches for safe testing

Think Git branches—but for your application’s runtime state.

Quick Start
Prerequisites

Modern browser with ES module support

Local development server (e.g., VS Code Live Server or Python http.server)

Installation

No build step required—just clone and serve:

# Serve the directory
python3 -m http.server 8000
# or use VS Code Live Server extension


Then open http://localhost:8000 in your browser.

Basic Usage
import { createElement, render } from './src/core/index.js';
import { useBranchState } from './src/multiverse/hooks.js';
import { fork, checkout, commit } from './src/multiverse/universe.js';

function Counter() {
  const [count, setCount] = useBranchState('count', 0);

  const experimentWithVariant = () => {
    const branchId = fork();  
    checkout(branchId);
  };

  return (
    <div>
      <h1>Count: {count}</h1>
      <button onClick={() => setCount(count + 1)}>Increment</button>
      <button onClick={experimentWithVariant}>Try Variant</button>
    </div>
  );
}

render(<Counter />, document.getElementById('app'));

Core Concepts
1. Multiverse State

State is organized into branches (universes), each maintaining independent values:

Universe.branches = {
  main: Map { "Counter#0::count" => 5 },
  b1699123456: Map { "Counter#0::count" => 10 },
  compare: Map { "Counter#0::count" => 3 }
}


Each branch is a Map keyed by componentName#hookIndex::userKey.

2. Branch Operations

Fork – Create a new branch copying the current state:

const newBranch = fork('experiment-1');


Checkout – Switch active branch:

checkout('experiment-1');


Commit – Merge a branch into main:

commit('experiment-1');


Discard – Delete a branch:

discard('experiment-1');

3. Branch Portals

Render components in isolated branch contexts:

<BranchPortal branch="compare" into={previewContainer}>
  <Counter />
</BranchPortal>


This renders the component using the state from the compare branch, fully isolated from the active branch.

Architecture
Core React Implementation

TinyReact Multiverse includes a minimal but complete React-like system:

Virtual DOM – Lightweight vnode representation

Reconciliation – Efficient diffing and patching

Hooks – useState, useEffect with closure handling

Event System – Synthetic event layer

Lifecycle – Mount, update, unmount phases

File Structure
src/
├── core/
│   ├── createElement.js     # JSX transformation target
│   ├── dom.js               # DOM utilities
│   ├── hooks.js             # Hook runtime
│   ├── reconcile.js         # Virtual DOM diffing
│   └── index.js             # Public API exports
├── multiverse/
│   ├── universe.js          # Branch management core
│   ├── hooks.js             # useBranchState hook
│   └── BranchPortal.js      # Isolated rendering component
├── dev/
│   └── DevPane.js           # Developer tools
└── examples/
    └── counter.jsx          # Demo application

Developer Tools

Press Alt+D to toggle the Multiverse Dev Pane.

The Dev Pane includes:

Branch List – View all active universes

Checkout Controls – Switch between branches

Commit Actions – Merge branches into main

Discard Options – Remove experiments

All updates appear in real time as branches are created or modified.

API Reference
Core API
createElement(type, props, ...children)

Creates a virtual DOM element (used by JSX).

createElement('div', { className: 'box' }, 'Hello');

render(vnode, container)

Mounts a component tree.

render(<App />, document.getElementById('root'));

useState(initialValue)

Standard React-like state hook.

const [value, setValue] = useState(0);

useEffect(effect, deps)

Runs side effects after render.

useEffect(() => {
  document.title = `Count: ${count}`;
}, [count]);

Multiverse API
useBranchState(key, initialValue)

Branch-aware state hook.

const [count, setCount] = useBranchState('count', 0);

fork(id?)

Creates a new branch.

const branchId = fork();              
const named = fork('experiment-A');

checkout(branchId)

Switches to another branch.

checkout('main');

commit(branchId)

Merges a branch into main.

commit('experiment-A');

discard(branchId)

Deletes a branch.

discard('experiment-A');

onUniverseChange(callback)

Subscribes to branch change events.

const unsubscribe = onUniverseChange(() => {
  console.log('Universe changed:', Universe.current);
});
unsubscribe();

<BranchPortal>

Renders children within a specific branch.

<BranchPortal branch="preview" into={containerElement}>
  <MyComponent />
</BranchPortal>

Use Cases
A/B Testing
function ProductPage() {
  const [price] = useBranchState('price', 99);

  useEffect(() => {
    analytics.track('page_view', { price, branch: Universe.current });
  }, [price]);

  return <div>Price: ${price}</div>;
}

Feature Previews
function Editor() {
  const [content] = useBranchState('content', '');

  const previewExperiment = () => {
    fork('preview');
    checkout('preview');
  };

  return (
    <>
      <textarea value={content} onChange={...} />
      <button onClick={previewExperiment}>Preview Changes</button>
    </>
  );
}

Time Travel Debugging
const snapshot1 = fork('state-at-bug');
checkout('state-at-bug');

Side-by-Side Comparison
<div style={{ display: 'flex', gap: '20px' }}>
  <div>
    <h2>Current Pricing</h2>
    <PricingTable />
  </div>
  <div>
    <h2>Proposed Pricing</h2>
    <BranchPortal branch="proposed" into={proposedContainer}>
      <PricingTable />
    </BranchPortal>
  </div>
</div>

Example: Interactive Counter

The included counter.js demonstrates:

Branch-aware state with useBranchState

Forking and checkout for variant experimentation

Keyboard accessibility with arrow key controls

Visual feedback with success animations

Dev Pane integration (Alt+D)

Commit workflow to merge experiments back to main

Run by opening index.html in your browser.

Testing the Framework
Manual Acceptance Tests

Open http://localhost:8000 in your browser and verify the following:

1. **Counter Test**
   - Click the `+` or `−` buttons to change the count
   - Press `↑` or `↓` arrow keys to change the count
   - Verify the count updates correctly and displays with a subtle pulse animation
   - Verify the page title updates to `n={count}`

2. **Fork Test**
   - Click the "Try Variant" button
   - Verify the button shows a green success animation
   - Verify feedback message "Variant created ✓" appears and fades after 2 seconds
   - Note that a new branch has been created and is now active (shown in footer metadata)

3. **Branch Isolation Test**
   - Create a variant by clicking "Try Variant"
   - Change the count in the variant branch (click + or − buttons)
   - Press `Alt+D` to open the Dev Pane
   - Click "Checkout" on the `main` branch to switch back
   - Verify the main branch count is unchanged (state is isolated per branch)
   - Switch back to your variant branch and verify its count is preserved

4. **Commit Test**
   - Create a variant and modify its count
   - Click "Commit to Main" button
   - Verify the button shows a green success animation
   - Verify feedback message "Committed ✓" appears
   - Open Dev Pane (`Alt+D`) and verify the variant branch state merged to main

5. **Dev Pane Test**
   - Press `Alt+D` to toggle the Dev Pane
   - Verify all active branches are listed
   - Verify the current branch is highlighted
   - Create new variants and verify they appear in real-time
   - Use "Checkout" buttons to switch between branches
   - Use "Discard" to remove experimental branches

6. **Keyboard Accessibility Test**
   - Press `Tab` to navigate between interactive elements
   - Verify focus rings are visible on all buttons
   - Use `↑` and `↓` arrow keys to increment/decrement
   - Verify all actions can be performed without a mouse
   - Test with screen reader: count updates announce via `aria-live="polite"`

Future Enhancements
Planned Features

Conflict resolution UI

Snapshot import/export

Branch history and time-travel per branch

Automatic branch garbage collection

React DevTools integration

Middleware system

Persistent universe storage

Performance Optimizations

Structural sharing between branches

Lazy branch initialization

Portal render memoization

Virtual scrolling for branch lists

Technical Details
Hook Key Generation

State keys follow the format:

componentName#hookIndex::userKey


Example:

"Counter#0::count"


This ensures stable, unique, and user-controllable state identity.

Branch Isolation

Each BranchPortal:

Saves current branch

Switches to target branch

Renders children in isolation

Restores previous branch

This guarantees true state isolation between universes.

Re-render Strategy

When useBranchState updates:

Value is written to the active branch’s Map

A microtask is queued for reconciliation

The component re-renders using the updated branch data

Contributing

This framework is a proof-of-concept. Contributions are welcome for:

TypeScript definitions

Additional hooks (useReducer, useContext)

Testing utilities

Example apps

Performance improvements

Error boundaries

License

MIT License – free to use, modify, and distribute.

Acknowledgments

Inspired by:

React – Component model and hooks API

Git – Branching and merging concepts

Immer – Structural sharing

Redux DevTools – Time-travel debugging

Support

For questions or issues:

Open an issue on the repository

Review inline code comments

Experiment with the counter example

Built with curiosity and a desire to explore new paradigms in state management.
“What if your app’s state was a multiverse?”