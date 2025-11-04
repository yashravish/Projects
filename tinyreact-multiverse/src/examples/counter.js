import { createElement, render, useEffect } from '../core/index.js';
import { useBranchState } from '../multiverse/hooks.js';
import { Universe, fork, checkout, commit } from '../multiverse/universe.js';
import { mountDevPane } from '../dev/DevPane.js';

const h = (t, p, ...c) => createElement(t, p, ...c);

function Counter({ isPreview = false, cardTitle = null }) {
  const [n, setN] = useBranchState('n', 0);
  const [feedback, setFeedback] = useBranchState('feedback', '');

  // Avoid title thrash from the preview
  useEffect(() => {
    if (!isPreview) document.title = `n=${n}`;
  }, [n, isPreview]);

  // Add pulse animation when count changes
  useEffect(() => {
    if (isPreview) return;
    const countEl = document.querySelector('.count');
    if (countEl) {
      countEl.classList.add('count--pulse');
      setTimeout(() => countEl.classList.remove('count--pulse'), 200);
    }
  }, [n, isPreview]);

  // Keyboard accessibility
  useEffect(() => {
    if (isPreview) return;

    const handleKeyPress = (e) => {
      if (e.key === 'ArrowUp' || e.key === '+') {
        e.preventDefault();
        setN(n + 1);
      } else if (e.key === 'ArrowDown' || e.key === '-') {
        e.preventDefault();
        setN(n - 1);
      }
    };

    window.addEventListener('keydown', handleKeyPress);
    return () => window.removeEventListener('keydown', handleKeyPress);
  }, [n, isPreview]);

  const tryVariant = () => {
    const id = fork();
    checkout(id);

    // Visual feedback
    const btn = document.querySelector('.btn-secondary');
    if (btn) {
      btn.classList.add('btn--success');
      setTimeout(() => btn.classList.remove('btn--success'), 200);
    }

    setFeedback('Variant created ✓');
    setTimeout(() => setFeedback(''), 2000);
  };

  const commitMain = () => {
    commit(Universe.current);

    // Visual feedback
    const btn = document.querySelector('.btn-primary');
    if (btn) {
      btn.classList.add('btn--success');
      setTimeout(() => btn.classList.remove('btn--success'), 200);
    }

    setFeedback('Committed ✓');
    setTimeout(() => setFeedback(''), 2000);
  };

  if (cardTitle) {
    // Card variant for preview grid
    return h('div', { className: 'card-counter' },
      h('div', { style: 'font-size: 48px; font-weight: 300; margin-bottom: 20px; letter-spacing: -0.02em;' },
        'Count: ',
        h('span', { className: 'count', 'aria-live': 'polite' }, String(n))
      ),
      h('div', { className: 'segmented' },
        h('button', {
          className: 'segment-btn',
          onClick: () => setN(n + 1),
          'aria-label': 'Increment'
        }, '+'),
        h('div', { className: 'segment-divider', 'aria-hidden': 'true' }),
        h('button', {
          className: 'segment-btn',
          onClick: () => setN(n - 1),
          'aria-label': 'Decrement'
        }, '−')
      )
    );
  }

  return h('header', { className: 'header' },
    h('div', { className: 'state-section' },
      h('h1', { className: 'title' },
        'Count: ',
        h('span', { className: 'count', 'aria-live': 'polite' }, String(n))
      ),
      h('div', { className: 'segmented' },
        h('button', {
          className: 'segment-btn',
          onClick: () => setN(n + 1),
          'aria-label': 'Increment'
        }, '+'),
        h('div', { className: 'segment-divider', 'aria-hidden': 'true' }),
        h('button', {
          className: 'segment-btn',
          onClick: () => setN(n - 1),
          'aria-label': 'Decrement'
        }, '−')
      )
    ),
    !isPreview && h('div', { className: 'actions-section' },
      h('div', { className: 'toolbar' },
        h('button', { className: 'btn btn-secondary', onClick: tryVariant }, 'Try Variant'),
        h('button', { className: 'btn btn-primary', onClick: commitMain }, 'Commit to Main')
      ),
      feedback && h('p', { className: 'feedback' }, feedback)
    )
  );
}

function App() {
  return h('section', { className: 'container' },
    h(Counter, { isPreview: false }),
    h('footer', { className: 'app-footer' },
      h('div', { className: 'footer-hint' },
        'Press ',
        h('kbd', null, '↑'),
        ' / ',
        h('kbd', null, '↓'),
        ' or click buttons'
      ),
      h('div', { className: 'footer-meta' },
        'Current: ',
        h('strong', null, Universe.current),
        ' • Branch: ',
        h('strong', null, 'compare')
      )
    )
  );
}

export function start() {
  mountDevPane();
  
  // Ensure a compare branch exists mirroring main
  if (!Universe.branches.compare) {
    Universe.branches.compare = new Map(Universe.branches.main);
  }
  
  render(h(App), document.getElementById('app'));
}
