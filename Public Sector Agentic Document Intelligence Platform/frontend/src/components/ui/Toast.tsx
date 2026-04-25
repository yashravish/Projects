import {
  createContext,
  ReactNode,
  useCallback,
  useContext,
  useMemo,
  useReducer,
  useRef,
} from 'react';
import { cn } from '@/lib/cn';

type Tone = 'info' | 'success' | 'error';

interface Toast {
  id: number;
  tone: Tone;
  message: string;
}

interface ToastCtx {
  push: (message: string, tone?: Tone) => void;
}

const ToastContext = createContext<ToastCtx | null>(null);

interface State {
  items: Toast[];
}

type Action = { type: 'push'; toast: Toast } | { type: 'dismiss'; id: number };

function reducer(state: State, action: Action): State {
  switch (action.type) {
    case 'push':
      return { items: [...state.items, action.toast] };
    case 'dismiss':
      return { items: state.items.filter((t) => t.id !== action.id) };
  }
}

const toneRingClass: Record<Tone, string> = {
  info: 'border-rule-soft',
  success: 'border-forest/70 bg-forest/5',
  error: 'border-seal/70 bg-seal/5',
};

export function ToastProvider({ children }: { children: ReactNode }) {
  const [state, dispatch] = useReducer(reducer, { items: [] });
  const counter = useRef(0);

  const push = useCallback((message: string, tone: Tone = 'info') => {
    const id = ++counter.current;
    dispatch({ type: 'push', toast: { id, tone, message } });
    setTimeout(() => dispatch({ type: 'dismiss', id }), 5000);
  }, []);

  const value = useMemo(() => ({ push }), [push]);

  return (
    <ToastContext.Provider value={value}>
      {children}
      <div
        aria-live="polite"
        aria-atomic="true"
        className="fixed bottom-6 right-6 z-50 flex w-[min(28rem,90vw)] flex-col gap-2"
      >
        {state.items.map((t) => (
          <div
            key={t.id}
            role="status"
            className={cn(
              'animate-rise-in border-hair px-4 py-3 bg-paper',
              toneRingClass[t.tone],
            )}
          >
            <p className="rubric">{t.tone === 'error' ? 'incident' : t.tone}</p>
            <p className="text-sm mt-1 text-ink-80">{t.message}</p>
          </div>
        ))}
      </div>
    </ToastContext.Provider>
  );
}

export function useToast(): ToastCtx {
  const ctx = useContext(ToastContext);
  if (!ctx) throw new Error('useToast must be used within <ToastProvider>');
  return ctx;
}
