interface LoadingStateProps {
  message?: string;
}

export default function LoadingState({ message = 'Loading...' }: LoadingStateProps) {
  return (
    <div className="loading-state" role="status" aria-live="polite">
      <div className="loading-state__track" aria-hidden>
        <span className="loading-state__pulse" />
      </div>
      <span className="loading-state__message">{message}</span>
    </div>
  );
}
