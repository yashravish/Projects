interface ErrorMessageProps {
  message: string | null;
  variant?: 'error' | 'info' | 'success';
}

export default function ErrorMessage({ message, variant = 'error' }: ErrorMessageProps) {
  if (!message) return null;
  return (
    <div className={`alert ${variant}`} role={variant === 'error' ? 'alert' : 'status'}>
      {message}
    </div>
  );
}
