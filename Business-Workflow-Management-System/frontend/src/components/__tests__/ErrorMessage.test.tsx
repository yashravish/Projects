import { render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';
import ErrorMessage from '../ErrorMessage';

describe('ErrorMessage', () => {
  it('renders nothing when message is null', () => {
    const { container } = render(<ErrorMessage message={null} />);
    expect(container).toBeEmptyDOMElement();
  });

  it('renders an alert role for errors', () => {
    render(<ErrorMessage message="Something went wrong" />);
    const alert = screen.getByRole('alert');
    expect(alert).toHaveTextContent('Something went wrong');
  });

  it('renders a status role for non-error variants', () => {
    render(<ErrorMessage message="All good" variant="success" />);
    const status = screen.getByRole('status');
    expect(status).toHaveTextContent('All good');
  });
});
