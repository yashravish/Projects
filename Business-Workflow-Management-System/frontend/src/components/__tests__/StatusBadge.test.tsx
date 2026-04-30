import { render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';
import StatusBadge from '../StatusBadge';
import PriorityBadge from '../PriorityBadge';

describe('StatusBadge', () => {
  it('renders the status label', () => {
    render(<StatusBadge status="submitted" />);
    expect(screen.getByText('Submitted')).toBeInTheDocument();
  });

  it('renders priority labels', () => {
    render(<PriorityBadge priority="high" />);
    expect(screen.getByText('High')).toBeInTheDocument();
  });
});
