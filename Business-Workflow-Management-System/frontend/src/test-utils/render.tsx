import type { ReactElement } from 'react';
import { render, type RenderOptions } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';

interface CustomRenderOptions extends Omit<RenderOptions, 'wrapper'> {
  initialPath?: string;
}

export function renderWithRouter(
  ui: ReactElement,
  { initialPath = '/', ...options }: CustomRenderOptions = {}
) {
  return render(ui, {
    wrapper: ({ children }) => (
      <MemoryRouter initialEntries={[initialPath]}>{children}</MemoryRouter>
    ),
    ...options,
  });
}
