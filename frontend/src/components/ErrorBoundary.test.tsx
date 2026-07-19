import { afterEach, describe, expect, it, vi } from 'vitest'
import { createRoot } from 'react-dom/client'
import { flushSync } from 'react-dom'
import { ErrorBoundary } from './ErrorBoundary'

// No testing-library in this repo — render with the real react-dom client into
// jsdom and read the container. flushSync makes the render (and the boundary's
// catch) synchronous.
function render(ui: React.ReactNode) {
  const container = document.createElement('div')
  const root = createRoot(container)
  flushSync(() => root.render(ui))
  return container
}

afterEach(() => vi.restoreAllMocks())

describe('ErrorBoundary', () => {
  it('renders its children when nothing throws', () => {
    const c = render(
      <ErrorBoundary>
        <span>all fine</span>
      </ErrorBoundary>
    )
    expect(c.textContent).toContain('all fine')
    expect(c.textContent).not.toContain('Reload')
  })

  it('catches a render crash and offers a lossless reload instead of a blank page', () => {
    vi.spyOn(console, 'error').mockImplementation(() => {}) // React logs the throw
    const Boom = () => {
      throw new Error('boom at render')
    }
    const c = render(
      <ErrorBoundary>
        <Boom />
      </ErrorBoundary>
    )
    expect(c.textContent).toContain('Something broke in the editor')
    expect(c.textContent).toContain('nothing is lost')
    expect(c.textContent).toContain('boom at render') // the error is shown, not hidden
    expect(c.querySelector('button')?.textContent).toBe('Reload')
  })
})
