import { Component, type ReactNode } from 'react'

// The app-level error boundary: a render crash anywhere below would otherwise
// unmount the whole tree to a blank page. The kernel holds all real state (the
// project, runs, checkpoints), so a reload is lossless — offer it instead of a
// white screen. Class component because boundaries have no hook equivalent.
export class ErrorBoundary extends Component<
  { children: ReactNode },
  { error: Error | null }
> {
  state = { error: null as Error | null }

  static getDerivedStateFromError(error: Error) {
    return { error }
  }

  componentDidCatch(error: Error) {
    console.error('[lamplighter] render error:', error)
  }

  render() {
    if (!this.state.error) return this.props.children
    return (
      <div
        style={{
          display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center',
          height: '100vh', gap: 12, background: 'var(--bg)',
        }}
      >
        <span style={{ color: 'var(--error)', fontSize: 16, fontWeight: 700 }}>
          Something broke in the editor
        </span>
        <span style={{ color: 'var(--text-4)', fontSize: 12, maxWidth: 480, textAlign: 'center', lineHeight: 1.6 }}>
          Your project, runs, and checkpoints live in the kernel, so nothing is lost —
          reload to reconnect.
        </span>
        <code style={{ color: 'var(--text-6)', fontSize: 11 }}>{String(this.state.error)}</code>
        <button
          onClick={() => window.location.reload()}
          style={{
            marginTop: 8, background: 'var(--accent)', color: 'var(--text-on-accent)',
            border: 'none', borderRadius: 6, padding: '8px 20px',
            fontSize: 13, cursor: 'pointer', fontWeight: 600,
          }}
        >
          Reload
        </button>
      </div>
    )
  }
}
