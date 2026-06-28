import { useEffect, useState } from 'react'
import { useRegistry } from './hooks/useRegistry'
import { useValidation } from './hooks/useValidation'
import { Canvas } from './components/Canvas'
import { Inspector } from './components/Inspector'
import { NodePalette } from './components/NodePalette'
import { useGraphStore } from './store/graphStore'

export default function App() {
  const { data: registry, isLoading, error } = useRegistry()
  const toDomainGraph = useGraphStore((s) => s.toDomainGraph)
  const loadGraph = useGraphStore((s) => s.loadGraph)
  const seedDefault = useGraphStore((s) => s.seedDefault)
  const graphIssues = useGraphStore((s) => s.graphIssues)
  const [hydrated, setHydrated] = useState(false)

  // Restore the cached graph from the backend before opening the WebSocket, so
  // reopening a closed tab brings back the design instead of clobbering it.
  useEffect(() => {
    if (!registry) return
    let cancelled = false
    ;(async () => {
      try {
        const res = await fetch('/api/graph')
        if (res.ok && !cancelled) {
          loadGraph(await res.json(), registry)
        } else if (res.status === 404 && !cancelled) {
          // Fresh session — no cached graph. Seed an Input → Output scaffold so
          // the canvas opens with the happy-path skeleton instead of blank.
          seedDefault(registry)
        }
      } catch {
        // backend hiccup — start with an empty canvas
      }
      if (!cancelled) setHydrated(true)
    })()
    return () => {
      cancelled = true
    }
  }, [registry, loadGraph, seedDefault])

  const { sendMove, sessionStopped, reconnecting, reconnect } = useValidation(hydrated, registry)

  const handleExport = async () => {
    const graph = toDomainGraph()
    const res = await fetch('/api/codegen', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(graph),
    })
    const body = await res.json()
    if (!res.ok) {
      alert(`Export failed: ${body.detail ?? 'unknown error'}`)
      return
    }
    const blob = new Blob([body.code as string], { type: 'text/plain' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = 'model.py'
    a.click()
    URL.revokeObjectURL(url)
  }

  if (isLoading) {
    return (
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100vh', background: '#0d0d1a', color: '#4a9eff', fontFamily: 'monospace' }}>
        Connecting to backend…
      </div>
    )
  }

  if (error || !registry) {
    return (
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100vh', background: '#0d0d1a', color: '#ff6b6b', fontFamily: 'monospace' }}>
        Backend unavailable — is <code style={{ margin: '0 4px' }}>python main.py</code> running?
      </div>
    )
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100vh', background: '#0d0d1a' }}>
      {/* Titlebar */}
      <div
        style={{
          height: 44,
          background: '#12121f',
          borderBottom: '1px solid #2a2a4a',
          display: 'flex',
          alignItems: 'center',
          padding: '0 16px',
          gap: 12,
          flexShrink: 0,
        }}
      >
        <span style={{ fontFamily: 'monospace', fontWeight: 700, color: '#4a9eff', fontSize: 16, letterSpacing: 1 }}>
          Lamplighter
        </span>
        <span style={{ color: '#2a2a4a', fontSize: 18 }}>|</span>
        <span style={{ fontFamily: 'monospace', color: '#444', fontSize: 12 }}>
          PyTorch Model Builder
        </span>
        {reconnecting && !sessionStopped && (
          <span
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: 6,
              fontFamily: 'monospace',
              color: '#e0a030',
              fontSize: 12,
            }}
          >
            <span
              style={{
                width: 7,
                height: 7,
                borderRadius: '50%',
                background: '#e0a030',
                animation: 'lamplighter-pulse 1s ease-in-out infinite',
              }}
            />
            Reconnecting...
          </span>
        )}
        <button
          onClick={handleExport}
          style={{
            marginLeft: 'auto',
            background: '#4a9eff',
            color: '#fff',
            border: 'none',
            borderRadius: 6,
            padding: '6px 16px',
            fontFamily: 'monospace',
            fontSize: 13,
            cursor: 'pointer',
            fontWeight: 600,
          }}
          onMouseEnter={(e) => (e.currentTarget.style.background = '#3a8eef')}
          onMouseLeave={(e) => (e.currentTarget.style.background = '#4a9eff')}
        >
          Export model.py
        </button>
      </div>

      {/* Graph-level validation banner */}
      {graphIssues.length > 0 && (
        <div
          style={{
            background: '#2a1a1a',
            borderBottom: '1px solid #ff6b6b44',
            color: '#ff9a9a',
            fontFamily: 'monospace',
            fontSize: 12,
            padding: '6px 16px',
            display: 'flex',
            flexDirection: 'column',
            gap: 2,
            flexShrink: 0,
          }}
        >
          {graphIssues.map((issue, i) => (
            <span key={i}>⚠ {issue}</span>
          ))}
        </div>
      )}

      {/* Main panels */}
      <div style={{ display: 'flex', flex: 1, overflow: 'hidden' }}>
        <NodePalette registry={registry} />
        <Canvas registry={registry} onNodeMove={sendMove} />
        <Inspector registry={registry} />
      </div>

      {sessionStopped && (
        <div
          style={{
            position: 'fixed',
            inset: 0,
            background: 'rgba(13, 13, 26, 0.85)',
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            justifyContent: 'center',
            gap: 12,
            zIndex: 1000,
            fontFamily: 'monospace',
            backdropFilter: 'blur(2px)',
          }}
        >
          <span style={{ color: '#ff6b6b', fontSize: 18, fontWeight: 700, letterSpacing: 1 }}>
            Session stopped
          </span>
          <span style={{ color: '#888', fontSize: 13 }}>
            The session was stopped from the notebook. Restart it with{' '}
            <code style={{ color: '#4a9eff' }}>lamplighter.start()</code>, then reconnect.
          </span>
          <button
            onClick={reconnect}
            style={{
              marginTop: 8,
              background: '#4a9eff',
              color: '#fff',
              border: 'none',
              borderRadius: 6,
              padding: '8px 20px',
              fontFamily: 'monospace',
              fontSize: 13,
              cursor: 'pointer',
              fontWeight: 600,
            }}
            onMouseEnter={(e) => (e.currentTarget.style.background = '#3a8eef')}
            onMouseLeave={(e) => (e.currentTarget.style.background = '#4a9eff')}
          >
            Reconnect
          </button>
        </div>
      )}
    </div>
  )
}
