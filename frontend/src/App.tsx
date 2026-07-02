import { useEffect, useState } from 'react'
import { useRegistry } from './hooks/useRegistry'
import { useValidation } from './hooks/useValidation'
import { useTheme } from './hooks/useTheme'
import { Canvas } from './components/Canvas'
import { CodePanel } from './components/CodePanel'
import { Inspector } from './components/Inspector'
import { NodePalette } from './components/NodePalette'
import { TrainingTab } from './components/TrainingTab'
import { DataTab } from './components/DataTab'
import { useGraphStore } from './store/graphStore'

export default function App() {
  const { data: registry, isLoading, error } = useRegistry()
  const toDomainGraph = useGraphStore((s) => s.toDomainGraph)
  const loadGraph = useGraphStore((s) => s.loadGraph)
  const seedDefault = useGraphStore((s) => s.seedDefault)
  const graphIssues = useGraphStore((s) => s.graphIssues)
  const code = useGraphStore((s) => s.code)
  const activeTab = useGraphStore((s) => s.activeTab)
  const setActiveTab = useGraphStore((s) => s.setActiveTab)
  const { theme, toggle: toggleTheme } = useTheme()
  const [hydrated, setHydrated] = useState(false)
  const [showCode, setShowCode] = useState(false)

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

  const { sendMove, sessionStopped, reconnecting, reconnect, setCodePreview } = useValidation(
    hydrated,
    registry
  )

  // Tell the backend to start/stop generating code as the panel opens/closes, so
  // codegen is skipped entirely while the panel is collapsed.
  useEffect(() => {
    setCodePreview(showCode)
  }, [showCode, setCodePreview])

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
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100vh', background: 'var(--bg)', color: 'var(--accent)', fontFamily: 'monospace' }}>
        Connecting to backend…
      </div>
    )
  }

  if (error || !registry) {
    return (
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100vh', background: 'var(--bg)', color: 'var(--error)', fontFamily: 'monospace' }}>
        Backend unavailable — is <code style={{ margin: '0 4px' }}>python main.py</code> running?
      </div>
    )
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100vh', background: 'var(--bg)' }}>
      {/* Titlebar */}
      <div
        style={{
          height: 44,
          background: 'var(--panel)',
          borderBottom: '1px solid var(--border)',
          display: 'flex',
          alignItems: 'center',
          padding: '0 16px',
          gap: 12,
          flexShrink: 0,
        }}
      >
        <span style={{ fontFamily: 'monospace', fontWeight: 700, color: 'var(--accent)', fontSize: 16, letterSpacing: 1 }}>
          Lamplighter
        </span>
        <span style={{ color: 'var(--border)', fontSize: 18 }}>|</span>
        <span style={{ fontFamily: 'monospace', color: 'var(--text-8)', fontSize: 12 }}>
          PyTorch Model Builder
        </span>
        {reconnecting && !sessionStopped && (
          <span
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: 6,
              fontFamily: 'monospace',
              color: 'var(--warn)',
              fontSize: 12,
            }}
          >
            <span
              style={{
                width: 7,
                height: 7,
                borderRadius: '50%',
                background: 'var(--warn)',
                animation: 'lamplighter-pulse 1s ease-in-out infinite',
              }}
            />
            Reconnecting...
          </span>
        )}
        <button
          onClick={toggleTheme}
          title={`Switch to ${theme === 'dark' ? 'light' : 'dark'} mode`}
          style={{
            marginLeft: 'auto',
            background: 'none',
            color: 'var(--text-3)',
            border: '1px solid var(--border)',
            borderRadius: 6,
            padding: '5px 11px',
            fontFamily: 'monospace',
            fontSize: 14,
            cursor: 'pointer',
            lineHeight: 1,
          }}
        >
          {theme === 'dark' ? '☀' : '☾'}
        </button>
        <button
          onClick={() => setShowCode((v) => !v)}
          style={{
            background: showCode ? 'var(--surface)' : 'none',
            color: showCode ? 'var(--text)' : 'var(--text-3)',
            border: '1px solid var(--border)',
            borderRadius: 6,
            padding: '5px 14px',
            fontFamily: 'monospace',
            fontSize: 13,
            cursor: 'pointer',
            fontWeight: 600,
          }}
        >
          {showCode ? 'Hide code' : 'Show code'}
        </button>
        <button
          onClick={handleExport}
          style={{
            background: 'var(--accent)',
            color: 'var(--text-on-accent)',
            border: 'none',
            borderRadius: 6,
            padding: '6px 16px',
            fontFamily: 'monospace',
            fontSize: 13,
            cursor: 'pointer',
            fontWeight: 600,
          }}
          onMouseEnter={(e) => (e.currentTarget.style.background = 'var(--accent-hover)')}
          onMouseLeave={(e) => (e.currentTarget.style.background = 'var(--accent)')}
        >
          Export model.py
        </button>
      </div>

      {/* Tabs */}
      <div
        style={{
          display: 'flex',
          background: 'var(--panel)',
          borderBottom: '1px solid var(--border)',
          padding: '0 12px',
          gap: 4,
          flexShrink: 0,
        }}
      >
        {(['model', 'data', 'training'] as const).map((tab) => (
          <button
            key={tab}
            onClick={() => setActiveTab(tab)}
            style={{
              background: 'none',
              border: 'none',
              borderBottom: `2px solid ${activeTab === tab ? 'var(--accent)' : 'transparent'}`,
              color: activeTab === tab ? 'var(--text)' : 'var(--text-5)',
              cursor: 'pointer',
              fontFamily: 'monospace',
              fontSize: 13,
              fontWeight: 600,
              padding: '8px 14px',
              textTransform: 'capitalize',
            }}
          >
            {tab}
          </button>
        ))}
      </div>

      {activeTab === 'training' ? (
        <TrainingTab />
      ) : activeTab === 'data' ? (
        <DataTab />
      ) : (
        <>
          {/* Graph-level validation banner */}
          {graphIssues.length > 0 && (
            <div
              style={{
                background: 'var(--error-bg)',
                borderBottom: '1px solid var(--error-border)',
                color: 'var(--error-text)',
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

          {showCode && <CodePanel code={code} onClose={() => setShowCode(false)} />}
        </>
      )}

      {sessionStopped && (
        <div
          style={{
            position: 'fixed',
            inset: 0,
            background: 'var(--overlay)',
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
          <span style={{ color: 'var(--error)', fontSize: 18, fontWeight: 700, letterSpacing: 1 }}>
            Session stopped
          </span>
          <span style={{ color: 'var(--text-4)', fontSize: 13 }}>
            The session was stopped from the notebook. Restart it with{' '}
            <code style={{ color: 'var(--accent)' }}>lamplighter.start()</code>, then reconnect.
          </span>
          <button
            onClick={reconnect}
            style={{
              marginTop: 8,
              background: 'var(--accent)',
              color: 'var(--text-on-accent)',
              border: 'none',
              borderRadius: 6,
              padding: '8px 20px',
              fontFamily: 'monospace',
              fontSize: 13,
              cursor: 'pointer',
              fontWeight: 600,
            }}
            onMouseEnter={(e) => (e.currentTarget.style.background = 'var(--accent-hover)')}
            onMouseLeave={(e) => (e.currentTarget.style.background = 'var(--accent)')}
          >
            Reconnect
          </button>
        </div>
      )}
    </div>
  )
}
