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
        }
      } catch {
        // no cached graph (404) or backend hiccup — start with an empty canvas
      }
      if (!cancelled) setHydrated(true)
    })()
    return () => {
      cancelled = true
    }
  }, [registry, loadGraph])

  const { sendMove } = useValidation(hydrated, registry)

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

      {/* Main panels */}
      <div style={{ display: 'flex', flex: 1, overflow: 'hidden' }}>
        <NodePalette registry={registry} />
        <Canvas registry={registry} onNodeMove={sendMove} />
        <Inspector registry={registry} />
      </div>
    </div>
  )
}
