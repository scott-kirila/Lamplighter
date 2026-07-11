import { useEffect, useState } from 'react'
import { useRegistry } from './hooks/useRegistry'
import { useValidation } from './hooks/useValidation'
import { useTrainingCode } from './hooks/useTrainingCode'
import { useTheme } from './hooks/useTheme'
import { Canvas } from './components/Canvas'
import { CodePanel } from './components/CodePanel'
import { Inspector } from './components/Inspector'
import { NodePalette } from './components/NodePalette'
import { TrainingTab } from './components/TrainingTab'
import { SystemView } from './components/SystemView'
import { useGraphStore } from './store/graphStore'
import { Toolbar } from './components/Toolbar'

export default function App() {
  const { data: registry, isLoading, error } = useRegistry()
  const toDomainGraph = useGraphStore((s) => s.toDomainGraph)
  const loadProject = useGraphStore((s) => s.loadProject)
  const seedDefault = useGraphStore((s) => s.seedDefault)
  const setActiveTab = useGraphStore((s) => s.setActiveTab)
  const graphIssues = useGraphStore((s) => s.graphIssues)
  const code = useGraphStore((s) => s.code)
  const activeTab = useGraphStore((s) => s.activeTab)
  const models = useGraphStore((s) => s.models)
  const activeModelId = useGraphStore((s) => s.activeModelId)
  const activeModelName = models.find((m) => m.id === activeModelId)?.name ?? 'Model'
  const openModel = useGraphStore((s) => s.openModel)
  const { theme, toggle: toggleTheme } = useTheme()
  const [hydrated, setHydrated] = useState(false)
  const [showCode, setShowCode] = useState(false)
  // Per-tab code panel content (each fetched only while visible on its tab).
  const trainingCode = useTrainingCode(showCode && activeTab === 'training')

  // Restore the cached graph from the backend before opening the WebSocket, so
  // reopening a closed tab brings back the project instead of clobbering it.
  useEffect(() => {
    if (!registry) return
    let cancelled = false
    ;(async () => {
      try {
        const res = await fetch('/api/project')
        if (res.ok && !cancelled) {
          loadProject(await res.json(), registry)
        } else if (res.status === 404 && !cancelled) {
          // Fresh session — no cached project. Seed an Input → Output scaffold so
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
  }, [registry, loadProject, seedDefault])

  const {
    sendMove,
    sendSystemMove,
    sessionStopped,
    reconnecting,
    reconnect,
    setCodePreview,
    validationError,
    dismissValidationError,
  } = useValidation(hydrated, registry)

  // Tell the backend to start/stop generating code as the panel opens/closes, so
  // codegen is skipped entirely while the panel is collapsed.
  useEffect(() => {
    setCodePreview(showCode)
  }, [showCode, setCodePreview])

  const handleExport = async () => {
    const graph = toDomainGraph()
    // With several models, name the exported class after the active model
    // (Generator/Discriminator); a lone model stays the classic GeneratedModel.
    const q = models.length > 1 ? `?name=${encodeURIComponent(activeModelName)}` : ''
    const res = await fetch(`/api/codegen${q}`, {
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
      <Toolbar
        registry={registry}
        validationError={validationError}
        dismissValidationError={dismissValidationError}
        reconnecting={reconnecting}
        sessionStopped={sessionStopped}
        theme={theme}
        onToggleTheme={toggleTheme}
        showCode={showCode}
        onToggleCode={() => setShowCode((v) => !v)}
        onExport={handleExport}
      />

      {/* Tabs. With several models, each opens in its own subtab beside the
          Models (system) overview, so switching between them is one click. A
          single-model project keeps just [ Models | Training ] — the Models tab
          then doubles as that model's canvas. */}
      <div
        style={{
          display: 'flex',
          background: 'var(--panel)',
          borderBottom: '1px solid var(--border)',
          padding: '0 12px',
          gap: 4,
          flexShrink: 0,
          overflowX: 'auto',
        }}
      >
        {(() => {
          const multiModel = models.length > 1
          const tabs: { label: string; active: boolean; onClick: () => void }[] = [
            {
              label: 'Models',
              // Multi-model: Models is the overview only (each model has its own
              // tab). Single-model: it also stands in for the lone canvas.
              active: multiModel ? activeTab === 'system' : activeTab === 'system' || activeTab === 'model',
              onClick: () => setActiveTab('system'),
            },
            ...(multiModel
              ? models.map((m) => ({
                  label: m.name,
                  active: activeTab === 'model' && activeModelId === m.id,
                  onClick: () => openModel(m.id),
                }))
              : []),
            { label: 'Training', active: activeTab === 'training', onClick: () => setActiveTab('training') },
          ]
          return tabs.map(({ label, active, onClick }, i) => (
            <button
              key={i}
              onClick={onClick}
              style={{
                background: 'none',
                border: 'none',
                borderBottom: `2px solid ${active ? 'var(--accent)' : 'transparent'}`,
                color: active ? 'var(--text)' : 'var(--text-5)',
                cursor: 'pointer',
                fontFamily: 'monospace',
                fontSize: 13,
                fontWeight: 600,
                padding: '8px 14px',
                whiteSpace: 'nowrap',
              }}
            >
              {label}
            </button>
          ))
        })()}
      </div>

      {activeTab === 'system' ? (
        <SystemView registry={registry} onModelMove={sendSystemMove} />
      ) : activeTab === 'training' ? (
        <>
          <TrainingTab />
          {showCode && (
            <CodePanel
              code={trainingCode}
              title="Generated train()"
              onClose={() => setShowCode(false)}
            />
          )}
        </>
      ) : (
        <>
          {/* No breadcrumb: the tab strip names the active model and clicking
              "Models" returns to the overview. */}

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
