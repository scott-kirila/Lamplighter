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
import { OverviewView } from './components/OverviewView'
import { useGraphStore } from './store/graphStore'
import { Toolbar } from './components/Toolbar'

// A tab in the two-tier strip. `subtle` renders the smaller, subordinate style
// used by the per-model second row.
function TabButton({
  label,
  active,
  onClick,
  subtle = false,
}: {
  label: string
  active: boolean
  onClick: () => void
  subtle?: boolean
}) {
  return (
    <button
      onClick={onClick}
      style={{
        background: 'none',
        border: 'none',
        borderBottom: `2px solid ${active ? 'var(--accent)' : 'transparent'}`,
        color: active ? 'var(--text)' : 'var(--text-5)',
        cursor: 'pointer',
        fontSize: subtle ? 12 : 13,
        fontWeight: subtle ? 500 : 600,
        padding: subtle ? '6px 12px' : '8px 14px',
        whiteSpace: 'nowrap',
      }}
    >
      {label}
    </button>
  )
}

export default function App() {
  const { data: registry, isLoading, error } = useRegistry()
  const loadProject = useGraphStore((s) => s.loadProject)
  const seedDefault = useGraphStore((s) => s.seedDefault)
  const setActiveTab = useGraphStore((s) => s.setActiveTab)
  const graphIssues = useGraphStore((s) => s.graphIssues)
  const code = useGraphStore((s) => s.code)
  const activeTab = useGraphStore((s) => s.activeTab)
  const lastModelsTab = useGraphStore((s) => s.lastModelsTab)
  const models = useGraphStore((s) => s.models)
  const activeModelId = useGraphStore((s) => s.activeModelId)
  const trainingView = useGraphStore((s) => s.trainingView)
  const setTrainingView = useGraphStore((s) => s.setTrainingView)
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
    sendOverviewMove,
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

  if (isLoading) {
    return (
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100vh', background: 'var(--bg)', color: 'var(--accent)' }}>
        Connecting to backend…
      </div>
    )
  }

  if (error || !registry) {
    return (
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100vh', background: 'var(--bg)', color: 'var(--error)' }}>
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
      />

      {/* Two-tier tabs. Row 1 is the primary sections (Models | Training). Under
          Models, Row 2 is always shown — an Overview subtab (the OverviewView)
          plus one subtab per model — so the hierarchy reads the same whether a
          project has one model or several (no single-vs-multi special-casing).
          Startup lands on the Overview — see the whole project first. */}
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
        <TabButton
          label="Models"
          active={activeTab === 'overview' || activeTab === 'model'}
          // Return to the Models subtab you left (Overview or a model canvas),
          // mirroring how Training remembers its own sub-view.
          onClick={() => setActiveTab(lastModelsTab)}
        />
        <TabButton label="Training" active={activeTab === 'training'} onClick={() => setActiveTab('training')} />
        {/* Right cluster: the code-panel toggle beside the React Flow
            attribution (MIT). The toggle lives down here rather than in the
            titlebar because it's tab-dependent (the Models overview has no
            code panel) — hiding it here doesn't shift the titlebar buttons. */}
        <span style={{ marginLeft: 'auto', display: 'flex', alignItems: 'center', gap: 10 }}>
          {activeTab !== 'overview' && (
            <button
              onClick={() => setShowCode((v) => !v)}
              style={{
                background: showCode ? 'var(--surface)' : 'none',
                color: showCode ? 'var(--text)' : 'var(--text-3)',
                border: '1px solid var(--border)', borderRadius: 6, padding: '3px 12px',
                fontSize: 12, cursor: 'pointer', fontWeight: 600,
                margin: '4px 0',
              }}
            >
              {showCode ? 'Hide code' : 'Show code'}
            </button>
          )}
          <span style={{ color: 'var(--border)', fontSize: 16 }}>|</span>
          <a
            href="https://reactflow.dev"
            target="_blank"
            rel="noopener noreferrer"
            title="Node canvas powered by React Flow (xyflow)"
            style={{
              color: 'var(--text-8)',
              fontSize: 11,
              textDecoration: 'none',
            }}
          >
            Built with React Flow
          </a>
        </span>
      </div>

      {(activeTab === 'overview' || activeTab === 'model') && (
        <div
          style={{
            display: 'flex',
            background: 'var(--bg)',
            borderBottom: '1px solid var(--border)',
            padding: '0 20px',
            gap: 4,
            flexShrink: 0,
            overflowX: 'auto',
          }}
        >
          <TabButton subtle label="Overview" active={activeTab === 'overview'} onClick={() => setActiveTab('overview')} />
          {models.map((m) => (
            <TabButton
              key={m.id}
              subtle
              label={m.name}
              active={activeTab === 'model' && activeModelId === m.id}
              onClick={() => openModel(m.id)}
            />
          ))}
        </div>
      )}

      {/* Training's own second row — the run dashboard vs the model preview,
          same tab strip as Models' Overview/Model so the hierarchy reads alike. */}
      {activeTab === 'training' && (
        <div
          style={{
            display: 'flex',
            background: 'var(--bg)',
            borderBottom: '1px solid var(--border)',
            padding: '0 20px',
            gap: 4,
            flexShrink: 0,
            overflowX: 'auto',
          }}
        >
          <TabButton subtle label="Dashboard" active={trainingView === 'dashboard'} onClick={() => setTrainingView('dashboard')} />
          <TabButton subtle label="Preview" active={trainingView === 'preview'} onClick={() => setTrainingView('preview')} />
          <TabButton subtle label="Optimize" active={trainingView === 'optimize'} onClick={() => setTrainingView('optimize')} />
        </div>
      )}

      {activeTab === 'overview' ? (
        <OverviewView registry={registry} onModelMove={sendOverviewMove} />
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
