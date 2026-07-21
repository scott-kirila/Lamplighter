import { useEffect, useState } from 'react'
import { useGraphStore } from '../store/graphStore'
import { useTemplates } from '../hooks/useTemplates'
import { ConfirmModal } from './ConfirmModal'
import type { DomainProject, NodeDef } from '../types/graph'

// One row of the New-project menu: label + a small description line.
function MenuRow({
  label,
  description,
  onPick,
}: {
  label: string
  description: string
  onPick: () => void
}) {
  return (
    <button
      onClick={onPick}
      style={{
        display: 'block', width: '100%', textAlign: 'left', background: 'none',
        border: 'none', borderRadius: 6, padding: '7px 10px', cursor: 'pointer',
      }}
      onMouseEnter={(e) => (e.currentTarget.style.background = 'var(--field)')}
      onMouseLeave={(e) => (e.currentTarget.style.background = 'transparent')}
    >
      <div style={{ color: 'var(--text)', fontSize: 12.5, fontWeight: 600 }}>{label}</div>
      <div style={{ color: 'var(--text-6)', fontSize: 10.5, lineHeight: 1.35 }}>{description}</div>
    </button>
  )
}

interface ToolbarProps {
  registry: Record<string, NodeDef>
  // Live WS status, owned by App's useValidation instance.
  validationError: string | null
  dismissValidationError: () => void
  reconnecting: boolean
  sessionStopped: boolean
  // Theme, owned by App. (The code-panel toggle lives in App's tab row, next
  // to the attribution — it's tab-dependent, the titlebar isn't.)
  theme: string
  onToggleTheme: () => void
}

// The titlebar: identity, live-status banners, and the editor actions
// (undo/redo, New project, theme, export). Undo/redo and the
// New-project flow are self-contained here — they only touch the store,
// registry, and templates — so App just owns the WS/theme/export it already had.
export function Toolbar({
  registry,
  validationError,
  dismissValidationError,
  reconnecting,
  sessionStopped,
  theme,
  onToggleTheme,
}: ToolbarProps) {
  const undo = useGraphStore((s) => s.undo)
  const redo = useGraphStore((s) => s.redo)
  const canUndo = useGraphStore((s) => s.past.length > 0)
  const canRedo = useGraphStore((s) => s.future.length > 0)
  const resetProject = useGraphStore((s) => s.resetProject)
  const loadProject = useGraphStore((s) => s.loadProject)
  const freshStart = useGraphStore((s) => s.freshStart)
  const setActiveTab = useGraphStore((s) => s.setActiveTab)

  // ⌘Z / ⌃Z undo, ⌘⇧Z / ⌃Y redo — skipped while a text field has focus, so
  // the browser's native text-editing undo stays intact inside inputs.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (!(e.metaKey || e.ctrlKey)) return
      const t = e.target as HTMLElement | null
      if (t && (t.tagName === 'INPUT' || t.tagName === 'TEXTAREA' || t.isContentEditable)) return
      if (e.key.toLowerCase() === 'z') {
        e.preventDefault()
        if (e.shiftKey) redo()
        else undo()
      } else if (e.key.toLowerCase() === 'y') {
        e.preventDefault()
        redo()
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [undo, redo])

  // The New-project flow: blank, or one of the built-in templates (fetched
  // lazily — the list only loads once the menu first opens).
  const [newMenuOpen, setNewMenuOpen] = useState(false)
  const { data: templates } = useTemplates(newMenuOpen)
  // A new-project choice held for confirmation (`what` names it in the prompt,
  // `run` performs it) — the same styled modal the run flows use, not a native
  // window.confirm.
  const [pendingNew, setPendingNew] = useState<{ what: string; run: () => void } | null>(null)

  const loadTemplate = async (name: string) => {
    try {
      const res = await fetch(`/api/templates/${encodeURIComponent(name)}`)
      if (!res.ok) return
      const project = (await res.json()) as DomainProject
      loadProject(project, registry)
      freshStart() // a template load is a new project — fresh history + dashboard
      setActiveTab('overview') // land on the Models overview — see the whole project
    } catch {
      /* backend hiccup — keep the current project */
    }
  }
  const newBlank = () => {
    setNewMenuOpen(false)
    setPendingNew({ what: '', run: () => resetProject(registry) })
  }
  const newFromTemplate = (name: string, label: string) => {
    setNewMenuOpen(false)
    setPendingNew({ what: ` from the ${label} template`, run: () => loadTemplate(name) })
  }

  return (
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
      <span className="lamplighter-wordmark">Lamplighter</span>
      <span style={{ color: 'var(--border)', fontSize: 18 }}>|</span>
      <span style={{ color: 'var(--text-8)', fontSize: 12 }}>
        PyTorch Model Builder
      </span>
      {/* An unexpected backend exception while validating: the canvas is
          showing stale results until the next successful edit. */}
      {validationError && !sessionStopped && (
        <span
          data-tip={validationError}
          style={{
            display: 'flex', alignItems: 'center', gap: 6,
            color: 'var(--error)', fontSize: 12, maxWidth: 420,
          }}
        >
          <span style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
            ✗ backend error — shapes may be stale: {validationError}
          </span>
          <button
            onClick={dismissValidationError}
            data-tip="Dismiss"
            style={{
              background: 'none', border: 'none', color: 'var(--error)',
              cursor: 'pointer', fontSize: 12, padding: 0,
            }}
          >
            ✕
          </button>
        </span>
      )}
      {reconnecting && !sessionStopped && (
        <span
          style={{
            display: 'flex', alignItems: 'center', gap: 6,
            color: 'var(--warn)', fontSize: 12,
          }}
        >
          <span
            style={{
              width: 7, height: 7, borderRadius: '50%', background: 'var(--warn)',
              animation: 'lamplighter-pulse 1s ease-in-out infinite',
            }}
          />
          Reconnecting...
        </span>
      )}
      {/* Undo/redo over the project — structure and params, not layout drags. */}
      <button
        onClick={undo}
        disabled={!canUndo}
        aria-label="undo"
        data-tip="Undo (⌘Z)"
        style={{
          marginLeft: 'auto', background: 'none',
          color: canUndo ? 'var(--text-3)' : 'var(--text-7)',
          border: '1px solid var(--border)', borderRadius: 6, padding: '5px 11px',
          fontSize: 14, cursor: canUndo ? 'pointer' : 'default', lineHeight: 1,
        }}
      >
        ↩
      </button>
      <button
        onClick={redo}
        disabled={!canRedo}
        aria-label="redo"
        data-tip="Redo (⌘⇧Z)"
        style={{
          background: 'none', color: canRedo ? 'var(--text-3)' : 'var(--text-7)',
          border: '1px solid var(--border)', borderRadius: 6, padding: '5px 11px',
          fontSize: 14, cursor: canRedo ? 'pointer' : 'default', lineHeight: 1,
        }}
      >
        ↪
      </button>
      {/* A clean slate: the editor-standard "new document". Confirmed, since it
          replaces the current project (and its autosave); checkpoints and
          registered data are untouched. */}
      <span style={{ position: 'relative' }}>
        <button
          onClick={() => setNewMenuOpen((v) => !v)}
          data-tip="New project from a template — blank, or a built-in (checkpoints are kept)"
          style={{
            background: newMenuOpen ? 'var(--surface)' : 'none', color: 'var(--text-3)',
            border: '1px solid var(--border)', borderRadius: 6, padding: '5px 14px',
            fontSize: 13, cursor: 'pointer', fontWeight: 600,
          }}
        >
          Templates ▾
        </button>
        {newMenuOpen && (
          <>
            {/* click-away backdrop */}
            <div onClick={() => setNewMenuOpen(false)} style={{ position: 'fixed', inset: 0, zIndex: 99 }} />
            <div
              style={{
                position: 'absolute', top: '110%', right: 0, zIndex: 100, minWidth: 260,
                background: 'var(--surface)', border: '1px solid var(--border)',
                borderRadius: 8, padding: 6,
                boxShadow: 'var(--shadow-md)',
              }}
            >
              <MenuRow label="Blank" description="The empty Input → Output scaffold." onPick={newBlank} />
              {(templates ?? []).map((t) => (
                <MenuRow
                  key={t.name}
                  label={t.label}
                  description={t.description}
                  onPick={() => newFromTemplate(t.name, t.label)}
                />
              ))}
            </div>
          </>
        )}
      </span>
      <button
        onClick={onToggleTheme}
        aria-label="toggle theme"
        data-tip={`Switch to ${theme === 'dark' ? 'light' : 'dark'} mode`}
        style={{
          background: 'none', color: 'var(--text-3)', border: '1px solid var(--border)',
          borderRadius: 6, padding: '5px 11px', fontSize: 14,
          cursor: 'pointer', lineHeight: 1,
        }}
      >
        {theme === 'dark' ? '☀' : '☾'}
      </button>
      {pendingNew && (
        <ConfirmModal
          onCancel={() => setPendingNew(null)}
          actions={[
            {
              label: 'start new project',
              primary: true,
              onClick: () => {
                const run = pendingNew.run
                setPendingNew(null)
                run()
              },
            },
          ]}
        >
          Start a new project{pendingNew.what}? The current models, wiring, and training
          config are replaced (the autosave is overwritten). Saved checkpoints are kept.
        </ConfirmModal>
      )}
    </div>
  )
}
