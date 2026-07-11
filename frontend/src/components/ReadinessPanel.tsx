import { useEffect, useState } from 'react'
import { useGraphStore } from '../store/graphStore'
import { useDataVariables } from '../hooks/useDataVariables'

interface DiagnosticCheck {
  level: 'ok' | 'warn' | 'error'
  title: string
  detail: string
}

const CHECK_ICON: Record<string, { glyph: string; color: string }> = {
  ok: { glyph: '✓', color: 'var(--accent)' },
  warn: { glyph: '⚠', color: 'var(--warn)' },
  error: { glyph: '✗', color: 'var(--error)' },
}

// Pre-flight data↔model checks that need the real registered tensors —
// sample-count alignment, class-range-vs-loss (the CUDA-assert catcher),
// batch-size × BatchNorm traps. They live next to ▶ Run rather than inline on
// the canvas (where shape/fit already show). Re-runs on any change to the loop,
// the data nodes, the models, or the registry.
export function ReadinessPanel() {
  const toProject = useGraphStore((s) => s.toProject)
  const training = useGraphStore((s) => s.training)
  const dataNodes = useGraphStore((s) => s.dataNodes)
  const models = useGraphStore((s) => s.models)
  const nodes = useGraphStore((s) => s.nodes)
  const activeModelId = useGraphStore((s) => s.activeModelId)
  const modelGraphs = useGraphStore((s) => s.modelGraphs)
  const { data: registered } = useDataVariables(true)

  const [checks, setChecks] = useState<DiagnosticCheck[]>([])
  const diagKey = JSON.stringify([
    training,
    dataNodes.map((d) => [d.kind, d.config]),
    models.map((m) => {
      const ns = m.id === activeModelId ? nodes : modelGraphs[m.id]?.nodes ?? []
      return ns.map((n) => [n.data.nodeType, n.data.params])
    }),
    registered,
  ])
  useEffect(() => {
    let cancelled = false
    const t = window.setTimeout(async () => {
      try {
        const res = await fetch('/api/data/diagnose', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(toProject()),
        })
        if (res.ok && !cancelled) setChecks((await res.json()).checks)
      } catch {
        /* backend hiccup — keep the last checklist */
      }
    }, 350)
    return () => {
      cancelled = true
      window.clearTimeout(t)
    }
  }, [diagKey, toProject])

  return (
    <div style={{ flex: 1, display: 'flex', flexDirection: 'column', minHeight: 0, padding: '16px 20px' }}>
      <div
        style={{
          fontFamily: 'monospace', fontSize: 11, color: 'var(--text-4)',
          textTransform: 'uppercase', letterSpacing: 1, marginBottom: 14,
        }}
      >
        Readiness
      </div>
      <div style={{ flex: 1, overflowY: 'auto', fontFamily: 'monospace', fontSize: 12 }}>
        {checks.length === 0 ? (
          <div style={{ color: 'var(--text-6)', lineHeight: 1.8 }}>
            Register data with <span style={{ color: 'var(--accent)' }}>sess.data(X=X, y=y)</span> and wire a
            data node into your model on the Models canvas — checks against the model appear here, then press ▶ Run.
          </div>
        ) : (
          checks.map((c, i) => {
            const icon = CHECK_ICON[c.level] ?? CHECK_ICON.warn
            return (
              <div key={i} style={{ display: 'flex', gap: 10, marginBottom: 10, lineHeight: 1.5 }}>
                <span style={{ color: icon.color, flexShrink: 0 }}>{icon.glyph}</span>
                <div style={{ minWidth: 0 }}>
                  <div style={{ color: c.level === 'error' ? 'var(--error)' : 'var(--text-2)' }}>{c.title}</div>
                  {c.detail && <div style={{ color: 'var(--text-5)', fontSize: 11 }}>{c.detail}</div>}
                </div>
              </div>
            )
          })
        )}
      </div>
    </div>
  )
}
