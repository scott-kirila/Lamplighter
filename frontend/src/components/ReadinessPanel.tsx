import type { Readiness } from '../hooks/useReadiness'
import { eyebrow } from '../styles/ui'

const CHECK_ICON: Record<string, { glyph: string; color: string }> = {
  ok: { glyph: '✓', color: 'var(--accent)' },
  warn: { glyph: '⚠', color: 'var(--warn)' },
  error: { glyph: '✗', color: 'var(--error)' },
}

// The pre-flight readiness checklist (data↔model diagnostics from useReadiness),
// beside ▶ Run rather than inline on the canvas (where shape/fit already show).
export function ReadinessPanel({ readiness }: { readiness: Readiness }) {
  const { checks, status } = readiness
  return (
    <div style={{ flex: 1, display: 'flex', flexDirection: 'column', minHeight: 0, padding: '16px 20px' }}>
      <div
        style={{
          fontSize: 11, color: 'var(--text-4)',
          ...eyebrow, marginBottom: 14,
        }}
      >
        Readiness
      </div>
      <div style={{ flex: 1, overflowY: 'auto', fontSize: 12 }}>
        {status === 'unavailable' ? (
          // The diagnose call failed — admit uncertainty rather than showing a
          // stale checklist as if it were current.
          <div style={{ color: 'var(--text-6)', lineHeight: 1.8 }}>
            <span style={{ color: 'var(--warn)' }}>⚠ Readiness checks unavailable</span> — the backend
            didn't respond. Run still works; any blocker will surface as a run error.
          </div>
        ) : checks.length === 0 ? (
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
