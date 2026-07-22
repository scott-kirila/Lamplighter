import { useState } from 'react'
import type { Readiness } from '../hooks/useReadiness'
import { border, eyebrow } from '../styles/ui'
import { readinessSummary } from '../lib/readinessSummary'

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


// The same checks, as a one-line strip that stays available once a run exists.
//
// The full panel above only renders before the first run — after that the epoch
// table owns the pane, and every warn-level finding went dark for the life of
// the tab. That is backwards: class imbalance, a BatchNorm meeting a ragged
// final batch, a backbone fed unnormalized data, causal-LM leakage — these
// matter most during the tweak-and-rerun loop, which is exactly when they
// disappeared. The data was being computed continuously the whole time; only
// the render site was gated.
//
// Collapsed by default so it costs one line, but the summary itself carries the
// verdict (colour, count, and the first problem's own words), so a glance is
// enough and expanding is optional.
export function ReadinessStrip({ readiness }: { readiness: Readiness }) {
  const { checks, status } = readiness
  const [open, setOpen] = useState(false)

  if (status !== 'ready' || checks.length === 0) return null

  const { level, text, more } = readinessSummary(checks)
  const icon = CHECK_ICON[level]

  return (
    <div style={{ borderBottom: border, background: 'var(--panel)', flexShrink: 0 }}>
      <button
        onClick={() => setOpen((v) => !v)}
        aria-expanded={open}
        style={{
          display: 'flex', alignItems: 'center', gap: 8, width: '100%',
          background: 'none', border: 'none', cursor: 'pointer',
          padding: '7px 20px', textAlign: 'left', fontSize: 11.5,
          color: 'var(--text-3)',
        }}
      >
        <span style={{ color: 'var(--text-6)', width: 8, flexShrink: 0 }}>{open ? '▾' : '▸'}</span>
        <span style={{ ...eyebrow, fontSize: 10, color: 'var(--text-4)', flexShrink: 0 }}>Readiness</span>
        <span style={{ color: icon.color, flexShrink: 0 }}>{icon.glyph}</span>
        <span
          style={{
            color: level === 'error' ? 'var(--error)' : 'var(--text-3)',
            overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', minWidth: 0,
          }}
        >
          {text}
        </span>
        {more > 0 && (
          <span style={{ color: 'var(--text-5)', flexShrink: 0 }}>+{more} more</span>
        )}
      </button>
      {open && (
        <div style={{ padding: '0 20px 12px 36px', fontSize: 12 }}>
          {checks.map((c, i) => {
            const ic = CHECK_ICON[c.level] ?? CHECK_ICON.warn
            return (
              <div key={i} style={{ display: 'flex', gap: 10, marginBottom: 8, lineHeight: 1.5 }}>
                <span style={{ color: ic.color, flexShrink: 0 }}>{ic.glyph}</span>
                <div style={{ minWidth: 0 }}>
                  <div style={{ color: c.level === 'error' ? 'var(--error)' : 'var(--text-2)' }}>{c.title}</div>
                  {c.detail && <div style={{ color: 'var(--text-5)', fontSize: 11 }}>{c.detail}</div>}
                </div>
              </div>
            )
          })}
        </div>
      )}
    </div>
  )
}
