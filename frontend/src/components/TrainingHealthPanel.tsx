import { useState } from 'react'
import { useTrainingHealth, type Verdict } from '../hooks/useTrainingHealth'

const CHIP: Record<Verdict['level'], { glyph: string; color: string }> = {
  ok: { glyph: '✓', color: 'var(--accent)' },
  warn: { glyph: '⚠', color: 'var(--warn)' },
  error: { glyph: '✗', color: 'var(--error)' },
}

const BARS = '▁▂▃▄▅▆▇█'
// A unicode-block sparkline over a numeric series (non-finite → a blank slot).
function sparkline(nums: number[]): string {
  const finite = nums.filter(Number.isFinite)
  if (finite.length === 0) return ''
  const min = Math.min(...finite)
  const span = Math.max(...finite) - min || 1
  return nums
    .map((n) =>
      Number.isFinite(n) ? BARS[Math.min(BARS.length - 1, Math.floor(((n - min) / span) * (BARS.length - 1)))] : ' '
    )
    .join('')
}

const fmt = (n?: number) => (n === undefined ? '—' : n === 0 ? '0' : n.toExponential(1))

// Per-layer training-health readout for the current run: each layer's update
// ratio over epochs (sparkline), its latest value, and a verdict — keyed by the
// canvas node the layer maps to. Renders nothing until a run streams health.
export function TrainingHealthPanel() {
  const roles = useTrainingHealth()
  const [open, setOpen] = useState(true)
  if (roles.length === 0) return null

  const counts = { ok: 0, warn: 0, error: 0 }
  roles.forEach((r) => r.layers.forEach((l) => (counts[l.verdict.level] += 1)))
  const summary = [
    `${CHIP.ok.glyph} ${counts.ok}`,
    counts.warn ? `${CHIP.warn.glyph} ${counts.warn}` : '',
    counts.error ? `${CHIP.error.glyph} ${counts.error}` : '',
  ].filter(Boolean)

  return (
    <div style={{ borderTop: '1px solid var(--border)', background: 'var(--panel)', flexShrink: 0, fontFamily: 'monospace' }}>
      <button
        onClick={() => setOpen((o) => !o)}
        style={{
          width: '100%', display: 'flex', alignItems: 'center', gap: 10, background: 'none', border: 'none',
          cursor: 'pointer', padding: '8px 16px', fontFamily: 'monospace', fontSize: 11, color: 'var(--text-4)',
          textTransform: 'uppercase', letterSpacing: 1,
        }}
      >
        <span style={{ color: 'var(--text-6)' }}>{open ? '▾' : '▸'}</span>
        Training health
        <span style={{ display: 'flex', gap: 10, marginLeft: 'auto', textTransform: 'none', letterSpacing: 0 }}>
          {summary.map((s, i) => (
            <span
              key={i}
              style={{ color: i === 0 ? 'var(--accent)' : s.startsWith(CHIP.warn.glyph) ? 'var(--warn)' : 'var(--error)' }}
            >
              {s}
            </span>
          ))}
        </span>
      </button>

      {open && (
        <div style={{ maxHeight: 220, overflowY: 'auto', padding: '0 16px 12px' }}>
          {roles.map((r) => (
            <div key={r.role} style={{ marginBottom: 8 }}>
              {roles.length > 1 && (
                <div style={{ color: 'var(--text-6)', fontSize: 10, letterSpacing: 1, margin: '8px 0 6px' }}>
                  {r.role.toUpperCase()}
                </div>
              )}
              {r.layers.map((l) => {
                const chip = CHIP[l.verdict.level]
                return (
                  <div
                    key={l.layer}
                    title={l.verdict.note || undefined}
                    style={{ display: 'flex', alignItems: 'baseline', gap: 10, padding: '2px 0', fontSize: 12 }}
                  >
                    <span style={{ width: 110, flexShrink: 0, color: 'var(--text-3)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                      {l.node}
                    </span>
                    <span style={{ color: 'var(--accent-2)', letterSpacing: 1 }}>{sparkline(l.dw)}</span>
                    <span style={{ width: 62, textAlign: 'right', color: 'var(--text-5)' }}>{fmt(l.dw[l.dw.length - 1])}</span>
                    <span style={{ marginLeft: 'auto', display: 'flex', gap: 6, alignItems: 'baseline', minWidth: 0 }}>
                      <span style={{ color: chip.color, flexShrink: 0 }}>{chip.glyph}</span>
                      <span style={{ color: l.verdict.level === 'ok' ? 'var(--text-6)' : chip.color, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                        {l.verdict.label}
                      </span>
                    </span>
                  </div>
                )
              })}
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
