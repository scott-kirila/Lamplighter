import { useState } from 'react'
import { concernColor, useTrainingHealth } from '../hooks/useTrainingHealth'

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
// The colour that carries the reading: green (fine) → amber (maybe) → red
// (problem). Not-yet-scored layers stay a neutral grey.
const dotColor = (concern: number | null) => (concern === null ? 'var(--text-7)' : concernColor(concern))

// Per-layer training-health readout for the current run: each layer's update
// ratio over epochs (sparkline) and its latest value, colour-coded by concern —
// green→amber→red, deliberately unlabelled so the colour evokes the reading
// rather than the tool asserting a verdict. Renders nothing until a run streams
// health. Rows are keyed to the canvas node the layer maps to (hover for the
// factual context).
export function TrainingHealthPanel() {
  const roles = useTrainingHealth()
  const [open, setOpen] = useState(true)
  if (roles.length === 0) return null

  // Header dot = the worst concern anywhere, so a collapsed panel still signals.
  const worst = Math.max(0, ...roles.flatMap((r) => r.layers.map((l) => l.concern ?? 0)))

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
        <span
          title="Worst layer — green seems fine, amber maybe, red likely a problem"
          style={{
            marginLeft: 'auto', width: 9, height: 9, borderRadius: '50%', background: dotColor(worst), flexShrink: 0,
          }}
        />
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
              {/* Column header — the number is the update ratio ‖Δw‖/‖w‖: the
                  sparkline is its history, the value the latest epoch's; the dot
                  is the colour-coded reading. */}
              <div
                style={{
                  display: 'flex', alignItems: 'baseline', gap: 10, padding: '2px 0 4px',
                  fontSize: 9.5, color: 'var(--text-7)', textTransform: 'uppercase', letterSpacing: 0.5,
                  borderBottom: '1px solid var(--border)', marginBottom: 4,
                }}
              >
                <span style={{ width: 110, flexShrink: 0 }}>Layer</span>
                <span>Update ratio (Δw/w) — spark · latest</span>
                <span style={{ marginLeft: 'auto' }}>Health</span>
              </div>
              {r.layers.map((l) => (
                <div
                  key={l.layer}
                  title={l.note}
                  style={{ display: 'flex', alignItems: 'baseline', gap: 10, padding: '2px 0', fontSize: 12 }}
                >
                  <span style={{ width: 110, flexShrink: 0, color: 'var(--text-3)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                    {l.node}
                  </span>
                  <span style={{ color: dotColor(l.concern), letterSpacing: 1 }}>{sparkline(l.dw)}</span>
                  <span style={{ width: 62, textAlign: 'right', color: 'var(--text-5)' }}>{fmt(l.dw[l.dw.length - 1])}</span>
                  <span
                    style={{
                      marginLeft: 'auto', alignSelf: 'center', width: 9, height: 9, borderRadius: '50%',
                      background: dotColor(l.concern), flexShrink: 0,
                    }}
                  />
                </div>
              ))}
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
