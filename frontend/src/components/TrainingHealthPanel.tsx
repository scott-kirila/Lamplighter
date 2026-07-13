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
          {roles.map((r) => {
            // Pad every layer's sparkline to the run's full epoch span so columns
            // line up while KEEPING each layer's leading epochs. The update ratio
            // ‖Δw‖/‖w‖ is a between-epochs value (no epoch-1 point) while a
            // dead-unit fraction has one, so the shorter series get a leading
            // spacer — rendered as transparent block glyphs (below), not spaces,
            // because a space is a different width and lands the bars half a cell off.
            const epochs = Math.max(...r.layers.map((l) => (l.dead.length > 0 ? l.dead : l.dw).length))
            return (
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
                <span style={{ width: 128, flexShrink: 0 }}>Layer</span>
                <span>Δw/w · % dead</span>
                <span style={{ marginLeft: 'auto' }}>Health</span>
              </div>
              {r.layers.map((l) => {
                // Parametric layers show the update ratio; activation layers (no
                // params) show their dead-unit fraction instead.
                const isDead = l.dead.length > 0
                const series = isDead ? l.dead : l.dw
                // Leading epochs this layer lacks (e.g. Δw/w has no epoch-1 reading)
                // — filled by a transparent block spacer so bars start under epoch 2.
                const pad = epochs - series.length
                const latest = isDead
                  ? `${Math.round((l.dead[l.dead.length - 1] ?? 0) * 100)}% dead`
                  : fmt(l.dw[l.dw.length - 1])
                return (
                  <div
                    key={l.layer}
                    title={`${l.layer} · ${l.node} — ${l.note}`}
                    style={{ display: 'flex', alignItems: 'baseline', gap: 10, padding: '2px 0', fontSize: 12 }}
                  >
                    {/* layer_N is the generated self.layer_N name — a unique,
                        code-cross-referenceable id, since the type alone repeats. */}
                    <span style={{ width: 128, flexShrink: 0, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                      <span style={{ color: 'var(--text-6)' }}>{l.layer}</span>{' '}
                      <span style={{ color: 'var(--text-3)' }}>{l.node}</span>
                    </span>
                    <span style={{ color: dotColor(l.concern), letterSpacing: 1 }}>
                      {pad > 0 && <span style={{ color: 'transparent' }}>{BARS[0].repeat(pad)}</span>}
                      {sparkline(series)}
                    </span>
                    <span style={{ width: 72, textAlign: 'right', color: 'var(--text-5)' }}>{latest}</span>
                    <span
                      style={{
                        marginLeft: 'auto', alignSelf: 'center', width: 9, height: 9, borderRadius: '50%',
                        background: dotColor(l.concern), flexShrink: 0,
                      }}
                    />
                  </div>
                )
              })}
            </div>
            )
          })}
        </div>
      )}
    </div>
  )
}
