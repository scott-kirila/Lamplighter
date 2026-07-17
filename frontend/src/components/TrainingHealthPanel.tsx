import { useState } from 'react'
import { concernColor, useTrainingHealth } from '../hooks/useTrainingHealth'
import { sparkBars } from '../lib/sparkline'
import { useRunStore } from '../store/runStore'

const SPARK_W = 360
const SPARK_H = 13

// The per-layer bar strip, one bar per epoch. Fixed width with every epoch
// drawn — the same no-truncation philosophy as the loss charts: the strip
// spans the run's planned epochs (bars grow rightward as they stream) and
// pixel density, not truncation, does the compression on long runs. Each bar
// wears the concern colour the score read AT that epoch (`colorAt`), so the
// strip is a timeline of the reading itself — history keeps its own colours
// instead of being repainted with the latest verdict.
function Sparkline({ series, span, colorAt }: { series: number[]; span: number; colorAt: (slot: number) => string }) {
  return (
    <svg width={SPARK_W} height={SPARK_H} style={{ alignSelf: 'center', flexShrink: 0 }}>
      {sparkBars(series, span, SPARK_W, SPARK_H).map((b) => (
        // A hairline gap between bars once they're wide enough to afford one.
        <rect key={b.i} x={b.x} y={b.y} width={Math.max(0.5, b.w - (b.w > 2.5 ? 1 : 0))} height={b.h} fill={colorAt(b.i)} />
      ))}
    </svg>
  )
}

const fmt = (n?: number) => (n === undefined ? '—' : n === 0 ? '0' : n.toExponential(1))

// The concern reading as a tiny meter: fill LENGTH carries it (empty = fine,
// full = problem), colour reinforces — so the verdict survives red-green
// colourblindness, which a hue-only dot didn't. Unscored layers: empty track.
function ConcernMeter({ concern, title }: { concern: number | null; title?: string }) {
  return (
    <span
      title={title}
      style={{
        marginLeft: 'auto', alignSelf: 'center', width: 36, height: 5, borderRadius: 3,
        background: 'var(--border)', overflow: 'hidden', flexShrink: 0,
      }}
    >
      {concern !== null && (
        <span
          style={{
            display: 'block', height: '100%', borderRadius: 3,
            width: `${Math.max(8, Math.round(concern * 100))}%`,
            background: concernColor(concern),
          }}
        />
      )}
    </span>
  )
}

// Per-layer training-health readout for the current run: each layer's update
// ratio over epochs (sparkline) and its latest value, colour-coded by concern —
// green→amber→red, deliberately unlabelled so the colour evokes the reading
// rather than the tool asserting a verdict. Renders nothing until a run streams
// health. Rows are keyed to the canvas node the layer maps to (hover for the
// factual context).
export function TrainingHealthPanel() {
  const roles = useTrainingHealth()
  // The run's planned epoch count (each epoch event carries it) — the bar
  // strips span it so bars grow rightward mid-run, like the loss charts.
  const planned = useRunStore((s) => s.runEpochs[s.runEpochs.length - 1]?.epochs ?? 0)
  const [open, setOpen] = useState(true)
  if (roles.length === 0) return null

  // Header meter = the worst concern anywhere, so a collapsed panel still signals.
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
        <ConcernMeter
          concern={worst}
          title="Worst layer — an empty green meter seems fine, a full red one likely a problem"
        />
      </button>

      {open && (
        <div style={{ maxHeight: 220, overflowY: 'auto', padding: '0 16px 12px' }}>
          {roles.map((r) => {
            // Pad every layer's sparkline to the run's full epoch span so columns
            // line up while KEEPING each layer's leading epochs. The update ratio
            // ‖Δw‖/‖w‖ is a between-epochs value (no epoch-1 point) while a
            // dead-unit fraction has one, so the shorter series get a leading
            // NaN pad (blank slots). Every strip spans the same slot count, so
            // the columns stay epoch-aligned.
            const epochs = Math.max(...r.layers.map((l) => (l.dead.length > 0 ? l.dead : l.dw).length))
            const span = Math.max(planned, epochs)
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
                <span title="One bar per epoch, spanning the run's planned epochs">
                  Δw/w · % dead · per epoch
                </span>
                <span style={{ marginLeft: 'auto' }}>Health</span>
              </div>
              {r.layers.map((l) => {
                // Parametric layers show the update ratio; activation layers (no
                // params) show their dead-unit fraction instead.
                const isDead = l.dead.length > 0
                const series = isDead ? l.dead : l.dw
                // Leading epochs this layer lacks (e.g. Δw/w has no epoch-1 reading)
                // — padded blank so bars start under epoch 2.
                const pad = epochs - series.length
                const display = [...Array<number>(pad).fill(NaN), ...series]
                // Display slot → health snapshot: concernSeries is indexed by
                // snapshot, the strip by the role's padded epoch span.
                const snapOffset = l.concernSeries.length - epochs
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
                    <Sparkline
                      series={display}
                      span={span}
                      colorAt={(slot) => {
                        const c = l.concernSeries[slot + snapOffset]
                        return c == null ? 'var(--text-7)' : concernColor(c)
                      }}
                    />
                    <span style={{ width: 72, textAlign: 'right', color: 'var(--text-5)' }}>{latest}</span>
                    <ConcernMeter concern={l.concern} />
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
