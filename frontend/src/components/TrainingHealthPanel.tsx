import { useState } from 'react'
import { concernColor, type RoleHealth } from '../hooks/useTrainingHealth'
import { sparkBars } from '../lib/sparkline'

// The sparkline's coordinate system. The <svg> scales to fill its cell (the
// cell width now tracks the graphs column), so these are viewBox units, not px.
const SPARK_W = 360
const SPARK_H = 13

// The per-layer bar strip, one bar per epoch. Every epoch drawn — the same
// no-truncation philosophy as the loss charts: the strip spans the run's planned
// epochs (bars grow rightward as they stream) and pixel density, not truncation,
// does the compression on long runs. Each bar wears the concern colour the score
// read AT that epoch (`colorAt`), so the strip is a timeline of the reading
// itself — history keeps its own colours instead of being repainted with the
// latest verdict. It fills its container width (preserveAspectRatio="none"
// stretches the timeline horizontally, which is meaningless-free for bars).
function Sparkline({ series, span, colorAt }: { series: number[]; span: number; colorAt: (slot: number) => string }) {
  return (
    <svg
      viewBox={`0 0 ${SPARK_W} ${SPARK_H}`}
      width="100%"
      height={SPARK_H}
      preserveAspectRatio="none"
      style={{ display: 'block' }}
    >
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

// The vertical rule between the band's two columns — a 1px line centred in the
// same 7px the dashboard's Separator uses, so stacked rows read as one line
// aligned under the graphs↔results divider above.
function ColDivider() {
  return (
    <div style={{ width: 7, flexShrink: 0, alignSelf: 'stretch', display: 'flex', justifyContent: 'center' }}>
      <div style={{ width: 1, background: 'var(--border)' }} />
    </div>
  )
}

// Per-layer derived values, shared by both layouts (split columns / single row).
type Row = ReturnType<typeof layerRow>
function layerRow(l: RoleHealth['layers'][number], epochs: number, span: number) {
  // Parametric layers show the update ratio; activation layers (no params) show
  // their dead-unit fraction instead.
  const isDead = l.dead.length > 0
  const series = isDead ? l.dead : l.dw
  // Leading epochs this layer lacks (e.g. Δw/w has no epoch-1 reading) — padded
  // blank so bars start under epoch 2.
  const pad = epochs - series.length
  const display = [...Array<number>(pad).fill(NaN), ...series]
  // Display slot → health snapshot: concernSeries is indexed by snapshot, the
  // strip by the role's padded epoch span.
  const snapOffset = l.concernSeries.length - epochs
  const latest = isDead
    ? `${Math.round((l.dead[l.dead.length - 1] ?? 0) * 100)}% dead`
    : fmt(l.dw[l.dw.length - 1])
  const bars = (
    <Sparkline
      series={display}
      span={span}
      colorAt={(slot) => {
        const c = l.concernSeries[slot + snapOffset]
        return c == null ? 'var(--text-7)' : concernColor(c)
      }}
    />
  )
  // layer_N is the generated self.layer_N name — a unique, code-cross-
  // referenceable id, since the type alone repeats.
  const name = (
    <>
      <span style={{ color: 'var(--text-6)' }}>{l.layer}</span>{' '}
      <span style={{ color: 'var(--text-3)' }}>{l.node}</span>
    </>
  )
  return { key: l.layer, title: `${l.layer} · ${l.node} — ${l.note}`, concern: l.concern, bars, name, latest }
}

// Per-layer training-health readout for the current run: each layer's update
// ratio over epochs (sparkline) and its latest value, colour-coded by concern —
// green→amber→red, deliberately unlabelled so the colour evokes the reading
// rather than the tool asserting a verdict. Rows are keyed to the canvas node
// the layer maps to (hover for the factual context).
//
// It's the dashboard's full-width bottom strip, anchored below the
// graphs|results columns. When both columns are up, `split` carries their live
// ratio and the band mirrors it: bars on the LEFT (tracking the graphs column),
// layer name / value / status on the RIGHT (tracking the results column). With
// no split (results hidden) it falls back to one full-width row per layer. The
// header ▾ button collapses the strip to just itself, handing the vertical space
// back to the columns above — self-contained state, since the strip no longer
// lives in a resizable pane.
export function TrainingHealthPanel({
  roles,
  planned,
  split,
}: {
  roles: RoleHealth[]
  planned: number
  split?: { graphs: number; results: number } | null
}) {
  const [collapsed, setCollapsed] = useState(false)

  if (roles.length === 0) return null

  // Header meter = the worst concern anywhere, so a collapsed strip still signals.
  const worst = Math.max(0, ...roles.flatMap((r) => r.layers.map((l) => l.concern ?? 0)))

  // The right-column trailer: latest value + status meter. Shared by both layouts.
  const meta = (row: Row) => (
    <>
      <span style={{ flexShrink: 0, width: 72, textAlign: 'right', color: 'var(--text-5)' }}>{row.latest}</span>
      <ConcernMeter concern={row.concern} />
    </>
  )

  return (
    <div style={{ flexShrink: 0, display: 'flex', flexDirection: 'column', overflow: 'hidden', borderTop: '1px solid var(--border)', background: 'var(--panel)', fontFamily: 'monospace' }}>
      <button
        onClick={() => setCollapsed((c) => !c)}
        style={{
          flexShrink: 0, width: '100%', display: 'flex', alignItems: 'center', gap: 10, background: 'none', border: 'none',
          cursor: 'pointer', padding: '8px 16px', fontFamily: 'monospace', fontSize: 11, color: 'var(--text-4)',
          textTransform: 'uppercase', letterSpacing: 1,
        }}
      >
        <span style={{ color: 'var(--text-6)' }}>{collapsed ? '▸' : '▾'}</span>
        Training health
        <ConcernMeter
          concern={worst}
          title="Worst layer — an empty green meter seems fine, a full red one likely a problem"
        />
      </button>

      {!collapsed && (
        <div className="health-scroll" style={{ maxHeight: 260, overflowY: 'auto', overflowX: 'hidden', padding: '0 16px 12px' }}>
          {roles.map((r) => {
            // Pad every layer's sparkline to the run's full epoch span so columns
            // line up while KEEPING each layer's leading epochs. Every strip spans
            // the same slot count, so the columns stay epoch-aligned.
            const epochs = Math.max(...r.layers.map((l) => (l.dead.length > 0 ? l.dead : l.dw).length))
            const span = Math.max(planned, epochs)
            const rows = r.layers.map((l) => layerRow(l, epochs, span))
            return (
              <div key={r.role} style={{ marginBottom: 8 }}>
                {roles.length > 1 && (
                  <div style={{ color: 'var(--text-6)', fontSize: 10, letterSpacing: 1, margin: '8px 0 6px' }}>
                    {r.role.toUpperCase()}
                  </div>
                )}
                {/* Column header — the number is the update ratio ‖Δw‖/‖w‖: the
                    sparkline is its history, the value the latest epoch's; the
                    meter is the colour-coded reading. */}
                {split ? (
                  // Negative side margins cancel the body's 16px inset so this row
                  // spans the full group width — identical flex geometry to the
                  // graphs↔results split above (flexBasis:0 panels + a 7px rule),
                  // so the divider lands exactly under the top Separator. The 16px
                  // breathing room moves inside the cells, where it can't shift the
                  // divider.
                  <div style={{ display: 'flex', alignItems: 'stretch', margin: '0 -16px 4px', fontSize: 9.5, color: 'var(--text-7)', textTransform: 'uppercase', letterSpacing: 0.5 }}>
                    {/* Cells carry NO padding — flex-basis:0 can't shrink below an
                        item's padding, so padding on the cell would add to its
                        base size and (being asymmetric L vs R) shove the divider
                        off the top Separator. The insets live on inner wrappers,
                        which don't feed the cell's flex base. */}
                    <div style={{ flexGrow: split.graphs, flexBasis: 0, minWidth: 0, display: 'flex' }}>
                      <div style={{ flex: 1, minWidth: 0, borderBottom: '1px solid var(--border)', padding: '2px 0 4px 16px' }}>
                        <span title="One bar per epoch, spanning the run's planned epochs">Δw/w · % dead · per epoch</span>
                      </div>
                    </div>
                    <ColDivider />
                    <div style={{ flexGrow: split.results, flexBasis: 0, minWidth: 0, display: 'flex' }}>
                      <div style={{ flex: 1, minWidth: 0, display: 'flex', gap: 10, borderBottom: '1px solid var(--border)', padding: '2px 16px 4px 12px' }}>
                        <span style={{ flex: 1, minWidth: 0 }}>Layer</span>
                        <span>Health</span>
                      </div>
                    </div>
                  </div>
                ) : (
                  <div style={{ display: 'flex', alignItems: 'baseline', gap: 10, padding: '2px 0 4px', fontSize: 9.5, color: 'var(--text-7)', textTransform: 'uppercase', letterSpacing: 0.5, borderBottom: '1px solid var(--border)', marginBottom: 4 }}>
                    <span style={{ width: 128, flexShrink: 0 }}>Layer</span>
                    <span title="One bar per epoch, spanning the run's planned epochs">Δw/w · % dead · per epoch</span>
                    <span style={{ marginLeft: 'auto' }}>Health</span>
                  </div>
                )}
                {rows.map((row) =>
                  split ? (
                    // Full-bleed row (see column-header note) so the rule tracks
                    // the top Separator; inner wrappers (not the flex cells) carry
                    // the side insets, so cell padding can't shift the divider.
                    <div key={row.key} title={row.title} style={{ display: 'flex', alignItems: 'stretch', margin: '0 -16px', fontSize: 12 }}>
                      <div style={{ flexGrow: split.graphs, flexBasis: 0, minWidth: 0, display: 'flex' }}>
                        <div style={{ flex: 1, minWidth: 0, display: 'flex', alignItems: 'center', padding: '2px 0 2px 16px' }}>{row.bars}</div>
                      </div>
                      <ColDivider />
                      <div style={{ flexGrow: split.results, flexBasis: 0, minWidth: 0, display: 'flex' }}>
                        <div style={{ flex: 1, minWidth: 0, display: 'flex', alignItems: 'baseline', gap: 10, padding: '2px 16px 2px 12px' }}>
                          <span style={{ flex: 1, minWidth: 0, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{row.name}</span>
                          {meta(row)}
                        </div>
                      </div>
                    </div>
                  ) : (
                    <div key={row.key} title={row.title} style={{ display: 'flex', alignItems: 'baseline', gap: 10, padding: '2px 0', fontSize: 12 }}>
                      <span style={{ width: 128, flexShrink: 0, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{row.name}</span>
                      <div style={{ flex: 1, minWidth: 0, display: 'flex' }}>{row.bars}</div>
                      {meta(row)}
                    </div>
                  ),
                )}
              </div>
            )
          })}
        </div>
      )}
    </div>
  )
}
