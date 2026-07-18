import { useEffect, useRef, useState } from 'react'
import { fmtMetric } from '../lib/epochMetrics'
import { useRunStore, type RunEpoch } from '../store/runStore'
import {
  chartDomain,
  chartTicks,
  clampDomain,
  comparisonCharts,
  discoverCharts,
  epochTicks,
  epochX,
  logUsable,
  mergedLossSeries,
  polylinePoints,
  xyPolylinePoints,
  yDomainValue,
  type ChartScale,
  type CompareRun,
  type Series,
  type XYSeries,
} from '../lib/runChart'

// Plot margins: room for y tick labels (left) and the epoch axis (bottom).
const M = { top: 8, right: 12, bottom: 22, left: 48 }

// Distinct series colors within a chart (theme tokens, so the charts adapt to
// light/dark for free). `val` keeps its familiar dashed secondary accent; other
// series (train, and a GAN's g/d) cycle the palette.
const PALETTE = ['var(--accent)', 'var(--accent-2)', 'var(--warn)', 'var(--error-bright)']
const seriesName = (s: Series) => s.label ?? s.key
function seriesStyle(s: Series, i: number): { color: string; dash?: string } {
  const name = seriesName(s)
  if (name === 'val') return { color: 'var(--accent-2)', dash: '5 4' }
  // A compared run's val series (name·val) keeps the dash so train/val stay
  // distinguishable across overlaid runs; the color cycles per series.
  if (name.endsWith('·val')) return { color: PALETTE[i % PALETTE.length], dash: '5 4' }
  return { color: PALETTE[i % PALETTE.length] }
}

// Track the rendered width so the SVG draws at true pixel coordinates — which
// is what lets text (ticks, labels) render undistorted inside it.
function useContainerWidth(): [React.RefObject<HTMLDivElement | null>, number] {
  const ref = useRef<HTMLDivElement>(null)
  const [width, setWidth] = useState(0)
  useEffect(() => {
    const el = ref.current
    if (!el) return
    const ro = new ResizeObserver((entries) => setWidth(entries[0].contentRect.width))
    ro.observe(el)
    return () => ro.disconnect()
  }, [])
  return [ref, width]
}

// The y-scale choice, persisted per chart group so "loss stays log" survives
// runs and reloads.
function useChartScale(group: string): [ChartScale, () => void] {
  const key = `lamplighter-chart-scale-${group}`
  const [choice, setChoice] = useState<ChartScale>(() =>
    localStorage.getItem(key) === 'log' ? 'log' : 'linear'
  )
  const toggle = () => {
    const next: ChartScale = choice === 'log' ? 'linear' : 'log'
    setChoice(next)
    localStorage.setItem(key, next)
  }
  return [choice, toggle]
}

function ScaleToggle({ choice, onToggle }: { choice: ChartScale; onToggle: () => void }) {
  return (
    <button
      onClick={onToggle}
      title={choice === 'log' ? 'Switch to a linear y axis' : 'Log-scale y axis — separates curves orders of magnitude apart'}
      style={{
        marginLeft: 'auto',
        background: choice === 'log' ? 'var(--surface)' : 'none',
        color: choice === 'log' ? 'var(--text-3)' : 'var(--text-6)',
        border: '1px solid var(--border)', borderRadius: 3, padding: '1px 7px',
        fontFamily: 'monospace', fontSize: 10, cursor: 'pointer', lineHeight: 1.4,
      }}
    >
      log-scale
    </button>
  )
}

// A fixed title slot so the series keys begin at the same x on every stacked
// chart — otherwise a longer title (e.g. the loss chart's granularity note)
// would push its train/val keys out of line with the accuracy chart's.
const CHART_TITLE_W = 62
const chartTitle: React.CSSProperties = {
  color: 'var(--text-4)', textTransform: 'uppercase', letterSpacing: 1,
  fontSize: 10, minWidth: CHART_TITLE_W, flexShrink: 0,
}

function Chart({
  group,
  title,
  series,
  planned,
  height,
  bestEpoch,
}: {
  group: string
  title: string
  series: Series[]
  planned: number
  height: number
  bestEpoch?: number | null
}) {
  const [ref, width] = useContainerWidth()
  // Log helps adversarial runs: one loss can sit orders of magnitude below the
  // other and flat-line on a linear axis. Accuracy is a bounded proportion —
  // log adds nothing there, so that chart has no toggle.
  const supportsLog = group !== 'acc'
  const [scaleChoice, toggleScale] = useChartScale(group)
  // A log axis plots positive values only — with none, render linear.
  const scale: ChartScale = supportsLog && scaleChoice === 'log' && logUsable(series) ? 'log' : 'linear'

  // Accuracy is a proportion — the pad must not show values outside [0, 1].
  const domain = chartDomain(series, scale)
  const { min, max } = group === 'acc' ? clampDomain(domain, 0, 1) : domain
  const plotW = Math.max(width - M.left - M.right, 0)
  const plotH = height - M.top - M.bottom
  // Domain space → pixel; raw values go through the scale transform first.
  const yPos = (t: number) => M.top + plotH - ((t - min) / (max - min)) * plotH
  const yFor = (v: number) => yPos(yDomainValue(v, min, max, scale))

  // Best-val marker: a ring on the val_loss point at the best epoch.
  const valSeries = series.find((s) => s.key === 'val_loss')
  const best =
    bestEpoch != null && valSeries && bestEpoch <= valSeries.values.length
      ? { x: M.left + epochX(bestEpoch, planned, plotW), y: yFor(valSeries.values[bestEpoch - 1]) }
      : null

  return (
    <div ref={ref} style={{ flex: 1, minWidth: 0 }}>
      {/* Key: line-style swatch + name + latest value per series. */}
      {/* Wraps: a big comparison adds legend lines, never intrinsic width —
          an unwrappable legend once pushed the charts past the viewport. */}
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '2px 14px', fontSize: 11, marginBottom: 4, alignItems: 'center', minWidth: 0 }}>
        <span style={chartTitle}>{title}</span>
        {series.map((s, i) => {
          const { color, dash } = seriesStyle(s, i)
          return (
            <span key={s.key} style={{ display: 'inline-flex', alignItems: 'center', gap: 5, color }}>
              <svg width={18} height={4} style={{ display: 'block' }}>
                <line x1={0} y1={2} x2={18} y2={2} stroke={color} strokeWidth={2} strokeDasharray={dash} />
              </svg>
              {seriesName(s)} {fmtMetric(s.values[s.values.length - 1])}
            </span>
          )
        })}
        {best && <span style={{ color: 'var(--warn)' }}>◦ best @{bestEpoch}</span>}
        {supportsLog && <ScaleToggle choice={scaleChoice} onToggle={toggleScale} />}
      </div>

      {width > 0 && (
        <svg width={width} height={height} style={{ display: 'block', background: 'var(--field)', borderRadius: 4 }}>
          {/* y ticks: gridline + right-aligned value label (real quantities —
              a log tick's label is 10^position) */}
          {chartTicks(min, max, scale).map((t) => (
            <g key={t.value}>
              <line x1={M.left} y1={yPos(t.value)} x2={M.left + plotW} y2={yPos(t.value)}
                stroke="var(--border)" strokeWidth={1} />
              <text x={M.left - 7} y={yPos(t.value)} textAnchor="end" dominantBaseline="middle"
                fontSize={9.5} fontFamily="monospace" fill="var(--text-6)">
                {t.label}
              </text>
            </g>
          ))}

          {/* x axis: baseline, epoch tick marks + labels, axis title */}
          <line x1={M.left} y1={M.top + plotH} x2={M.left + plotW} y2={M.top + plotH}
            stroke="var(--border)" strokeWidth={1} />
          {epochTicks(planned).map((e) => {
            const x = M.left + epochX(e, planned, plotW)
            return (
              <g key={e}>
                <line x1={x} y1={M.top + plotH} x2={x} y2={M.top + plotH + 3}
                  stroke="var(--text-7)" strokeWidth={1} />
                <text x={x} y={height - 6} textAnchor="middle" fontSize={9.5}
                  fontFamily="monospace" fill="var(--text-6)">
                  {e}
                </text>
              </g>
            )
          })}
          {/* axis title in the (otherwise empty) bottom-left corner */}
          <text x={M.left - 7} y={height - 6} textAnchor="end" fontSize={9.5}
            fontFamily="monospace" fill="var(--text-7)">
            epoch
          </text>

          {/* the series */}
          {series.map((s, i) => {
            const { color, dash } = seriesStyle(s, i)
            return (
              <polyline
                key={s.key}
                points={polylinePoints(s.values, planned, min, max, plotW, plotH, scale)}
                transform={`translate(${M.left}, ${M.top})`}
                fill="none"
                stroke={color}
                strokeWidth={1.5}
                strokeDasharray={dash}
              />
            )
          })}

          {/* best-val epoch marker */}
          {best && (
            <circle cx={best.x} cy={best.y} r={4} fill="none" stroke="var(--warn)" strokeWidth={1.5} />
          )}
        </svg>
      )}
    </div>
  )
}

// The merged loss chart: step-resolution curves on a continuous epoch axis
// (x = 0 at the run's start, integer positions = epoch ends), with epoch-only
// series (val) overlaid at their integers and the best-val ring on top. Step
// series REPLACE their same-named epoch series — one quantity, one line, at
// the finer granularity; without steps (IterableDataset, old runs) it renders
// the epoch series alone, which is the pre-merge chart.
function LossChart({
  epochs,
  lossKeys,
  planned,
  height,
  bestEpoch,
}: {
  epochs: RunEpoch[]
  lossKeys: string[]
  planned: number
  height: number
  bestEpoch?: number | null
}) {
  const stepMetrics = useRunStore((s) => s.stepMetrics)
  const [ref, width] = useContainerWidth()
  const [scaleChoice, toggleScale] = useChartScale('loss')

  const merged: XYSeries[] = mergedLossSeries(epochs, lossKeys, stepMetrics)
  const perStep = merged.some((s) => s.raw)
  const asSeries = merged.map((s) => ({ key: s.key, values: s.points.map((p) => p.y) }))
  const scale: ChartScale = scaleChoice === 'log' && logUsable(asSeries) ? 'log' : 'linear'
  const { min, max } = chartDomain(asSeries, scale)

  // One hue per METRIC, shared by its raw layer and its mean line — keyed by
  // label, not series index (indices shift when raw layers are present).
  const labels = [...new Set(merged.map((s) => s.label))]
  const styleOf = (s: XYSeries) =>
    s.label === 'val'
      ? { color: 'var(--accent-2)', dash: '5 4' }
      : { color: PALETTE[labels.indexOf(s.label) % PALETTE.length], dash: undefined }

  const plotW = Math.max(width - M.left - M.right, 0)
  const plotH = height - M.top - M.bottom
  const xMax = Math.max(planned, ...merged.flatMap((s) => s.points.map((p) => p.x)), 1)
  const xPos = (x: number) => M.left + (x / xMax) * plotW
  const yPos = (t: number) => M.top + plotH - ((t - min) / (max - min)) * plotH
  const yFor = (v: number) => yPos(yDomainValue(v, min, max, scale))

  // Best-val marker: a ring on the val point at the best epoch — looked up by
  // its TRUE epoch number (a resumed run's epochs don't start at 1).
  const bestVal =
    bestEpoch != null ? epochs.find((e) => e.epoch === bestEpoch)?.metrics['val_loss'] : undefined
  const best = bestVal !== undefined ? { x: xPos(bestEpoch!), y: yFor(bestVal) } : null

  return (
    <div ref={ref} style={{ flex: 1, minWidth: 0 }}>
      {/* Wraps: a big comparison adds legend lines, never intrinsic width —
          an unwrappable legend once pushed the charts past the viewport. */}
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '2px 14px', fontSize: 11, marginBottom: 4, alignItems: 'center', minWidth: 0 }}>
        <span style={chartTitle}>loss</span>
        {merged.filter((s) => !s.raw).map((s) => {
          const { color, dash } = styleOf(s)
          const last = s.points[s.points.length - 1]
          return (
            <span key={s.key} style={{ display: 'inline-flex', alignItems: 'center', gap: 5, color }}>
              <svg width={18} height={4} style={{ display: 'block' }}>
                <line x1={0} y1={2} x2={18} y2={2} stroke={color} strokeWidth={2} strokeDasharray={dash} />
              </svg>
              {s.label} {last ? fmtMetric(last.y) : ''}
            </span>
          )
        })}
        {/* best@ is a value, so it groups with the train/val keys; the
            granularity note trails everything (and never pushes the keys out of
            line with the accuracy chart above/below). */}
        {best && <span style={{ color: 'var(--warn)' }}>◦ best @{bestEpoch}</span>}
        {perStep && <span style={{ color: 'var(--text-7)', fontSize: 10 }}>steps + epoch mean</span>}
        <ScaleToggle choice={scaleChoice} onToggle={toggleScale} />
      </div>

      {width > 0 && (
        <svg width={width} height={height} style={{ display: 'block', background: 'var(--field)', borderRadius: 4 }}>
          {chartTicks(min, max, scale).map((t) => (
            <g key={t.value}>
              <line x1={M.left} y1={yPos(t.value)} x2={M.left + plotW} y2={yPos(t.value)}
                stroke="var(--border)" strokeWidth={1} />
              <text x={M.left - 7} y={yPos(t.value)} textAnchor="end" dominantBaseline="middle"
                fontSize={9.5} fontFamily="monospace" fill="var(--text-6)">
                {t.label}
              </text>
            </g>
          ))}

          <line x1={M.left} y1={M.top + plotH} x2={M.left + plotW} y2={M.top + plotH}
            stroke="var(--border)" strokeWidth={1} />
          {epochTicks(planned).map((e) => (
            <g key={e}>
              <line x1={xPos(e)} y1={M.top + plotH} x2={xPos(e)} y2={M.top + plotH + 3}
                stroke="var(--text-7)" strokeWidth={1} />
              <text x={xPos(e)} y={height - 6} textAnchor="middle" fontSize={9.5}
                fontFamily="monospace" fill="var(--text-6)">
                {e}
              </text>
            </g>
          ))}
          <text x={M.left - 7} y={height - 6} textAnchor="end" fontSize={9.5}
            fontFamily="monospace" fill="var(--text-7)">
            epoch
          </text>

          {merged.map((s) => {
            const { color, dash } = styleOf(s)
            return (
              <polyline
                key={s.key}
                points={xyPolylinePoints(s.points, xMax, min, max, plotW, plotH, scale)}
                transform={`translate(${M.left}, ${M.top})`}
                fill="none"
                stroke={color}
                strokeWidth={s.raw ? 1 : 1.5}
                strokeOpacity={s.raw ? 0.35 : 1}
                strokeDasharray={dash}
              />
            )
          })}

          {best && (
            <circle cx={best.x} cy={best.y} r={4} fill="none" stroke="var(--warn)" strokeWidth={1.5} />
          )}
        </svg>
      )}
    </div>
  )
}

// Live loss/accuracy curves for a streaming (or finished) run. The x-axis spans
// the planned epoch count, so curves grow toward the right edge as epochs land.
// `compare` overlays stored checkpoints' curves (labeled name·series) onto the
// same charts — the x-axis stretches to the longest run.
export function RunCharts({
  epochs,
  height = 84,
  bestEpoch = null,
  compare = [],
  stacked = false,
}: {
  epochs: RunEpoch[]
  height?: number
  bestEpoch?: number | null
  compare?: CompareRun[]
  // Stack loss over accuracy (a narrow charts column) instead of side by side.
  stacked?: boolean
}) {
  if (epochs.length === 0 && compare.length === 0) return null
  const planned = Math.max(
    epochs.length > 0 ? epochs[epochs.length - 1].epochs : 0,
    ...compare.flatMap((r) => Object.values(r.history).map((v) => v.length)),
    1
  )
  const charts = compare.length > 0 ? comparisonCharts(epochs, compare) : discoverCharts(epochs)
  if (charts.length === 0) return null
  return (
    <div style={{ display: 'flex', flexDirection: stacked ? 'column' : 'row', gap: 16, marginBottom: 12 }}>
      {charts.map((c) =>
        // The loss chart merges in the step-resolution stream (single-run view
        // only — compare overlays are epoch histories, kept as plain lines).
        c.group === 'loss' && compare.length === 0 ? (
          <LossChart
            key={c.group}
            epochs={epochs}
            lossKeys={c.series.map((s) => s.key)}
            planned={planned}
            height={height}
            bestEpoch={bestEpoch}
          />
        ) : (
          <Chart
            key={c.group}
            group={c.group}
            title={c.title}
            series={c.series}
            planned={planned}
            height={height}
            // The best-val ring belongs to the loss chart (only supervised has val_loss).
            bestEpoch={c.group === 'loss' ? bestEpoch : null}
          />
        )
      )}
    </div>
  )
}
