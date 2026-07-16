import { useEffect, useRef, useState } from 'react'
import type { RunEpoch } from '../store/runStore'
import {
  chartDomain,
  chartTicks,
  comparisonCharts,
  discoverCharts,
  epochTicks,
  epochX,
  logUsable,
  polylinePoints,
  type ChartScale,
  type CompareRun,
  type Series,
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
  // The y-scale, persisted per chart group so "loss stays log" survives runs
  // and reloads. Log helps adversarial runs: one loss can sit orders of
  // magnitude below the other and flat-line on a linear axis. Accuracy is a
  // bounded proportion — log adds nothing there, so that chart has no toggle.
  const supportsLog = group !== 'acc'
  const scaleKey = `lamplighter-chart-scale-${group}`
  const [scaleChoice, setScaleChoice] = useState<ChartScale>(() =>
    localStorage.getItem(scaleKey) === 'log' ? 'log' : 'linear'
  )
  const toggleScale = () => {
    const next: ChartScale = scaleChoice === 'log' ? 'linear' : 'log'
    setScaleChoice(next)
    localStorage.setItem(scaleKey, next)
  }
  // A log axis plots positive values only — with none, render linear.
  const scale: ChartScale = supportsLog && scaleChoice === 'log' && logUsable(series) ? 'log' : 'linear'

  const { min, max } = chartDomain(series, scale)
  const plotW = Math.max(width - M.left - M.right, 0)
  const plotH = height - M.top - M.bottom
  // Domain space → pixel; raw values go through the scale transform first.
  const yPos = (t: number) => M.top + plotH - ((t - min) / (max - min)) * plotH
  const yFor = (v: number) =>
    yPos(scale === 'log' ? (v > 0 ? Math.max(min, Math.min(max, Math.log10(v))) : min) : v)

  // Best-val marker: a ring on the val_loss point at the best epoch.
  const valSeries = series.find((s) => s.key === 'val_loss')
  const best =
    bestEpoch != null && valSeries && bestEpoch <= valSeries.values.length
      ? { x: M.left + epochX(bestEpoch, planned, plotW), y: yFor(valSeries.values[bestEpoch - 1]) }
      : null

  return (
    <div ref={ref} style={{ flex: 1, minWidth: 0 }}>
      {/* Key: line-style swatch + name + latest value per series. */}
      <div style={{ display: 'flex', gap: 14, fontSize: 11, marginBottom: 4, alignItems: 'center' }}>
        <span style={{ color: 'var(--text-4)', textTransform: 'uppercase', letterSpacing: 1, fontSize: 10 }}>
          {title}
        </span>
        {series.map((s, i) => {
          const { color, dash } = seriesStyle(s, i)
          return (
            <span key={s.key} style={{ display: 'inline-flex', alignItems: 'center', gap: 5, color }}>
              <svg width={18} height={4} style={{ display: 'block' }}>
                <line x1={0} y1={2} x2={18} y2={2} stroke={color} strokeWidth={2} strokeDasharray={dash} />
              </svg>
              {seriesName(s)} {s.values[s.values.length - 1].toFixed(4)}
            </span>
          )
        })}
        {best && <span style={{ color: 'var(--warn)' }}>◦ best @{bestEpoch}</span>}
        {supportsLog && (
        <button
          onClick={toggleScale}
          title={scaleChoice === 'log' ? 'Switch to a linear y axis' : 'Log-scale y axis — separates curves orders of magnitude apart'}
          style={{
            marginLeft: 'auto',
            background: scaleChoice === 'log' ? 'var(--surface)' : 'none',
            color: scaleChoice === 'log' ? 'var(--text-3)' : 'var(--text-6)',
            border: '1px solid var(--border)', borderRadius: 3, padding: '1px 7px',
            fontFamily: 'monospace', fontSize: 10, cursor: 'pointer', lineHeight: 1.4,
          }}
        >
          log
        </button>
        )}
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

// Live loss/accuracy curves for a streaming (or finished) run. The x-axis spans
// the planned epoch count, so curves grow toward the right edge as epochs land.
// `compare` overlays stored checkpoints' curves (labeled name·series) onto the
// same charts — the x-axis stretches to the longest run.
export function RunCharts({
  epochs,
  height = 84,
  bestEpoch = null,
  compare = [],
}: {
  epochs: RunEpoch[]
  height?: number
  bestEpoch?: number | null
  compare?: CompareRun[]
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
    <div style={{ display: 'flex', gap: 16, marginBottom: 12 }}>
      {charts.map((c) => (
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
      ))}
    </div>
  )
}
