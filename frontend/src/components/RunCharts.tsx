import { useEffect, useRef, useState } from 'react'
import type { RunEpoch } from '../store/graphStore'
import {
  chartDomain,
  discoverCharts,
  epochTicks,
  epochX,
  linearTicks,
  polylinePoints,
  tickLabel,
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
  if (seriesName(s) === 'val') return { color: 'var(--accent-2)', dash: '5 4' }
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
  title,
  series,
  planned,
  height,
  bestEpoch,
}: {
  title: string
  series: Series[]
  planned: number
  height: number
  bestEpoch?: number | null
}) {
  const [ref, width] = useContainerWidth()
  const { min, max } = chartDomain(series)
  const plotW = Math.max(width - M.left - M.right, 0)
  const plotH = height - M.top - M.bottom
  const yFor = (v: number) => M.top + plotH - ((v - min) / (max - min)) * plotH

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
      </div>

      {width > 0 && (
        <svg width={width} height={height} style={{ display: 'block', background: 'var(--field)', borderRadius: 4 }}>
          {/* y ticks: gridline + right-aligned value label */}
          {linearTicks(min, max).map((t) => (
            <g key={t}>
              <line x1={M.left} y1={yFor(t)} x2={M.left + plotW} y2={yFor(t)}
                stroke="var(--border)" strokeWidth={1} />
              <text x={M.left - 7} y={yFor(t)} textAnchor="end" dominantBaseline="middle"
                fontSize={9.5} fontFamily="monospace" fill="var(--text-6)">
                {tickLabel(t)}
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
                points={polylinePoints(s.values, planned, min, max, plotW, plotH)}
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
export function RunCharts({
  epochs,
  height = 84,
  bestEpoch = null,
}: {
  epochs: RunEpoch[]
  height?: number
  bestEpoch?: number | null
}) {
  if (epochs.length === 0) return null
  const planned = epochs[epochs.length - 1].epochs
  const charts = discoverCharts(epochs)
  if (charts.length === 0) return null
  return (
    <div style={{ display: 'flex', gap: 16, marginBottom: 12 }}>
      {charts.map((c) => (
        <Chart
          key={c.group}
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
