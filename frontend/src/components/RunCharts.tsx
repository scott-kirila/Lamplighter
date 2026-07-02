import { useEffect, useRef, useState } from 'react'
import type { RunEpoch } from '../store/graphStore'
import {
  chartDomain,
  epochTicks,
  epochX,
  linearTicks,
  polylinePoints,
  seriesFor,
  tickLabel,
  type Series,
} from '../lib/runChart'

// Plot margins: room for y tick labels (left) and the epoch axis (bottom).
const M = { top: 8, right: 12, bottom: 22, left: 48 }

// Series styling: train = solid accent, val = dashed secondary accent — theme
// tokens, so the charts adapt to light/dark for free.
const seriesColor = (key: string) => (key.startsWith('val') ? 'var(--accent-2)' : 'var(--accent)')
const seriesDash = (key: string) => (key.startsWith('val') ? '5 4' : undefined)
const seriesLabel = (key: string) => (key.startsWith('val') ? 'val' : 'train')

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
}: {
  title: string
  series: Series[]
  planned: number
  height: number
}) {
  const [ref, width] = useContainerWidth()
  const { min, max } = chartDomain(series)
  const plotW = Math.max(width - M.left - M.right, 0)
  const plotH = height - M.top - M.bottom
  const yFor = (v: number) => M.top + plotH - ((v - min) / (max - min)) * plotH

  return (
    <div ref={ref} style={{ flex: 1, minWidth: 0 }}>
      {/* Key: line-style swatch + name + latest value per series. */}
      <div style={{ display: 'flex', gap: 14, fontSize: 11, marginBottom: 4, alignItems: 'center' }}>
        <span style={{ color: 'var(--text-4)', textTransform: 'uppercase', letterSpacing: 1, fontSize: 10 }}>
          {title}
        </span>
        {series.map((s) => (
          <span key={s.key} style={{ display: 'inline-flex', alignItems: 'center', gap: 5, color: seriesColor(s.key) }}>
            <svg width={18} height={4} style={{ display: 'block' }}>
              <line x1={0} y1={2} x2={18} y2={2} stroke={seriesColor(s.key)} strokeWidth={2}
                strokeDasharray={seriesDash(s.key)} />
            </svg>
            {seriesLabel(s.key)} {s.values[s.values.length - 1].toFixed(4)}
          </span>
        ))}
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
          {series.map((s) => (
            <polyline
              key={s.key}
              points={polylinePoints(s.values, planned, min, max, plotW, plotH)}
              transform={`translate(${M.left}, ${M.top})`}
              fill="none"
              stroke={seriesColor(s.key)}
              strokeWidth={1.5}
              strokeDasharray={seriesDash(s.key)}
            />
          ))}
        </svg>
      )}
    </div>
  )
}

// Live loss/accuracy curves for a streaming (or finished) run. The x-axis spans
// the planned epoch count, so curves grow toward the right edge as epochs land.
export function RunCharts({ epochs, height = 84 }: { epochs: RunEpoch[]; height?: number }) {
  if (epochs.length === 0) return null
  const planned = epochs[epochs.length - 1].epochs
  const loss = seriesFor(epochs, ['train_loss', 'val_loss'])
  const acc = seriesFor(epochs, ['train_acc', 'val_acc'])
  if (loss.length === 0 && acc.length === 0) return null
  return (
    <div style={{ display: 'flex', gap: 16, marginBottom: 12 }}>
      {loss.length > 0 && <Chart title="loss" series={loss} planned={planned} height={height} />}
      {acc.length > 0 && <Chart title="accuracy" series={acc} planned={planned} height={height} />}
    </div>
  )
}
