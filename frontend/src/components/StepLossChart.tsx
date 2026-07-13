import { useEffect, useRef, useState } from 'react'
import { useRunStore } from '../store/runStore'
import { linearTicks, tickLabel } from '../lib/runChart'

// A compact live per-step loss curve — finer than the epoch charts, so the shape
// *within* an epoch is visible: an early divergence, an LR-warmup dip, batch-scale
// instability. Self-scaling to the streamed window; x is the (throttled) step
// index. Live-only, so it self-hides until points stream and clears on a new run.
export function StepLossChart({ height = 72 }: { height?: number }) {
  const stepLoss = useRunStore((s) => s.stepLoss)
  const stepTotal = useRunStore((s) => s.stepTotal)
  const ref = useRef<HTMLDivElement>(null)
  const [width, setWidth] = useState(0)
  useEffect(() => {
    const el = ref.current
    if (!el) return
    const ro = new ResizeObserver((entries) => setWidth(entries[0].contentRect.width))
    ro.observe(el)
    return () => ro.disconnect()
  }, [])

  // Needs at least a segment to draw.
  if (stepLoss.length < 2) return null

  const M = { top: 8, right: 12, bottom: 20, left: 44 }
  const plotW = Math.max(width - M.left - M.right, 0)
  const plotH = height - M.top - M.bottom
  const losses = stepLoss.map((p) => p.loss)
  const min = Math.min(...losses)
  const max = Math.max(...losses)
  const span = max - min || 1
  // Fix the x-axis to the run's known total step count when we have it, so the
  // curve grows toward a fixed right edge (like the epoch chart) instead of the
  // axis rescaling each update. Fall back to the streamed range if it's unknown
  // (e.g. a loader without __len__).
  const lastStep = stepLoss[stepLoss.length - 1].step
  const xMin = stepTotal > 0 ? 1 : stepLoss[0].step
  const xMax = stepTotal > 0 ? stepTotal : lastStep
  const xSpan = xMax - xMin || 1
  const xAt = (step: number) => M.left + ((step - xMin) / xSpan) * plotW
  // Evenly spaced integer step ticks across the axis (deduped, since a narrow
  // early range can round two ticks to the same step).
  const xTicks = [...new Set(linearTicks(xMin, xMax, 4).map(Math.round))]
  const points = stepLoss
    .map((p) => `${xAt(p.step).toFixed(1)},${(M.top + plotH - ((p.loss - min) / span) * plotH).toFixed(1)}`)
    .join(' ')
  const latest = losses[losses.length - 1]

  return (
    <div ref={ref} style={{ marginBottom: 12 }}>
      <div style={{ display: 'flex', gap: 10, fontSize: 11, marginBottom: 4, alignItems: 'center' }}>
        <span style={{ color: 'var(--text-4)', textTransform: 'uppercase', letterSpacing: 1, fontSize: 10 }}>
          loss / step
        </span>
        <span style={{ color: 'var(--accent)' }}>{latest.toFixed(4)}</span>
      </div>
      {width > 0 && (
        <svg width={width} height={height} style={{ display: 'block', background: 'var(--field)', borderRadius: 4 }}>
          <text x={M.left - 6} y={M.top + 4} textAnchor="end" fontSize={9.5} fontFamily="monospace" fill="var(--text-6)">
            {tickLabel(max)}
          </text>
          <text x={M.left - 6} y={M.top + plotH} textAnchor="end" fontSize={9.5} fontFamily="monospace" fill="var(--text-6)">
            {tickLabel(min)}
          </text>

          {/* x axis: baseline, step tick marks + labels, axis title */}
          <line x1={M.left} y1={M.top + plotH} x2={M.left + plotW} y2={M.top + plotH} stroke="var(--border)" strokeWidth={1} />
          {xTicks.map((t) => (
            <g key={t}>
              <line x1={xAt(t)} y1={M.top + plotH} x2={xAt(t)} y2={M.top + plotH + 3} stroke="var(--text-7)" strokeWidth={1} />
              <text x={xAt(t)} y={height - 6} textAnchor="middle" fontSize={9.5} fontFamily="monospace" fill="var(--text-6)">
                {t}
              </text>
            </g>
          ))}
          <text x={M.left - 7} y={height - 6} textAnchor="end" fontSize={9.5} fontFamily="monospace" fill="var(--text-7)">
            step
          </text>

          <polyline points={points} fill="none" stroke="var(--accent)" strokeWidth={1.5} />
        </svg>
      )}
    </div>
  )
}
