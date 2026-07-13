import { useEffect, useRef, useState } from 'react'
import { useRunStore } from '../store/runStore'
import { linearTicks, tickLabel } from '../lib/runChart'

// Distinct series colors (theme tokens, so they adapt to light/dark).
const PALETTE = ['var(--accent)', 'var(--accent-2)', 'var(--warn)', 'var(--error-bright)']

// A compact live per-step metrics curve — finer than the epoch charts, so the
// shape *within* an epoch is visible (early divergence, an LR-warmup dip, batch
// instability). One line per streamed metric (supervised train_loss; a GAN's g/d;
// a VAE's recon/kl), sharing a self-scaled y-axis. The x-axis is pinned to the
// run's total step count when known. Live-only: self-hides until points stream and
// clears on a new run.
export function StepLossChart({ height = 84 }: { height?: number }) {
  const stepMetrics = useRunStore((s) => s.stepMetrics)
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
  if (stepMetrics.length < 2) return null

  // Series keys in first-seen order (one per streamed metric).
  const keys: string[] = []
  for (const p of stepMetrics) for (const k of Object.keys(p.metrics)) if (!keys.includes(k)) keys.push(k)

  const M = { top: 8, right: 12, bottom: 20, left: 44 }
  const plotW = Math.max(width - M.left - M.right, 0)
  const plotH = height - M.top - M.bottom

  // Shared y-domain across every series.
  const all = stepMetrics.flatMap((p) => Object.values(p.metrics))
  const min = Math.min(...all)
  const max = Math.max(...all)
  const span = max - min || 1
  const yAt = (v: number) => M.top + plotH - ((v - min) / span) * plotH

  // Fix the x-axis to the run's known total step count when we have it, so the
  // curve grows toward a fixed right edge (like the epoch chart) instead of the
  // axis rescaling. Fall back to the streamed range when it's unknown.
  const lastStep = stepMetrics[stepMetrics.length - 1].step
  const xMin = stepTotal > 0 ? 1 : stepMetrics[0].step
  const xMax = stepTotal > 0 ? stepTotal : lastStep
  const xSpan = xMax - xMin || 1
  const xAt = (step: number) => M.left + ((step - xMin) / xSpan) * plotW
  // Evenly spaced integer step ticks (deduped, since a narrow early range can
  // round two ticks to the same step).
  const xTicks = [...new Set(linearTicks(xMin, xMax, 4).map(Math.round))]

  const colorOf = (i: number) => PALETTE[i % PALETTE.length]
  const pointsFor = (k: string) =>
    stepMetrics
      .filter((p) => k in p.metrics)
      .map((p) => `${xAt(p.step).toFixed(1)},${yAt(p.metrics[k]).toFixed(1)}`)
      .join(' ')
  const latestOf = (k: string) => {
    for (let i = stepMetrics.length - 1; i >= 0; i--) if (k in stepMetrics[i].metrics) return stepMetrics[i].metrics[k]
    return undefined
  }

  return (
    <div ref={ref} style={{ marginBottom: 12 }}>
      <div style={{ display: 'flex', gap: 12, fontSize: 11, marginBottom: 4, alignItems: 'center', flexWrap: 'wrap' }}>
        <span style={{ color: 'var(--text-4)', textTransform: 'uppercase', letterSpacing: 1, fontSize: 10 }}>
          loss / step
        </span>
        {keys.map((k, i) => (
          <span key={k} style={{ display: 'inline-flex', alignItems: 'center', gap: 5, color: colorOf(i) }}>
            <svg width={18} height={4} style={{ display: 'block' }}>
              <line x1={0} y1={2} x2={18} y2={2} stroke={colorOf(i)} strokeWidth={2} />
            </svg>
            {k} {(latestOf(k) ?? 0).toFixed(4)}
          </span>
        ))}
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

          {keys.map((k, i) => (
            <polyline key={k} points={pointsFor(k)} fill="none" stroke={colorOf(i)} strokeWidth={1.5} />
          ))}
        </svg>
      )}
    </div>
  )
}
