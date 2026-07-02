import type { RunEpoch } from '../store/graphStore'
import { chartDomain, polylinePoints, seriesFor, type Series } from '../lib/runChart'

// Internal SVG coordinate space; the element scales to its container. Strokes
// use non-scaling-stroke so the aspect distortion doesn't fatten lines.
const W = 300
const H = 84

// Series styling: train = solid accent, val = dashed secondary accent — theme
// tokens, so the charts adapt to light/dark for free.
const seriesColor = (key: string) => (key.startsWith('val') ? 'var(--accent-2)' : 'var(--accent)')
const seriesDash = (key: string) => (key.startsWith('val') ? '5 4' : undefined)
const seriesLabel = (key: string) => (key.startsWith('val') ? 'val' : 'train')

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
  const { min, max } = chartDomain(series)
  return (
    <div style={{ flex: 1, minWidth: 0 }}>
      {/* Header: metric name + legend with the latest value per series. */}
      <div style={{ display: 'flex', gap: 12, fontSize: 11, marginBottom: 4, alignItems: 'baseline' }}>
        <span style={{ color: 'var(--text-4)', textTransform: 'uppercase', letterSpacing: 1, fontSize: 10 }}>
          {title}
        </span>
        {series.map((s) => (
          <span key={s.key} style={{ color: seriesColor(s.key) }}>
            {seriesLabel(s.key)} {s.values[s.values.length - 1].toFixed(4)}
          </span>
        ))}
      </div>
      <svg
        viewBox={`0 0 ${W} ${H}`}
        preserveAspectRatio="none"
        style={{ width: '100%', height, display: 'block', background: 'var(--field)', borderRadius: 4 }}
      >
        {[0.25, 0.5, 0.75].map((f) => (
          <line key={f} x1={0} y1={H * f} x2={W} y2={H * f} stroke="var(--border)" strokeWidth={1}
            vectorEffect="non-scaling-stroke" />
        ))}
        {series.map((s) => (
          <polyline
            key={s.key}
            points={polylinePoints(s.values, planned, min, max, W, H)}
            fill="none"
            stroke={seriesColor(s.key)}
            strokeWidth={1.5}
            strokeDasharray={seriesDash(s.key)}
            vectorEffect="non-scaling-stroke"
          />
        ))}
      </svg>
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
