import type { RunEpoch } from '../store/graphStore'

// Pure geometry/series helpers for the run charts (SVG polylines). Kept free of
// React so the mapping from streamed epochs to pixels is unit-testable.

export interface Series {
  key: string
  values: number[]
}

// Extract per-metric series from the streamed epochs, keeping only metrics that
// actually appear (e.g. no val series without a val split / val_loader).
export function seriesFor(epochs: RunEpoch[], keys: string[]): Series[] {
  return keys
    .map((key) => ({
      key,
      values: epochs.filter((e) => key in e.metrics).map((e) => e.metrics[key]),
    }))
    .filter((s) => s.values.length > 0)
}

// Padded y-domain across every series in a chart. A flat/single-value domain is
// widened so the line sits mid-chart instead of degenerating.
export function chartDomain(series: Series[]): { min: number; max: number } {
  const all = series.flatMap((s) => s.values)
  let min = Math.min(...all)
  let max = Math.max(...all)
  if (min === max) {
    min -= 0.5
    max += 0.5
  }
  const pad = (max - min) * 0.08
  return { min: min - pad, max: max + pad }
}

// Map a series onto SVG polyline points. The x-axis spans the *planned* epoch
// count, so the curve visibly grows toward the right edge as training runs.
export function polylinePoints(
  values: number[],
  plannedEpochs: number,
  min: number,
  max: number,
  width: number,
  height: number
): string {
  const slots = Math.max(plannedEpochs, values.length, 2) - 1
  return values
    .map((v, i) => {
      const x = (i / slots) * width
      const y = height - ((v - min) / (max - min)) * height
      return `${x.toFixed(2)},${y.toFixed(2)}`
    })
    .join(' ')
}
