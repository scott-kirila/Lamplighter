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

// Evenly spaced y-axis tick values across a (padded) domain, min → max.
export function linearTicks(min: number, max: number, count = 4): number[] {
  const step = (max - min) / (count - 1)
  return Array.from({ length: count }, (_, i) => min + i * step)
}

// Integer x-axis ticks over the planned epochs, using a "nice" step (1/2/5×10ⁿ)
// so at most ~6 labels appear: 12 → [2,4,…,12], 100 → [20,40,…,100].
export function epochTicks(planned: number, maxTicks = 6): number[] {
  if (planned <= 1) return [1]
  let step = 1
  while (planned / step > maxTicks) {
    step = step % 10 === 2 ? (step / 2) * 5 : step * 2 // 1 → 2 → 5 → 10 → 20 …
  }
  const ticks: number[] = []
  for (let e = step; e <= planned; e += step) ticks.push(e)
  return ticks
}

// Compact tick label: ~3 significant digits, no trailing zero noise.
export function tickLabel(v: number): string {
  if (v === 0) return '0'
  return String(parseFloat(v.toPrecision(3)))
}

// X pixel for an epoch number, matching polylinePoints' slot math.
export function epochX(epoch: number, planned: number, width: number): number {
  const slots = Math.max(planned, 2) - 1
  return ((epoch - 1) / slots) * width
}
