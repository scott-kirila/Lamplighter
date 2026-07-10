import type { RunEpoch } from '../store/graphStore'

// Pure geometry/series helpers for the run charts (SVG polylines). Kept free of
// React so the mapping from streamed epochs to pixels is unit-testable.

export interface Series {
  key: string
  values: number[]
  label?: string // series name within a chart (the metric key's prefix)
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

export interface ChartSpec {
  group: string
  title: string
  series: Series[]
}

const GROUP_TITLE: Record<string, string> = { loss: 'loss', acc: 'accuracy' }
// loss first, then accuracy, then anything else — stable within a group.
const groupOrder = (g: string) => (g === 'loss' ? 0 : g === 'acc' ? 1 : 2)

// Discover the charts to draw from whatever metrics a run streams — no hardcoded
// key list, so a recipe that reports g_loss/d_loss (GAN) charts just like
// train_loss/val_loss (supervised). A metric key splits on its first underscore:
// the prefix is the series name (train/val/g/d), the suffix is the chart group
// (loss/acc/…). So train_loss+val_loss share the "loss" chart, g_loss+d_loss
// share it too, and train_acc+val_acc form the "accuracy" chart.
export function discoverCharts(epochs: RunEpoch[]): ChartSpec[] {
  const keys: string[] = []
  for (const e of epochs) for (const k of Object.keys(e.metrics)) if (!keys.includes(k)) keys.push(k)
  const groups = new Map<string, Series[]>()
  for (const key of keys) {
    const us = key.indexOf('_')
    const label = us === -1 ? key : key.slice(0, us)
    const group = us === -1 ? key : key.slice(us + 1)
    const values = epochs.filter((e) => key in e.metrics).map((e) => e.metrics[key])
    if (values.length === 0) continue
    if (!groups.has(group)) groups.set(group, [])
    groups.get(group)!.push({ key, label, values })
  }
  return [...groups.entries()]
    .map(([group, series]) => ({ group, title: GROUP_TITLE[group] ?? group, series }))
    .sort((a, b) => groupOrder(a.group) - groupOrder(b.group))
}

// A stored run selected for comparison: its name + full per-epoch history
// (GET /api/checkpoints/{name}/history).
export interface CompareRun {
  name: string
  history: Record<string, number[]>
}

// The comparison view's charts: the live/last run's series (plain labels) plus
// each compared checkpoint's, tagged "name·prefix" (run-a·val) with a unique
// key — all grouped by metric suffix exactly like discoverCharts, so a compared
// GAN overlays its g/d losses the same way a supervised run overlays train/val.
export function comparisonCharts(epochs: RunEpoch[], compare: CompareRun[]): ChartSpec[] {
  const groups = new Map<string, Series[]>()
  const add = (key: string, label: string, group: string, values: number[]) => {
    if (values.length === 0) return
    if (!groups.has(group)) groups.set(group, [])
    groups.get(group)!.push({ key, label, values })
  }
  const split = (key: string) => {
    const us = key.indexOf('_')
    return us === -1 ? { label: key, group: key } : { label: key.slice(0, us), group: key.slice(us + 1) }
  }

  const liveKeys: string[] = []
  for (const e of epochs) for (const k of Object.keys(e.metrics)) if (!liveKeys.includes(k)) liveKeys.push(k)
  for (const key of liveKeys) {
    const { label, group } = split(key)
    add(key, label, group, epochs.filter((e) => key in e.metrics).map((e) => e.metrics[key]))
  }
  for (const run of compare) {
    for (const [key, values] of Object.entries(run.history)) {
      const { label, group } = split(key)
      add(`${run.name}:${key}`, `${run.name}·${label}`, group, values)
    }
  }
  return [...groups.entries()]
    .map(([group, series]) => ({ group, title: GROUP_TITLE[group] ?? group, series }))
    .sort((a, b) => groupOrder(a.group) - groupOrder(b.group))
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
