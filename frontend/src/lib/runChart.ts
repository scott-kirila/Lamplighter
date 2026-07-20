import type { RunEpoch } from '../store/runStore'

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
// A metric key's chart group — the suffix after the first underscore
// (train_loss → loss, mean_return → return); bare keys group as themselves.
// The step stream routes by the same rule, so an RL run's per-episode returns
// land on the RETURN chart, not the loss chart.
export function metricGroup(key: string): string {
  const us = key.indexOf('_')
  return us === -1 ? key : key.slice(us + 1)
}

export function discoverCharts(epochs: RunEpoch[]): ChartSpec[] {
  const keys: string[] = []
  for (const e of epochs) for (const k of Object.keys(e.metrics)) if (!keys.includes(k)) keys.push(k)
  const groups = new Map<string, Series[]>()
  for (const key of keys) {
    const us = key.indexOf('_')
    const label = us === -1 ? key : key.slice(0, us)
    const group = metricGroup(key)
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

// The y-axis scale. Log helps adversarial runs, where one loss can sit orders
// of magnitude below the other and flat-lines on a linear axis.
export type ChartScale = 'linear' | 'log'

// Whether a log axis has anything to show: it plots positive values only.
export function logUsable(series: Series[]): boolean {
  return series.some((s) => s.values.some((v) => v > 0))
}

// A value in domain space: raw for linear, log10 for log. Non-positive values
// have no log — they clamp to the floor (the standard log-plot treatment), so
// the polyline stays connected instead of breaking.
function domainValue(v: number, scale: ChartScale, floor: number): number {
  return scale === 'log' ? (v > 0 ? Math.log10(v) : floor) : v
}

// The same, clamped into a [min, max] domain — what a chart's y-mapping wants.
export function yDomainValue(v: number, min: number, max: number, scale: ChartScale): number {
  return Math.max(min, Math.min(max, domainValue(v, scale, min)))
}

// Padded y-domain across every series in a chart, in domain space (log10 units
// on a log axis, over the positive values only). A flat/single-value domain is
// widened so the line sits mid-chart instead of degenerating.
export function chartDomain(series: Series[], scale: ChartScale = 'linear'): { min: number; max: number } {
  const all = series.flatMap((s) => s.values)
  const domain = scale === 'log' ? all.filter((v) => v > 0).map(Math.log10) : all
  let min = Math.min(...domain)
  let max = Math.max(...domain)
  if (min === max) {
    min -= 0.5
    max += 0.5
  }
  const pad = (max - min) * 0.08
  return { min: min - pad, max: max + pad }
}

// Clamp a padded domain to hard data bounds — a proportion like accuracy can't
// exceed [0, 1], so the 8% pad must not invent a "1.01" top tick.
export function clampDomain(
  d: { min: number; max: number },
  lo: number,
  hi: number
): { min: number; max: number } {
  return { min: Math.max(d.min, lo), max: Math.min(d.max, hi) }
}

// Map a series onto SVG polyline points. The x-axis spans the *planned* epoch
// count, so the curve visibly grows toward the right edge as training runs.
// min/max are in domain space (see chartDomain).
export function polylinePoints(
  values: number[],
  plannedEpochs: number,
  min: number,
  max: number,
  width: number,
  height: number,
  scale: ChartScale = 'linear'
): string {
  const slots = Math.max(plannedEpochs, values.length, 2) - 1
  return values
    .map((v, i) => {
      const x = (i / slots) * width
      const t = yDomainValue(v, min, max, scale)
      const y = height - ((t - min) / (max - min)) * height
      return `${x.toFixed(2)},${y.toFixed(2)}`
    })
    .join(' ')
}

// Evenly spaced y-axis tick values across a (padded) domain, min → max.
export function linearTicks(min: number, max: number, count = 4): number[] {
  const step = (max - min) / (count - 1)
  return Array.from({ length: count }, (_, i) => min + i * step)
}

// y-axis ticks for either scale: `value` positions the gridline in domain
// space, `label` shows the real quantity (a log tick's label is 10^value).
// Log labels outside [0.01, 10000) go exponential — "1.3e-4" fits the tick
// gutter where "0.000133" clips.
export function chartTicks(
  min: number,
  max: number,
  scale: ChartScale = 'linear'
): { value: number; label: string }[] {
  return linearTicks(min, max).map((t) => {
    if (scale !== 'log') return { value: t, label: tickLabel(t) }
    const v = 10 ** t
    return { value: t, label: v >= 0.01 && v < 10000 ? tickLabel(v) : v.toExponential(1) }
  })
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

// Compact tick label: ~3 significant digits, no trailing zero noise. Tiny
// magnitudes go exponential (6.3e-4) so labels never outgrow — and clip in —
// the chart's fixed left gutter.
export function tickLabel(v: number): string {
  if (v === 0) return '0'
  if (Math.abs(v) < 0.01) return v.toExponential(1)
  return String(parseFloat(v.toPrecision(3)))
}

// X pixel for an epoch number, matching polylinePoints' slot math.
export function epochX(epoch: number, planned: number, width: number): number {
  const slots = Math.max(planned, 2) - 1
  return ((epoch - 1) / slots) * width
}

// ---- the merged loss chart: step + epoch series on one epoch axis ----------

export interface XYPoint {
  x: number // epoch-axis position (fractional for steps, integer for epochs)
  y: number
}

export interface XYSeries {
  key: string
  label: string
  points: XYPoint[]
  // The step-resolution texture layer: drawn faint under its epoch-mean line
  // and kept out of the legend (same quantity — the mean carries the reading).
  raw?: boolean
}

// A streamed step point as the merged chart consumes it: metrics plus the
// epoch-axis position the backend baked in at emit time (a resumed segment's
// points sit past its offset; absent = unknown loader length, no mapping).
export interface MergedStepPoint {
  epoch_x?: number
  metrics: Record<string, number>
}

/** The merged loss chart's series, all in epoch-axis x — the layered read:
 * per-batch loss is high-variance, so each metric's step-resolution curve is
 * a `raw` TEXTURE layer (drawn faint, no legend entry) under its epoch-mean
 * line, which carries the trend — TensorBoard's raw+smoothed idea, using the
 * mean the loop already computes instead of an EMA knob. Epoch-only series
 * (val) overlay at their TRUE epoch numbers (a resumed run's epochs don't
 * start at 1). Without positioned steps it degrades to the epoch series
 * alone — exactly the pre-merge chart. */
export function mergedLossSeries(
  epochs: { epoch: number; metrics: Record<string, number> }[],
  lossKeys: string[],
  steps: MergedStepPoint[]
): XYSeries[] {
  const label = (key: string) => (key.includes('_') ? key.slice(0, key.indexOf('_')) : key)
  const placed = steps.filter((p) => p.epoch_x !== undefined)
  const epochPoints = (k: string): XYPoint[] =>
    epochs.filter((e) => k in e.metrics).map((e) => ({ x: e.epoch, y: e.metrics[k] }))

  // Raw layers first, so the mean lines paint over them.
  const out: XYSeries[] = []
  const stepKeys: string[] = []
  for (const p of placed) for (const k of Object.keys(p.metrics)) if (!stepKeys.includes(k)) stepKeys.push(k)
  for (const k of stepKeys) {
    out.push({
      key: `${k}·steps`,
      label: label(k),
      raw: true,
      points: placed.filter((p) => k in p.metrics).map((p) => ({ x: p.epoch_x!, y: p.metrics[k] })),
    })
  }
  for (const k of lossKeys) {
    out.push({ key: k, label: label(k), points: epochPoints(k) })
  }
  return out
}

/** XY series → SVG polyline points. x is epoch-axis position over [0, xMax];
 * y goes through the scale transform (min/max in domain space, log-clamped). */
export function xyPolylinePoints(
  points: XYPoint[],
  xMax: number,
  min: number,
  max: number,
  width: number,
  height: number,
  scale: ChartScale = 'linear'
): string {
  const denom = Math.max(xMax, 1)
  return points
    .map((p) => {
      const x = (p.x / denom) * width
      const t = yDomainValue(p.y, min, max, scale)
      const y = height - ((t - min) / (max - min)) * height
      return `${x.toFixed(2)},${y.toFixed(2)}`
    })
    .join(' ')
}
