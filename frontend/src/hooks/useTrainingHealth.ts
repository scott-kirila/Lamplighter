import { useMemo } from 'react'
import { useRunStore, type HealthSnapshot, type HealthStat, type RunEpoch } from '../store/runStore'

export interface LayerHealth {
  layer: string // "layer_0"
  node: string // canvas-node label
  nodeId: string | null // canvas-node id (for badges)
  w: number[] // weight-norm series (parametric layers)
  dw: number[] // update-ratio series (parametric; starts at epoch 2)
  g: number[] // grad-norm series (best-effort; may be empty)
  dead: number[] // dead-unit fraction series (activation layers; else empty)
  // 0 (green — seems fine) … 1 (red — has the indications of a problem); null
  // when there isn't enough signal yet. Deliberately a continuous score, not a
  // labelled verdict: the color evokes a reading, the tool never asserts one.
  concern: number | null
  note: string // factual hover context (raw numbers, not a judgement)
}

export interface RoleHealth {
  role: string
  layers: LayerHealth[]
}

const RECENT = 3
const EXPLODING = 1 // updates ≥ the weights themselves — excluded from the reference

const mean = (xs: number[]) => xs.reduce((a, b) => a + b, 0) / xs.length
const clamp01 = (x: number) => Math.max(0, Math.min(1, x))

// A continuous concern score for one layer's update ratio ‖Δw‖/‖w‖: 0 (green) to
// 1 (red). Two axes, whichever is worse — the layer moving *too much* (updates
// approaching/exceeding the weights) or *far too little relative to the fastest
// layer* (the vanishing-gradient signal, which is a spread, so it's measured
// against `refDw`). No thresholds snap a verdict; the score is smooth in log
// space, so borderline layers land in the amber middle. `note` is factual.
export function concernScore(
  s: { w: number[]; dw: number[]; g: number[]; dead: number[] },
  refDw?: number
): { concern: number | null; note: string } {
  const wLast = s.w[s.w.length - 1]
  if (wLast !== undefined && !Number.isFinite(wLast)) return { concern: 1, note: 'weights are NaN/Inf' }

  // Activation layer: no update ratio — score by dead-unit fraction instead.
  // A few % is normal; 50%+ dead → red.
  const deadRecent = s.dead.filter(Number.isFinite).slice(-RECENT)
  if (deadRecent.length) {
    const d = mean(deadRecent)
    return { concern: clamp01(d / 0.5), note: `${Math.round(d * 100)}% dead units` }
  }

  const recent = s.dw.filter(Number.isFinite).slice(-RECENT)
  if (recent.length === 0) return { concern: null, note: 'no update ratio yet' }
  const avg = mean(recent)

  // Too much: 0.1/epoch → 0, ≥ 1/epoch (updates = weights) → 1.
  const explode = clamp01(Math.log10(avg) + 1)
  // Too little (relative): within 1 order of the fastest layer → 0, ≥ 2 orders
  // below → 1.
  let lag = 0
  let note = `Δw/w = ${avg.toExponential(1)}`
  if (refDw !== undefined && refDw > 0) {
    const ratio = avg / refDw
    lag = clamp01(-1 - Math.log10(ratio))
    if (ratio < 0.5) note += ` · ${Math.round(1 / ratio)}× below the fastest layer`
  }
  const gLast = s.g[s.g.length - 1]
  if (gLast !== undefined && gLast < 1e-6) note += ' · gradients ≈ 0'
  return { concern: Math.max(explode, lag), note }
}

// Green (0) → yellow (0.5) → red (1). Theme-independent by design — a health
// scale, not the app accent.
export function concernColor(concern: number): string {
  return `hsl(${(1 - clamp01(concern)) * 120}, 70%, 45%)`
}

// The reference the relative axis compares to: the fastest (best-learning) layer,
// excluding exploding layers so a diverging one can't become the reference.
// Vanishing is a smooth decay, so the median would sit mid-slope and hide it —
// "far below the layer that IS learning" is the signal, hence the max.
function referenceDw(perLayerDw: number[]): number | undefined {
  const normal = perLayerDw.filter((x) => Number.isFinite(x) && x <= EXPLODING)
  return normal.length ? Math.max(...normal) : undefined
}

// Pivot the streamed snapshots into per-role, per-layer series (in the latest
// snapshot's layer order) and attach each layer's concern score. Pure.
export function buildHealth(epochs: RunEpoch[]): RoleHealth[] {
  const withHealth = epochs.filter((e): e is RunEpoch & { health: HealthSnapshot } => !!e.health)
  if (withHealth.length === 0) return []
  const latest = withHealth[withHealth.length - 1].health
  return Object.entries(latest).map(([role, layers]) => {
    const series = Object.entries(layers).map(([layer, stat]) => {
      const pick = (get: (st: HealthStat) => number | undefined): number[] =>
        withHealth
          .map((e) => e.health[role]?.[layer])
          .map((st) => (st ? get(st) : undefined))
          .filter((x): x is number => x !== undefined)
      return {
        layer,
        node: stat.node,
        nodeId: stat.nodeId ?? null,
        w: pick((st) => st.w),
        dw: pick((st) => st.dw),
        g: pick((st) => st.g),
        dead: pick((st) => st.dead),
      }
    })
    // The reference needs every layer's recent value first — measured the same
    // way (recent-average) the score measures each layer.
    const refDw = referenceDw(
      series.map((s) => {
        const r = s.dw.filter(Number.isFinite).slice(-RECENT)
        return r.length ? mean(r) : NaN
      })
    )
    return { role, layers: series.map((s) => ({ ...s, ...concernScore(s, refDw) })) }
  })
}

// Worst concern (+ its note) per canvas node, for decorating the model canvas.
export function nodeHealth(roles: RoleHealth[]): Record<string, { concern: number; note: string }> {
  const out: Record<string, { concern: number; note: string }> = {}
  for (const r of roles) {
    for (const l of r.layers) {
      if (!l.nodeId || l.concern == null) continue
      if (!(l.nodeId in out) || l.concern > out[l.nodeId].concern) {
        out[l.nodeId] = { concern: l.concern, note: l.note }
      }
    }
  }
  return out
}

// Per-role, per-layer training health for the current run.
export function useTrainingHealth(): RoleHealth[] {
  const runEpochs = useRunStore((s) => s.runEpochs)
  return useMemo(() => buildHealth(runEpochs), [runEpochs])
}

// The current run's worst concern (+ note) for one canvas node (undefined if none).
export function useNodeHealth(nodeId: string): { concern: number; note: string } | undefined {
  const roles = useTrainingHealth()
  return useMemo(() => nodeHealth(roles)[nodeId], [roles, nodeId])
}
