import { useMemo } from 'react'
import { useRunStore, type HealthSnapshot, type HealthStat, type RunEpoch } from '../store/runStore'

// The verdict for one layer, derived from its update-ratio series (the backend
// streams raw norms; the thresholds and the "sustained" logic live here so they
// stay tunable in one place — the same split metrics use).
export interface Verdict {
  level: 'ok' | 'warn' | 'error'
  label: string
  note: string
}

export interface LayerHealth {
  layer: string // "layer_0"
  node: string // canvas-node label
  nodeId: string | null // canvas-node id (for badges)
  w: number[] // weight-norm series
  dw: number[] // update-ratio series (starts at epoch 2)
  g: number[] // grad-norm series (best-effort; may be empty)
  verdict: Verdict
}

export interface RoleHealth {
  role: string
  layers: LayerHealth[]
}

// Verdict thresholds on the update ratio ‖Δw‖/‖w‖ over the last few epochs.
// Two failure modes matter: ABSOLUTE (a layer essentially frozen, or updates
// larger than the weights) and — the key one for vanishing gradients — RELATIVE
// (a layer learning far slower than the rest of the model). Vanishing is a
// spread across layers, invisible to any single absolute band, so `refDw` (the
// model's typical update ratio) drives the "lagging" verdict.
const RECENT = 3
const FROZEN = 1e-6 // absolute: essentially not moving, regardless of peers
const EXPLODING = 1 // updates larger than the weights themselves
const LAG_FACTOR = 100 // >~2 orders below the best-learning layer → lagging

const mean = (xs: number[]) => xs.reduce((a, b) => a + b, 0) / xs.length

export function layerVerdict(s: { w: number[]; dw: number[]; g: number[] }, refDw?: number): Verdict {
  const wLast = s.w[s.w.length - 1]
  if (wLast !== undefined && !Number.isFinite(wLast)) {
    return { level: 'error', label: 'diverged', note: 'weights are NaN/Inf' }
  }
  const recent = s.dw.filter(Number.isFinite).slice(-RECENT)
  if (recent.length === 0) return { level: 'ok', label: '—', note: '' } // only epoch 1 so far
  const avg = mean(recent)
  const gLast = s.g[s.g.length - 1]
  const gradNote = gLast !== undefined && gLast < 1e-6 ? 'gradients ≈ 0 (vanishing)' : ''

  if (avg > EXPLODING) {
    return { level: 'error', label: 'exploding', note: 'updates larger than the weights' }
  }
  if (avg < FROZEN) {
    return { level: 'warn', label: 'stalled', note: gradNote || 'weights barely changing' }
  }
  // Relative: far below the rest of the model → likely a vanishing-gradient layer.
  if (refDw !== undefined && refDw > 0 && avg < refDw / LAG_FACTOR) {
    const slower = Math.round(refDw / avg)
    return { level: 'warn', label: 'lagging', note: gradNote || `learning ~${slower}× slower than the model's typical layer` }
  }
  return { level: 'ok', label: 'healthy', note: '' }
}

// The reference the relative "lagging" check compares to: the best-learning
// (fastest) layer's update ratio, excluding exploding layers so a diverging one
// can't become the reference. Vanishing gradients are a smooth top-to-bottom
// decay, so the *median* would sit mid-slope and hide it — "far below the layer
// that IS learning" is the signal, hence the max.
function referenceDw(perLayerDw: number[]): number | undefined {
  const normal = perLayerDw.filter((x) => Number.isFinite(x) && x <= EXPLODING)
  return normal.length ? Math.max(...normal) : undefined
}

// Pivot the streamed per-epoch snapshots into per-role, per-layer series (in the
// layer order of the latest snapshot) and attach each layer's verdict. Pure, so
// the panel is a dumb renderer and this is what the tests exercise.
export function buildHealth(epochs: RunEpoch[]): RoleHealth[] {
  const withHealth = epochs.filter((e): e is RunEpoch & { health: HealthSnapshot } => !!e.health)
  if (withHealth.length === 0) return []
  const latest = withHealth[withHealth.length - 1].health
  return Object.entries(latest).map(([role, layers]) => {
    // First pass: assemble each layer's series (no verdict yet).
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
      }
    })
    // The reference drives the relative "lagging" verdict, so it needs every
    // layer's recent value first. Use the same recent-average the verdict does,
    // so the layer and the reference are measured the same way.
    const refDw = referenceDw(
      series.map((s) => {
        const r = s.dw.filter(Number.isFinite).slice(-RECENT)
        return r.length ? mean(r) : NaN
      })
    )
    return { role, layers: series.map((s) => ({ ...s, verdict: layerVerdict(s, refDw) })) }
  })
}

// Flatten the per-role/layer verdicts to one verdict per canvas node (the most
// severe, if a node somehow recurs), for decorating the model canvas.
const SEVERITY: Record<Verdict['level'], number> = { ok: 0, warn: 1, error: 2 }
export function nodeVerdicts(roles: RoleHealth[]): Record<string, Verdict> {
  const out: Record<string, Verdict> = {}
  for (const r of roles) {
    for (const l of r.layers) {
      if (!l.nodeId) continue
      const cur = out[l.nodeId]
      if (!cur || SEVERITY[l.verdict.level] > SEVERITY[cur.level]) out[l.nodeId] = l.verdict
    }
  }
  return out
}

// The current run's health verdict for one canvas node (undefined if none).
export function useNodeVerdict(nodeId: string): Verdict | undefined {
  const roles = useTrainingHealth()
  return useMemo(() => nodeVerdicts(roles)[nodeId], [roles, nodeId])
}

// Per-role, per-layer training health for the current run's streamed snapshots.
export function useTrainingHealth(): RoleHealth[] {
  const runEpochs = useRunStore((s) => s.runEpochs)
  return useMemo(() => buildHealth(runEpochs), [runEpochs])
}
