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
// Healthy learning sits around ~1e-3; ~0 means the layer isn't moving; >1 means
// updates are larger than the weights themselves (diverging).
const RECENT = 3
const STALLED = 1e-5
const EXPLODING = 1

export function layerVerdict(s: { w: number[]; dw: number[]; g: number[] }): Verdict {
  const wLast = s.w[s.w.length - 1]
  if (wLast !== undefined && !Number.isFinite(wLast)) {
    return { level: 'error', label: 'diverged', note: 'weights are NaN/Inf' }
  }
  const recent = s.dw.filter(Number.isFinite).slice(-RECENT)
  if (recent.length === 0) return { level: 'ok', label: '—', note: '' } // only epoch 1 so far
  const avg = recent.reduce((a, b) => a + b, 0) / recent.length
  if (avg > EXPLODING) {
    return { level: 'error', label: 'exploding', note: 'updates larger than the weights' }
  }
  if (avg < STALLED) {
    const gLast = s.g[s.g.length - 1]
    const note = gLast !== undefined && gLast < 1e-6 ? 'gradients ≈ 0 (vanishing)' : 'weights barely changing'
    return { level: 'warn', label: 'stalled', note }
  }
  return { level: 'ok', label: 'healthy', note: '' }
}

// Pivot the streamed per-epoch snapshots into per-role, per-layer series (in the
// layer order of the latest snapshot) and attach each layer's verdict. Pure, so
// the panel is a dumb renderer and this is what the tests exercise.
export function buildHealth(epochs: RunEpoch[]): RoleHealth[] {
  const withHealth = epochs.filter((e): e is RunEpoch & { health: HealthSnapshot } => !!e.health)
  if (withHealth.length === 0) return []
  const latest = withHealth[withHealth.length - 1].health
  return Object.entries(latest).map(([role, layers]) => ({
    role,
    layers: Object.entries(layers).map(([layer, stat]) => {
      const pick = (get: (st: HealthStat) => number | undefined): number[] =>
        withHealth
          .map((e) => e.health[role]?.[layer])
          .map((st) => (st ? get(st) : undefined))
          .filter((x): x is number => x !== undefined)
      const w = pick((st) => st.w)
      const dw = pick((st) => st.dw)
      const g = pick((st) => st.g)
      return { layer, node: stat.node, w, dw, g, verdict: layerVerdict({ w, dw, g }) }
    }),
  }))
}

// Per-role, per-layer training health for the current run's streamed snapshots.
export function useTrainingHealth(): RoleHealth[] {
  const runEpochs = useRunStore((s) => s.runEpochs)
  return useMemo(() => buildHealth(runEpochs), [runEpochs])
}
