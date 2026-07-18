import { create } from 'zustand'

// The live training-run dashboard: the state streamed from an in-kernel run
// (over the WebSocket) and its per-epoch curves. Kept SEPARATE from the project
// store on purpose — the run is an ephemeral mirror of what the kernel is doing,
// with a different lifecycle from the design you edit: it isn't autosaved, isn't
// part of undo, and a "new project" discards it. Those boundaries were
// hand-maintained while this lived in graphStore; here they're structural (the
// design store simply can't reach these fields).

export type RunState = 'idle' | 'running' | 'done' | 'stopped' | 'failed'

// One layer's per-epoch training-health stats (see the runner's _collect_health):
// weight L2 norm, update ratio ‖Δw‖/‖w‖ (absent on epoch 1), grad norm (best-
// effort), and the canvas-node label the layer maps to.
export interface HealthStat {
  node: string // canvas-node label
  nodeId?: string | null // canvas-node id (for badges)
  // Parametric layers carry weight/update/grad norms; activation layers (no
  // params) instead carry `dead` — the fraction of units that stayed ~0 all epoch.
  w?: number
  dw?: number
  g?: number
  dead?: number
}
// role → layer_N → stat, for one epoch.
export type HealthSnapshot = Record<string, Record<string, HealthStat>>

// One epoch of a streamed in-kernel training run.
export interface RunEpoch {
  epoch: number
  epochs: number
  metrics: Record<string, number>
  health?: HealthSnapshot
  // Wall-clock seconds this epoch took (live runs only; absent for epochs
  // rebuilt from history on a mid-run reconnect, which carries no timing).
  secs?: number
}

// Rebuild the per-epoch stream from a run's history dict (metric name → series),
// for tabs that join mid-run or after it — GET /api/run/status returns the full
// history, and the dashboard renders RunEpoch[]. A metric appears in an epoch's
// metrics only when its series reaches that epoch (e.g. no val without a
// val_loader).
export function epochsFromHistory(
  history: Record<string, number[]> | null | undefined,
  plannedEpochs: number,
  healthHistory?: HealthSnapshot[] | null
): RunEpoch[] {
  if (!history) return []
  const n = Math.max(0, ...Object.values(history).map((v) => v.length))
  return Array.from({ length: n }, (_, i) => ({
    epoch: i + 1,
    epochs: plannedEpochs,
    metrics: Object.fromEntries(
      Object.entries(history)
        .filter(([, v]) => i < v.length)
        .map(([k, v]) => [k, v[i]])
    ),
    health: healthHistory?.[i],
  }))
}

// The config the shown run actually used, from its snapshot (see the chips).
export interface RunConfig {
  recipe?: string
  epochs?: number
  device?: string
  lr?: number
  lrs?: Record<string, number>
}

// One streamed per-batch point (throttled server-side): the step index and that
// batch's metrics — a single train_loss for supervised, or a GAN's g/d and a
// VAE's recon/kl. Rebuilt on reconnect from the backend's mirror buffer.
export interface StepPoint {
  step: number
  // The point's epoch-axis position, baked server-side at birth (a resumed
  // segment's points sit past its offset). Absent = unknown loader length.
  epoch_x?: number
  metrics: Record<string, number>
}

// Cap the step buffer so a marathon run can't grow it unbounded — ~3 points
// per pixel of chart width; past it the buffer halves density (see below).
const STEP_LIMIT = 4000

interface RunStore {
  runState: RunState
  runEpochs: RunEpoch[]
  stepMetrics: StepPoint[]
  // Total steps the current run will take (0 = unknown), so the step chart fixes
  // its x-axis instead of rescaling as points stream in.
  stepTotal: number
  runError: string | null
  runSeed: number | null
  runBestEpoch: null | number
  // The config the shown run actually used (from its snapshot) — the dashboard
  // labels results with it, since the form edits the NEXT run and can drift.
  runConfig: RunConfig | null
  // The shown run's history name (reserved at start for live runs; the stored
  // name when viewing) — the Runs list highlights this row.
  runName: string | null
  // The KERNEL's run name — set only by status events/hydration, never by
  // viewing. "Keep weights" must target this run: the kernel holds ITS
  // weights, whatever the dashboard is currently showing.
  kernelRunName: string | null

  setRunStatus: (
    state: RunState,
    error: string | null,
    seed?: number | null,
    bestEpoch?: number | null,
    config?: RunConfig | null,
    runName?: string | null
  ) => void
  appendRunEpoch: (epoch: RunEpoch) => void
  appendRunStep: (step: number, metrics: Record<string, number>, total: number, epochX?: number | null) => void
  // Seed run state from GET /api/run/status on (re)connect, so a tab that joins
  // mid-run (or after) shows the run instead of waiting for the next WS event.
  hydrateRun: (
    state: RunState,
    error: string | null,
    epochs: RunEpoch[],
    seed?: number | null,
    bestEpoch?: number | null,
    steps?: StepPoint[],
    stepTotal?: number,
    config?: RunConfig | null,
    runName?: string | null
  ) => void
  // Replace run state wholesale — used when restoring a checkpoint, whose
  // status must overwrite the currently shown run.
  replaceRun: (
    state: RunState,
    error: string | null,
    epochs: RunEpoch[],
    seed?: number | null,
    bestEpoch?: number | null,
    steps?: StepPoint[],
    stepTotal?: number,
    config?: RunConfig | null,
    runName?: string | null,
    kernelName?: string | null
  ) => void
  // Back to idle with no curves — a "new project" (blank or template) discards
  // the run belonging to the project it replaces.
  reset: () => void
}

export const useRunStore = create<RunStore>((set) => ({
  runState: 'idle',
  runEpochs: [],
  stepMetrics: [],
  stepTotal: 0,
  runError: null,
  runSeed: null,
  runBestEpoch: null,
  runConfig: null,
  runName: null,
  kernelRunName: null,

  // Entering "running" clears the previous run's lines so the panel starts fresh.
  setRunStatus: (state, error, seed, bestEpoch, config, runName) =>
    set((s) => {
      const fresh = state === 'running' && s.runState !== 'running'
      return {
        runState: state,
        runError: error,
        runSeed: seed !== undefined ? seed : s.runSeed,
        runBestEpoch: bestEpoch !== undefined ? bestEpoch : s.runBestEpoch,
        runConfig: config ?? (fresh ? null : s.runConfig),
        runName: runName ?? (fresh ? null : s.runName),
        kernelRunName: runName ?? (fresh ? null : s.kernelRunName),
        runEpochs: fresh ? [] : s.runEpochs,
        stepMetrics: fresh ? [] : s.stepMetrics,
        stepTotal: fresh ? 0 : s.stepTotal,
      }
    }),

  // Append a throttled step-metrics point. The buffer is bounded, but at the
  // cap it HALVES its resolution (drops every other point) instead of sliding
  // a window: the chart's x-axis spans the whole run, so a sliding window
  // made long runs "disappear" into a sliver at the right edge — thinning
  // keeps the full run's shape at ever-coarser step density instead.
  // `total` (the run's fixed step count) is constant per run, so it just overwrites.
  appendRunStep: (step, metrics, total, epochX = null) =>
    set((s) => {
      const point: StepPoint = epochX != null ? { step, epoch_x: epochX, metrics } : { step, metrics }
      const next = [...s.stepMetrics, point]
      return {
        stepMetrics: next.length > STEP_LIMIT ? next.filter((_, i) => i % 2 === 0) : next,
        stepTotal: total,
      }
    }),

  // Ignore epochs at/behind the newest one — protects against the hydration
  // fetch racing a live run_epoch event (which could otherwise duplicate a line).
  appendRunEpoch: (epoch) =>
    set((s) => {
      const last = s.runEpochs[s.runEpochs.length - 1]
      if (last && epoch.epoch <= last.epoch) return {}
      return { runEpochs: [...s.runEpochs, epoch] }
    }),

  // Conservative merge: live WS events win. State applies only when this tab
  // hasn't seen a transition yet (a late joiner misses the "running" broadcast);
  // the fetched epoch list applies only when it's more complete than ours.
  hydrateRun: (state, error, epochs, seed = null, bestEpoch = null, steps = [], stepTotal = 0, config = null, runName = null) =>
    set((s) => ({
      runState: s.runState === 'idle' ? state : s.runState,
      runError: s.runError ?? error,
      runSeed: s.runSeed ?? seed,
      runBestEpoch: s.runBestEpoch ?? bestEpoch,
      runConfig: s.runConfig ?? config,
      runName: s.runName ?? runName,
      kernelRunName: s.kernelRunName ?? runName,
      runEpochs: epochs.length > s.runEpochs.length ? epochs : s.runEpochs,
      // The step chart was live-only and vanished on refresh — seed it from the
      // backend's buffer, but never clobber points this tab already streamed.
      stepMetrics: s.stepMetrics.length === 0 && steps.length > 0 ? steps : s.stepMetrics,
      stepTotal: s.stepTotal || stepTotal,
    })),

  // Wholesale replacement from a restored checkpoint's status — unlike
  // hydrateRun's merge, a restore must overwrite whatever run was showing.
  // `kernelName`, when given, is the run the kernel now holds — restore and
  // resume change it (the live model becomes the restored/warm-started run), so
  // the "keep weights" affordance follows to the right row; a read-only view
  // omits it and leaves the kernel's run untouched.
  replaceRun: (state, error, epochs, seed = null, bestEpoch = null, steps = [], stepTotal = 0, config = null, runName = null, kernelName) =>
    set((s) => ({
      runState: state,
      runError: error,
      runSeed: seed,
      runBestEpoch: bestEpoch,
      runConfig: config,
      runName,
      kernelRunName: kernelName === undefined ? s.kernelRunName : kernelName,
      runEpochs: epochs,
      // A checkpoint carries its own step curve (and its config) — a restored
      // run shows them, not the previous run's leftovers.
      stepMetrics: steps,
      stepTotal,
    })),

  reset: () =>
    set({ runState: 'idle', runEpochs: [], stepMetrics: [], stepTotal: 0, runError: null, runSeed: null, runBestEpoch: null, runConfig: null, runName: null, kernelRunName: null }),
}))
