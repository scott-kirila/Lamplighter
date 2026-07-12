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
  w: number
  dw?: number
  g?: number
}
// role → layer_N → stat, for one epoch.
export type HealthSnapshot = Record<string, Record<string, HealthStat>>

// One epoch of a streamed in-kernel training run.
export interface RunEpoch {
  epoch: number
  epochs: number
  metrics: Record<string, number>
  health?: HealthSnapshot
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

interface RunStore {
  runState: RunState
  runEpochs: RunEpoch[]
  runError: string | null
  runSeed: number | null
  runBestEpoch: number | null

  setRunStatus: (
    state: RunState,
    error: string | null,
    seed?: number | null,
    bestEpoch?: number | null
  ) => void
  appendRunEpoch: (epoch: RunEpoch) => void
  // Seed run state from GET /api/run/status on (re)connect, so a tab that joins
  // mid-run (or after) shows the run instead of waiting for the next WS event.
  hydrateRun: (
    state: RunState,
    error: string | null,
    epochs: RunEpoch[],
    seed?: number | null,
    bestEpoch?: number | null
  ) => void
  // Replace run state wholesale — used when restoring a checkpoint, whose
  // status must overwrite the currently shown run.
  replaceRun: (
    state: RunState,
    error: string | null,
    epochs: RunEpoch[],
    seed?: number | null,
    bestEpoch?: number | null
  ) => void
  // Back to idle with no curves — a "new project" (blank or template) discards
  // the run belonging to the project it replaces.
  reset: () => void
}

export const useRunStore = create<RunStore>((set) => ({
  runState: 'idle',
  runEpochs: [],
  runError: null,
  runSeed: null,
  runBestEpoch: null,

  // Entering "running" clears the previous run's lines so the panel starts fresh.
  setRunStatus: (state, error, seed, bestEpoch) =>
    set((s) => ({
      runState: state,
      runError: error,
      runSeed: seed !== undefined ? seed : s.runSeed,
      runBestEpoch: bestEpoch !== undefined ? bestEpoch : s.runBestEpoch,
      runEpochs: state === 'running' && s.runState !== 'running' ? [] : s.runEpochs,
    })),

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
  hydrateRun: (state, error, epochs, seed = null, bestEpoch = null) =>
    set((s) => ({
      runState: s.runState === 'idle' ? state : s.runState,
      runError: s.runError ?? error,
      runSeed: s.runSeed ?? seed,
      runBestEpoch: s.runBestEpoch ?? bestEpoch,
      runEpochs: epochs.length > s.runEpochs.length ? epochs : s.runEpochs,
    })),

  // Wholesale replacement from a restored checkpoint's status — unlike
  // hydrateRun's merge, a restore must overwrite whatever run was showing.
  replaceRun: (state, error, epochs, seed = null, bestEpoch = null) =>
    set({
      runState: state,
      runError: error,
      runSeed: seed,
      runBestEpoch: bestEpoch,
      runEpochs: epochs,
    }),

  reset: () =>
    set({ runState: 'idle', runEpochs: [], runError: null, runSeed: null, runBestEpoch: null }),
}))
