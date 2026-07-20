import { create } from 'zustand'

// The live sweep (Optimize view) mirror — the SweepManager's status, streamed
// over the WS as `sweep_status` events and hydrated from GET /api/sweep/status
// on (re)connect. Kept separate from the run store on purpose, same reasoning:
// an ephemeral mirror of kernel state, not part of the design/undo/autosave.

export type SweepState = 'idle' | 'running' | 'done' | 'stopped' | 'failed'

export interface SweepBest {
  run_name: string | null
  value: number
  params: Record<string, unknown>
}

export interface SweepStatus {
  state: SweepState
  error: string | null
  study: string | null
  n_trials: number
  trial: number | null // 1-based index of the running trial
  completed: number
  pruned: number
  failed: number
  metric: string
  direction: string
  best: SweepBest | null
  // Which params moved the metric (computed once at sweep end); null until
  // then / when incomputable (< 2 completed trials, degenerate study).
  importance: Record<string, number> | null
}

const IDLE: SweepStatus = {
  state: 'idle', error: null, study: null, n_trials: 0, trial: null,
  completed: 0, pruned: 0, failed: 0, metric: 'val_loss', direction: 'minimize', best: null,
  importance: null,
}

interface SweepStore extends SweepStatus {
  // WS events apply wholesale — the backend always sends the full shape.
  setSweepStatus: (status: SweepStatus) => void
  // (Re)connect hydration: seed only an idle store, so a racing live event
  // is never overwritten by a stale fetch (the run store's contract).
  hydrateSweep: (status: SweepStatus) => void
}

export const useSweepStore = create<SweepStore>((set, get) => ({
  ...IDLE,
  setSweepStatus: (status) => set(status),
  hydrateSweep: (status) => {
    if (get().state === 'idle') set(status)
  },
}))
