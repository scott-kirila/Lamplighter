import { useQuery } from '@tanstack/react-query'

// A stored checkpoint's listing entry (backend/checkpoints.py metas()).
export interface CheckpointMeta {
  name: string
  created: string
  epoch: number | null
  // The run's planned total — epoch < epochs marks an interrupted run,
  // which resume finishes by default.
  epochs: number | null
  best_epoch: number | null
  seed: number | null
  val_loss: number | null
  // Run-store fields — every entry the store emits carries them (state is null
  // only for a checkpoint kept before its run reached a terminal state).
  state: string | null
  source: string
  has_weights: boolean
  auto: boolean
  // Which model(s) this run trained — role → {id, name} (name frozen at run
  // time). Drives the Runs list's per-model scoping/labeling. Absent on runs
  // recorded before attribution shipped (treated as unattributed).
  models?: { id: string; name: string; role: string }[]
  // The run's recipe name (labels RL progress in iterations); absent on old sidecars.
  recipe?: string | null
  // The sweep study this run belongs to (an Optimize trial), or null — the
  // Optimize view's trials table filters the listing on this.
  study?: string | null
}

// Does a run belong to a model, for the Runs list's per-model scoping? True for
// attributed runs that name the model, and for UNattributed runs — those
// predate attribution (no `models` key) or carry an empty list (nothing to pin
// them to), and hiding history is worse than over-showing it. The single
// predicate shared by the list filter and the dashboard's follow effect.
export function belongsToModel(c: CheckpointMeta, modelId: string): boolean {
  return !c.models?.length || c.models.some((m) => m.id === modelId)
}

// Is this run a sweep trial the Runs list should tuck away? Auto trial records
// live in the Optimize view's trials table — their natural home — while the
// crowned <study>-best (named + weighted → auto=false) and any trial the user
// renames surface here like any kept run: naming already means keep-intent.
export function isSweepTrial(c: CheckpointMeta): boolean {
  return c.study != null && c.auto
}

// The session's checkpoint store listing — fetched on mount, then kept live by
// the WS "checkpoints" push (same setQueryData pattern as the data registry).
export function useCheckpoints() {
  return useQuery<CheckpointMeta[]>({
    queryKey: ['checkpoints'],
    queryFn: async () => {
      const res = await fetch('/api/checkpoints')
      if (!res.ok) throw new Error('Failed to load checkpoints')
      return (await res.json()).checkpoints
    },
    refetchOnWindowFocus: false,
  })
}
