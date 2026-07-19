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
