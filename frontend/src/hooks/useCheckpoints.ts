import { useQuery } from '@tanstack/react-query'

// A stored checkpoint's listing entry (backend/checkpoints.py metas()).
export interface CheckpointMeta {
  name: string
  created: string
  epoch: number | null
  best_epoch: number | null
  seed: number | null
  val_loss: number | null
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
