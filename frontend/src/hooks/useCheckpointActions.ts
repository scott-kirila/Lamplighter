import { useMutation } from '@tanstack/react-query'
import { apiFetch } from '../lib/api'

// The checkpoint-store write endpoints as one set of mutations — the single home
// for save/rename/delete (save is called from both Checkpoints and TrainingTab,
// which used to duplicate the fetch).
//
// Freshness is deliberately NOT handled here: the backend broadcasts the updated
// checkpoint list over the WebSocket (the "checkpoints" push in useValidation,
// which setQueryData's ['checkpoints']). So these must NOT invalidate/refetch
// that key — doing so would double-fetch and race the push. They exist for the
// shared endpoint home and uniform isPending/error, not cache management.
export function useCheckpointActions() {
  // "Keep weights" — upgrade a run's record with the kernel's live weights.
  const save = useMutation({
    mutationFn: (name: string) =>
      apiFetch('/api/checkpoints', { body: { name }, fallback: 'could not save the weights' }),
  })
  const rename = useMutation({
    mutationFn: ({ name, to }: { name: string; to: string }) =>
      apiFetch(`/api/checkpoints/${encodeURIComponent(name)}/rename`, {
        body: { name: to },
        fallback: 'could not rename the run',
      }),
  })
  const remove = useMutation({
    mutationFn: (name: string) =>
      apiFetch(`/api/checkpoints/${encodeURIComponent(name)}`, {
        method: 'DELETE',
        fallback: 'could not delete the checkpoint',
      }),
  })
  return { save, rename, remove }
}
