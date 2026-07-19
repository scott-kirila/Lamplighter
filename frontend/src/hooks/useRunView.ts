import { useCallback } from 'react'
import { epochsFromHistory, useRunStore } from '../store/runStore'

// Show a stored run on the dashboard — read-only: the kernel's live model and
// current run are untouched (restore stays the explicit weights action). Shared
// by the Runs list (clicking a row) and the Training tab's active-model follow,
// so there's a single view path. Returns an error message, or null on success.
//
// The returned function is stable (replaceRun is a stable zustand action), so it
// can sit in a useEffect dep list without re-firing every render.
export function useRunView() {
  const replaceRun = useRunStore((s) => s.replaceRun)
  return useCallback(
    async (runName: string): Promise<string | null> => {
      try {
        const res = await fetch(`/api/checkpoints/${encodeURIComponent(runName)}/view`)
        const status = await res.json().catch(() => ({}))
        if (!res.ok) return status.detail ?? 'could not load the run'
        replaceRun(
          status.state,
          status.error ?? null,
          epochsFromHistory(status.history, status.epochs ?? 0, status.health_history),
          status.seed ?? null,
          status.best_epoch ?? null,
          status.steps ?? [],
          status.step_total ?? 0,
          status.config ?? null,
          runName
          // kernelName omitted → the kernel's run is left untouched (read-only view).
        )
        return null
      } catch {
        return 'backend unreachable'
      }
    },
    [replaceRun]
  )
}
