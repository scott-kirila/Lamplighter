import { useMutation } from '@tanstack/react-query'
import { apiFetch } from '../lib/api'
import type { DomainProject } from '../types/graph'
import type { SweepConfig } from '../lib/sweepScript'

// Start/stop the in-kernel sweep. Same WS-freshness contract as the run
// controls: sweep state streams back as `sweep_status` events into the sweep
// store, so these never touch the query cache. A 400's detail carries the
// user-facing reason (bad config, Optuna's install hint, engine busy) —
// callers show `err.message`.
export function useSweepControls() {
  const start = useMutation({
    mutationFn: (body: { project: DomainProject; config: SweepConfig }) =>
      apiFetch('/api/sweep/start', { body, fallback: 'could not start the sweep' }),
  })
  const stop = useMutation({
    mutationFn: () => apiFetch('/api/sweep/stop'),
  })
  return { start, stop }
}
