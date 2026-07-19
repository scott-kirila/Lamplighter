import { useMutation } from '@tanstack/react-query'
import { apiFetch } from '../lib/api'
import type { DomainProject } from '../types/graph'

// Start/stop the in-kernel training run. Same WS-freshness contract as
// useCheckpointActions: run state streams back over the WebSocket
// (run_status / run_epoch, hydrated into the run store in useValidation), so
// these never touch the query cache. `start` posts the whole project (multi-model
// recipes send every model); the caller flips the run store to 'failed' with
// `err.message` when the start throws.
export function useRunControls() {
  const start = useMutation({
    mutationFn: (project: DomainProject) =>
      apiFetch('/api/run/start', { body: project, fallback: 'could not start the run' }),
  })
  const stop = useMutation({
    mutationFn: () => apiFetch('/api/run/stop'),
  })
  return { start, stop }
}
