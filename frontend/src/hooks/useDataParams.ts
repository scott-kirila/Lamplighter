import { useQuery } from '@tanstack/react-query'
import type { ParamDef } from '../types/graph'

// The Data panel form definition (source, batching), served by the backend and
// rendered with the same controls as node/training params. `show_if` on a param
// gates it to a source (e.g. torchvision-only fields).
export function useDataParams() {
  return useQuery<ParamDef[]>({
    queryKey: ['data-params'],
    queryFn: async () => {
      const res = await fetch('/api/data/params')
      if (!res.ok) throw new Error('Failed to load data params')
      return res.json()
    },
    staleTime: Infinity,
  })
}
