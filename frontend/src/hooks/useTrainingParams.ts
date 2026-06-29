import { useQuery } from '@tanstack/react-query'
import type { ParamDef } from '../types/graph'

// The training config form definition (loss/optimizer/hyperparams), served by the
// backend and rendered with the same controls as node params.
export function useTrainingParams() {
  return useQuery<ParamDef[]>({
    queryKey: ['training-params'],
    queryFn: async () => {
      const res = await fetch('/api/training/params')
      if (!res.ok) throw new Error('Failed to load training params')
      return res.json()
    },
    staleTime: Infinity,
  })
}
