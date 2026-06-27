import { useQuery } from '@tanstack/react-query'
import type { NodeDef } from '../types/graph'

export function useRegistry() {
  return useQuery<Record<string, NodeDef>>({
    queryKey: ['registry'],
    queryFn: async () => {
      const res = await fetch('/api/registry')
      if (!res.ok) throw new Error('Failed to load registry')
      return res.json()
    },
    staleTime: Infinity,
  })
}
