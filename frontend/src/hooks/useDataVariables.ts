import { useQuery } from '@tanstack/react-query'

// A data object registered on the session (sess.data(...)), with the Input shape
// it implies (when derivable) so the Data panel can push it into the model.
export interface DataVariable {
  name: string
  kind: string
  shape?: number[]
  dtype?: string
  batch_size?: number
  num_samples?: number
  input_shape?: { shape: string; dtype: string }
}

// Live listing — refetched on demand (a "refresh" button), since the registry
// changes as cells run sess.data(...). Only enabled for the memory source.
export function useDataVariables(enabled: boolean) {
  return useQuery<DataVariable[]>({
    queryKey: ['data-variables'],
    queryFn: async () => {
      const res = await fetch('/api/data/variables')
      if (!res.ok) throw new Error('Failed to load registered data')
      return (await res.json()).variables
    },
    enabled,
    staleTime: 0,
    refetchOnWindowFocus: false,
  })
}
