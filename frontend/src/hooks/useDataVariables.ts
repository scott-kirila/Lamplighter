import { useQuery } from '@tanstack/react-query'

// A data-like object detected live in the notebook kernel, with the Input shape
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

// Live listing — refetched on demand (a "refresh" button), since the set of
// notebook variables changes as cells run. Only enabled for the variable source.
export function useDataVariables(enabled: boolean) {
  return useQuery<DataVariable[]>({
    queryKey: ['data-variables'],
    queryFn: async () => {
      const res = await fetch('/api/data/variables')
      if (!res.ok) throw new Error('Failed to load notebook variables')
      return (await res.json()).variables
    },
    enabled,
    staleTime: 0,
    refetchOnWindowFocus: false,
  })
}
