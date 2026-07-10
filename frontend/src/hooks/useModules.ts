import { useQuery } from '@tanstack/react-query'

// An nn.Module class registered on the session (sess.modules(Name=Class)) —
// the Custom node's picker entries.
export interface RegisteredModule {
  name: string
  doc: string | null
}

// Live listing — refetched on demand (the ↻ button), since the registry changes
// as cells run sess.modules(...). Same shape as useDataVariables.
export function useModules(enabled: boolean) {
  return useQuery<RegisteredModule[]>({
    queryKey: ['registered-modules'],
    queryFn: async () => {
      const res = await fetch('/api/modules')
      if (!res.ok) throw new Error('Failed to load registered modules')
      return (await res.json()).modules
    },
    enabled,
    staleTime: 0,
    refetchOnWindowFocus: false,
  })
}
