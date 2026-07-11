import { useQuery } from '@tanstack/react-query'

// A built-in starting point for the New-project flow (metadata only; the full
// project is fetched when picked). Held green by the backend test suite.
export interface TemplateMeta {
  name: string
  label: string
  description: string
}

export function useTemplates(enabled: boolean) {
  return useQuery<TemplateMeta[]>({
    queryKey: ['templates'],
    queryFn: async () => {
      const res = await fetch('/api/templates')
      if (!res.ok) throw new Error('Failed to load templates')
      return (await res.json()).templates
    },
    enabled,
    staleTime: Infinity, // built-in — changes only with the backend
  })
}
