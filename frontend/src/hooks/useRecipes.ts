import { useQuery } from '@tanstack/react-query'
import type { ParamDef } from '../types/graph'

export interface RoleDef {
  role: string
  label: string
}

// A training recipe as the Training tab renders it: loop params, per-role
// params, role slots, and the data contract. Mirrors backend RecipeDef (minus
// the backend-only generator function).
export interface RecipeDef {
  name: string
  label: string
  roles: RoleDef[]
  params: ParamDef[]
  role_params: Record<string, ParamDef[]>
  needs_targets: boolean
  has_val: boolean
}

// The available training recipes (supervised, gan, …). The device param's
// choices are resolved live by the backend, so the form only offers what works.
export function useRecipes() {
  return useQuery<RecipeDef[]>({
    queryKey: ['recipes'],
    queryFn: async () => {
      const res = await fetch('/api/recipes')
      if (!res.ok) throw new Error('Failed to load recipes')
      return res.json()
    },
    staleTime: Infinity,
  })
}
