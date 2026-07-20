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
  // The role whose model receives the real data X (its Input is what the Data
  // tab picks/auto-fills): supervised "model", a GAN's "discriminator".
  data_role: string
  // "loader" (tensors/datasets) or "env" (an RL recipe — the data_role model
  // gets a Gymnasium environment node, not a dataset).
  data: string
  // The history curves a sweep may target (first = the Optimize default) —
  // only what this recipe's loop actually records.
  metrics: string[]
}

// The progress-unit word for a recipe's runs: an env (RL) recipe counts
// iterations (a batch of episodes + one update), everything else epochs.
export function unitWord(
  recipes: RecipeDef[] | undefined,
  recipeName: string | null | undefined
): 'epoch' | 'iter' {
  return recipes?.find((r) => r.name === recipeName)?.data === 'env' ? 'iter' : 'epoch'
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
