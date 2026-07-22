import { useCallback } from 'react'
import { useGraphStore } from '../store/graphStore'
import type { DomainProject, NodeDef } from '../types/graph'

/**
 * Load a built-in template as the current project.
 *
 * Shared by the Toolbar's Templates menu and the empty-canvas Start panel, so
 * "load a template" means exactly one thing: replace the project, reset the
 * history and dashboard (a template IS a new project — undoing back into the
 * previous one would be incoherent), and land on the Models overview.
 *
 * Neither caller confirms here. The Toolbar wraps it in one because it can be
 * invoked over real work; the Start panel does not, because it only appears
 * when there is none.
 */
export function useLoadTemplate(registry: Record<string, NodeDef>) {
  const loadProject = useGraphStore((s) => s.loadProject)
  const freshStart = useGraphStore((s) => s.freshStart)
  const setActiveTab = useGraphStore((s) => s.setActiveTab)

  return useCallback(
    async (name: string): Promise<boolean> => {
      try {
        const res = await fetch(`/api/templates/${encodeURIComponent(name)}`)
        if (!res.ok) return false
        const project = (await res.json()) as DomainProject
        loadProject(project, registry)
        freshStart()
        setActiveTab('overview')
        return true
      } catch {
        // backend hiccup — keep the current project
        return false
      }
    },
    [registry, loadProject, freshStart, setActiveTab]
  )
}
