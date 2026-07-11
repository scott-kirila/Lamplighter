import { useEffect, useState } from 'react'
import { useGraphStore } from '../store/graphStore'
import { useDataVariables } from './useDataVariables'

export interface DiagnosticCheck {
  level: 'ok' | 'warn' | 'error'
  title: string
  detail: string
}

// Pre-flight data↔model checks that need the real registered tensors —
// sample-count alignment, class-range-vs-loss (the CUDA-assert catcher),
// batch-size × BatchNorm traps. Debounced POST to /api/data/diagnose, re-run on
// any change to the loop, data nodes, models, or the registry. Shared by the
// Readiness panel (which lists the checks) and the Run button (which disables on
// an error-level one) so both read the same source of truth.
export function useReadiness(): DiagnosticCheck[] {
  const toProject = useGraphStore((s) => s.toProject)
  const training = useGraphStore((s) => s.training)
  const dataNodes = useGraphStore((s) => s.dataNodes)
  const models = useGraphStore((s) => s.models)
  const nodes = useGraphStore((s) => s.nodes)
  const activeModelId = useGraphStore((s) => s.activeModelId)
  const modelGraphs = useGraphStore((s) => s.modelGraphs)
  const { data: registered } = useDataVariables(true)

  const [checks, setChecks] = useState<DiagnosticCheck[]>([])
  const diagKey = JSON.stringify([
    training,
    dataNodes.map((d) => [d.kind, d.config]),
    models.map((m) => {
      const ns = m.id === activeModelId ? nodes : modelGraphs[m.id]?.nodes ?? []
      return ns.map((n) => [n.data.nodeType, n.data.params])
    }),
    registered,
  ])
  useEffect(() => {
    let cancelled = false
    const t = window.setTimeout(async () => {
      try {
        const res = await fetch('/api/data/diagnose', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(toProject()),
        })
        if (res.ok && !cancelled) setChecks((await res.json()).checks)
      } catch {
        /* backend hiccup — keep the last checklist */
      }
    }, 350)
    return () => {
      cancelled = true
      window.clearTimeout(t)
    }
  }, [diagKey, toProject])

  return checks
}
