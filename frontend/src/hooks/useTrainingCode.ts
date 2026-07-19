import { useEffect, useState } from 'react'
import { useGraphStore } from '../store/graphStore'

// The generated train() source for the live editor project — POSTed (not read
// from backend state) so it always reflects the canvas: the project's training
// config and the model's input count. Debounced; fetches only while `enabled`
// (the Training tab's code panel is open).
export function useTrainingCode(enabled: boolean): string | null {
  const training = useGraphStore((s) => s.training)
  const nodes = useGraphStore((s) => s.nodes)
  const toProject = useGraphStore((s) => s.toProject)

  const inputCount = nodes.filter((n) => n.data.nodeType === 'Input').length
  const configKey = JSON.stringify([training])

  const [code, setCode] = useState<string | null>(null)
  useEffect(() => {
    if (!enabled) return
    let cancelled = false
    const t = window.setTimeout(async () => {
      try {
        const res = await fetch('/api/training/code', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(toProject()),
        })
        if (res.ok && !cancelled) setCode((await res.json()).code)
      } catch {
        /* backend hiccup — leave the last preview */
      }
    }, 300)
    return () => {
      cancelled = true
      window.clearTimeout(t)
    }
  }, [enabled, configKey, inputCount, toProject])

  return code
}
