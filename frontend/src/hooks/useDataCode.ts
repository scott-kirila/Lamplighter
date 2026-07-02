import { useEffect, useState } from 'react'
import { useGraphStore } from '../store/graphStore'

// The generated make_dataloaders() source for the live editor graph — POSTed so
// it always reflects the canvas (data config + the model's input count, one X
// arg per Input). Debounced; fetches only while `enabled` (the Data tab's code
// panel is open).
export function useDataCode(enabled: boolean): string | null {
  const data = useGraphStore((s) => s.data)
  const nodes = useGraphStore((s) => s.nodes)
  const toDomainGraph = useGraphStore((s) => s.toDomainGraph)

  const inputCount = nodes.filter((n) => n.data.nodeType === 'Input').length
  const configKey = JSON.stringify(data)

  const [code, setCode] = useState<string | null>(null)
  useEffect(() => {
    if (!enabled) return
    let cancelled = false
    const t = window.setTimeout(async () => {
      try {
        const res = await fetch('/api/data/code', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(toDomainGraph()),
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
  }, [enabled, configKey, inputCount, toDomainGraph])

  return code
}
