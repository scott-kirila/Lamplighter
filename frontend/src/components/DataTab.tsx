import { useEffect, useState } from 'react'
import { useGraphStore } from '../store/graphStore'
import { useDataParams } from '../hooks/useDataParams'
import { ParamControl } from './Inspector'
import type { ParamDef } from '../types/graph'

// A param is shown unless its show_if names other params that don't all match the
// effective config. `effective` = stored values over the param defaults, so e.g.
// `dataset` appears as soon as source=torchvision even before `source` is touched.
export function paramVisible(param: ParamDef, effective: Record<string, unknown>): boolean {
  if (!param.show_if) return true
  return Object.entries(param.show_if).every(([k, v]) => effective[k] === v)
}

export function DataTab() {
  const { data: params } = useDataParams()
  const config = useGraphStore((s) => s.data)
  const setDataParam = useGraphStore((s) => s.setDataParam)

  // Generated make_dataloaders() preview. Refetched (debounced) after a config
  // change, by which time it has synced to the backend via the validation socket.
  const [code, setCode] = useState<string | null>(null)
  const configKey = JSON.stringify(config)
  useEffect(() => {
    let cancelled = false
    const t = window.setTimeout(async () => {
      try {
        const res = await fetch('/api/data/code')
        if (res.ok && !cancelled) setCode((await res.json()).code)
      } catch {
        /* backend hiccup — leave the last preview */
      }
    }, 400)
    return () => {
      cancelled = true
      window.clearTimeout(t)
    }
  }, [configKey])

  // Effective config (stored value or the param default), used to evaluate
  // show_if — so a field like `dataset` appears once source=torchvision even
  // before the user has touched `source`.
  const defaults = Object.fromEntries((params ?? []).map((p) => [p.name, p.default]))
  const effective: Record<string, unknown> = { ...defaults, ...config }

  return (
    <div style={{ display: 'flex', flex: 1, overflow: 'hidden' }}>
      {/* Config form */}
      <div
        style={{
          width: 320,
          background: 'var(--panel)',
          borderRight: '1px solid var(--border)',
          padding: 20,
          overflowY: 'auto',
          fontFamily: 'monospace',
          flexShrink: 0,
        }}
      >
        <div style={{ color: 'var(--text)', fontWeight: 700, fontSize: 14, marginBottom: 4 }}>
          Data
        </div>
        <div style={{ color: 'var(--text-6)', fontSize: 11, marginBottom: 16 }}>
          builds <span style={{ color: 'var(--accent)' }}>make_dataloaders()</span>
        </div>

        {(params ?? []).filter((p) => paramVisible(p, effective)).map((param) => (
          <div key={param.name} style={{ marginBottom: 14 }}>
            <label style={{ display: 'block', color: 'var(--text-5)', fontSize: 11, marginBottom: 4 }}>
              {param.label}
            </label>
            <ParamControl
              param={param}
              value={config[param.name]}
              nodeColor="var(--accent)"
              onChange={(next) => setDataParam(param.name, next)}
            />
          </div>
        ))}
      </div>

      {/* Generated make_dataloaders() preview */}
      <div style={{ flex: 1, display: 'flex', flexDirection: 'column', background: 'var(--bg)' }}>
        <div
          style={{
            height: 32,
            background: 'var(--panel)',
            borderBottom: '1px solid var(--border)',
            display: 'flex',
            alignItems: 'center',
            padding: '0 16px',
            fontFamily: 'monospace',
            fontSize: 11,
            color: 'var(--text-4)',
            textTransform: 'uppercase',
            letterSpacing: 1,
            flexShrink: 0,
          }}
        >
          Generated make_dataloaders()
        </div>
        <pre
          style={{
            margin: 0,
            padding: '16px 20px',
            overflow: 'auto',
            flex: 1,
            fontFamily: 'monospace',
            fontSize: 12,
            lineHeight: 1.5,
            color: 'var(--text)',
            whiteSpace: 'pre',
          }}
        >
          {code ?? ''}
        </pre>
      </div>
    </div>
  )
}
