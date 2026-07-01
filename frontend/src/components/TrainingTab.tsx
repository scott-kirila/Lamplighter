import { useEffect, useState } from 'react'
import { useGraphStore } from '../store/graphStore'
import { useTrainingParams } from '../hooks/useTrainingParams'
import { paramVisible } from '../lib/paramVisible'
import { ParamControl } from './Inspector'

export function TrainingTab() {
  const { data: params } = useTrainingParams()
  const training = useGraphStore((s) => s.training)
  const setTrainingParam = useGraphStore((s) => s.setTrainingParam)
  const nodes = useGraphStore((s) => s.nodes)
  const shapes = useGraphStore((s) => s.shapes)

  // Output-shape readout — context for choosing a loss without seeing the canvas.
  const outputNode = nodes.find((n) => n.data.nodeType === 'Output')
  const outShape = outputNode ? shapes[outputNode.id] : undefined

  // Effective config (stored over defaults) for evaluating show_if — so gated
  // params (batch_size/val_split under data=dataloader) hide correctly.
  const defaults = Object.fromEntries((params ?? []).map((p) => [p.name, p.default]))

  // Generated train() preview. Refetched (debounced) after a config change, by
  // which time the change has synced to the backend via the validation socket.
  const [code, setCode] = useState<string | null>(null)
  const trainingKey = JSON.stringify(training)
  useEffect(() => {
    let cancelled = false
    const t = window.setTimeout(async () => {
      try {
        const res = await fetch('/api/training/code')
        if (res.ok && !cancelled) setCode((await res.json()).code)
      } catch {
        /* backend hiccup — leave the last preview */
      }
    }, 400)
    return () => {
      cancelled = true
      window.clearTimeout(t)
    }
  }, [trainingKey])

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
          Training
        </div>
        <div style={{ color: 'var(--text-6)', fontSize: 11, marginBottom: 16 }}>
          model output:{' '}
          <span style={{ color: 'var(--accent)' }}>
            {outShape ? `[${outShape.join(', ')}]` : '—'}
          </span>
        </div>

        {(params ?? [])
          .filter((param) => paramVisible(param, { ...defaults, ...training }))
          .map((param) => (
          <div key={param.name} style={{ marginBottom: 14 }}>
            <label style={{ display: 'block', color: 'var(--text-5)', fontSize: 11, marginBottom: 4 }}>
              {param.label}
            </label>
            <ParamControl
              param={param}
              value={training[param.name]}
              nodeColor="var(--accent)"
              onChange={(next) => setTrainingParam(param.name, next)}
            />
          </div>
        ))}
      </div>

      {/* Generated train() preview */}
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
          Generated train()
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
