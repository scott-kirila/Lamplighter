import { useEffect, useState } from 'react'
import { useGraphStore } from '../store/graphStore'
import { useDataParams } from '../hooks/useDataParams'
import { useDataVariables, type DataVariable } from '../hooks/useDataVariables'
import { paramVisible } from '../lib/paramVisible'
import { OptionalControl, ParamControl } from './Inspector'
import type { ParamDef } from '../types/graph'

const SELECT_STYLE = {
  background: 'var(--field)',
  border: '1px solid var(--border)',
  borderRadius: 4,
  padding: '6px 8px',
  color: 'var(--text)',
  fontSize: 13,
  width: '100%',
  fontFamily: 'monospace',
  cursor: 'pointer',
} as const

// One label for a detected variable, e.g. "X — tensor [20, 8]".
function varLabel(v: DataVariable): string {
  const shape = v.shape ? ` [${v.shape.join(', ')}]` : v.num_samples != null ? ` (${v.num_samples})` : ''
  return `${v.name} — ${v.kind}${shape}`
}

// Picker for the "memory" source: optionally choose live notebook objects for X
// (and y), pushing the inferred shape into the model's Input node(s). Leaving the
// picks empty is fine — codegen then emits a generic make_dataloaders(X, y).
function VariablePicker() {
  const config = useGraphStore((s) => s.data)
  const setDataParam = useGraphStore((s) => s.setDataParam)
  const nodes = useGraphStore((s) => s.nodes)
  const updateNodeParam = useGraphStore((s) => s.updateNodeParam)
  const { data: variables, refetch, isFetching } = useDataVariables(true)
  const options = variables ?? []

  // Input nodes in forward-arg order (canvas position), matching model_inputs.
  const inputNodes = nodes
    .filter((n) => n.data.nodeType === 'Input')
    .sort((a, b) => a.position.y - b.position.y || a.position.x - b.position.x || a.id.localeCompare(b.id))
  const multi = inputNodes.length > 1

  // Applying a pick pushes the variable's inferred shape+dtype onto that Input.
  const applyShape = (nodeId: string, varName: string) => {
    const v = options.find((o) => o.name === varName)
    if (!v?.input_shape) return
    updateNodeParam(nodeId, 'shape', v.input_shape.shape)
    updateNodeParam(nodeId, 'dtype', v.input_shape.dtype)
  }

  // Per-input picks (multi-input). Transient — the applied Input shapes persist on
  // the nodes; single-input keeps x_var (also read by codegen to detect a
  // DataLoader/Dataset pick).
  const [picks, setPicks] = useState<Record<string, string>>({})

  const varSelect = (value: string, onPick: (v: string) => void, noneLabel = '— select —') => (
    <select value={value} onChange={(e) => onPick(e.target.value)} style={{ ...SELECT_STYLE, marginBottom: 10 }}>
      <option value="">{noneLabel}</option>
      {options.map((v) => (
        <option key={v.name} value={v.name}>{varLabel(v)}</option>
      ))}
    </select>
  )
  const label = (text: string) => (
    <label style={{ display: 'block', color: 'var(--text-5)', fontSize: 11, marginBottom: 4 }}>{text}</label>
  )
  const sectionHeader = (text: string) => (
    <div style={{ fontSize: 10, color: 'var(--text-8)', textTransform: 'uppercase', letterSpacing: 1, margin: '4px 0 8px' }}>
      {text}
    </div>
  )

  return (
    <div style={{ marginBottom: 16, paddingBottom: 12, borderBottom: '1px solid var(--border)' }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 8 }}>
        <span style={{ color: 'var(--text-5)', fontSize: 11 }}>Notebook variables → Input shape</span>
        <button
          type="button"
          onClick={() => refetch()}
          style={{
            background: 'none', border: '1px solid var(--border)', borderRadius: 4,
            color: 'var(--text-4)', cursor: 'pointer', fontSize: 11, padding: '2px 8px', fontFamily: 'monospace',
          }}
        >
          {isFetching ? '…' : '↻ refresh'}
        </button>
      </div>

      {options.length === 0 ? (
        <div style={{ color: 'var(--text-7)', fontSize: 11, marginBottom: 8 }}>
          No data-like variables found. Define an array / tensor / DataLoader in the notebook, then refresh.
        </div>
      ) : null}

      {/* Input(s) — a named input still reads as an input under this header. */}
      {sectionHeader('Input(s)')}
      {multi ? (
        inputNodes.map((n, i) => {
          const name = typeof n.data.params.name === 'string' ? n.data.params.name.trim() : ''
          return (
            <div key={n.id}>
              {label(name || `Input ${i}`)}
              {varSelect(picks[n.id] ?? '', (v) => {
                setPicks((p) => ({ ...p, [n.id]: v }))
                applyShape(n.id, v)
              })}
            </div>
          )
        })
      ) : (
        varSelect(String(config.x_var ?? ''), (v) => {
          setDataParam('x_var', v)
          if (inputNodes[0]) applyShape(inputNodes[0].id, v)
        })
      )}

      {sectionHeader('Target(s)')}
      {varSelect(String(config.y_var ?? ''), (v) => setDataParam('y_var', v), '— none / not needed —')}
    </div>
  )
}

export function DataTab() {
  const { data: params } = useDataParams()
  const config = useGraphStore((s) => s.data)
  const setDataParam = useGraphStore((s) => s.setDataParam)
  const nodes = useGraphStore((s) => s.nodes)
  const toDomainGraph = useGraphStore((s) => s.toDomainGraph)

  // make_dataloaders() depends on the data config and the model's input count
  // (one X per Input), so refetch when either changes.
  const inputCount = nodes.filter((n) => n.data.nodeType === 'Input').length
  const configKey = JSON.stringify(config)

  // Generated make_dataloaders() preview. POST the *live* editor graph so the
  // preview always matches the canvas (input count included), rather than the
  // backend's cached graph — which can lag on reload until the next validate.
  const [code, setCode] = useState<string | null>(null)
  useEffect(() => {
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
    }, 400)
    return () => {
      cancelled = true
      window.clearTimeout(t)
    }
  }, [configKey, inputCount, toDomainGraph])

  // Effective config (stored value or the param default), used to evaluate
  // show_if — so a field appears once its controlling param matches even before
  // the user has touched it.
  const defaults = Object.fromEntries((params ?? []).map((p) => [p.name, p.default]))
  const effective: Record<string, unknown> = { ...defaults, ...config }
  const source = String(effective.source ?? 'memory')

  // The memory source renders a dedicated picker for x_var/y_var, so drop those
  // from the generic control list to avoid duplicate (plain text) inputs.
  const genericParams = (params ?? []).filter(
    (p) => paramVisible(p, effective) && !(source === 'memory' && (p.name === 'x_var' || p.name === 'y_var'))
  )

  // Optional params (e.g. resize) get the None toggle, matching the Inspector.
  const field = (param: ParamDef) => {
    const props = {
      param,
      value: config[param.name],
      nodeColor: 'var(--accent)',
      onChange: (next: unknown) => setDataParam(param.name, next),
    }
    return (
      <div key={param.name} style={{ marginBottom: 14 }}>
        <label style={{ display: 'block', color: 'var(--text-5)', fontSize: 11, marginBottom: 4 }}>
          {param.label}
        </label>
        {param.optional ? <OptionalControl {...props} /> : <ParamControl {...props} />}
      </div>
    )
  }

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

        {/* Source selector renders first; the variable picker sits right under it. */}
        {genericParams.filter((p) => p.name === 'source').map(field)}

        {source === 'memory' && <VariablePicker />}

        {genericParams.filter((p) => p.name !== 'source').map(field)}
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
