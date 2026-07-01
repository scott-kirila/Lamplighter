import { useEffect, useState } from 'react'
import { useGraphStore } from '../store/graphStore'
import { useDataParams } from '../hooks/useDataParams'
import { useDataVariables, type DataVariable } from '../hooks/useDataVariables'
import { paramVisible } from '../lib/paramVisible'
import { ParamControl } from './Inspector'

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

// Picker for the "variable" source: choose live notebook objects for X (and y),
// and push the inferred shape into the model's Input node.
function VariablePicker() {
  const config = useGraphStore((s) => s.data)
  const setDataParam = useGraphStore((s) => s.setDataParam)
  const nodes = useGraphStore((s) => s.nodes)
  const updateNodeParam = useGraphStore((s) => s.updateNodeParam)
  const { data: variables, refetch, isFetching } = useDataVariables(true)

  const xVar = String(config.x_var ?? '')
  const yVar = String(config.y_var ?? '')
  const selected = (variables ?? []).find((v) => v.name === xVar)
  const inputNode = nodes.find((n) => n.data.nodeType === 'Input')

  const applyShape = () => {
    if (!selected?.input_shape || !inputNode) return
    updateNodeParam(inputNode.id, 'shape', selected.input_shape.shape)
    updateNodeParam(inputNode.id, 'dtype', selected.input_shape.dtype)
  }

  const options = variables ?? []
  return (
    <div style={{ marginBottom: 16, paddingBottom: 12, borderBottom: '1px solid var(--border)' }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 8 }}>
        <span style={{ color: 'var(--text-5)', fontSize: 11 }}>Notebook variables</span>
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

      <label style={{ display: 'block', color: 'var(--text-5)', fontSize: 11, marginBottom: 4 }}>Inputs (X)</label>
      <select value={xVar} onChange={(e) => setDataParam('x_var', e.target.value)} style={{ ...SELECT_STYLE, marginBottom: 10 }}>
        <option value="">— select —</option>
        {options.map((v) => (
          <option key={v.name} value={v.name}>{varLabel(v)}</option>
        ))}
      </select>

      <label style={{ display: 'block', color: 'var(--text-5)', fontSize: 11, marginBottom: 4 }}>Targets (y)</label>
      <select value={yVar} onChange={(e) => setDataParam('y_var', e.target.value)} style={{ ...SELECT_STYLE, marginBottom: 10 }}>
        <option value="">— none / not needed —</option>
        {options.map((v) => (
          <option key={v.name} value={v.name}>{varLabel(v)}</option>
        ))}
      </select>

      <button
        type="button"
        onClick={applyShape}
        disabled={!selected?.input_shape || !inputNode}
        title={!inputNode ? 'Add an Input node to the model first' : 'Set the Input node shape from this variable'}
        style={{
          width: '100%', background: 'var(--field)', border: `1px dashed var(--accent)`,
          borderRadius: 4, color: selected?.input_shape && inputNode ? 'var(--accent)' : 'var(--text-7)',
          cursor: selected?.input_shape && inputNode ? 'pointer' : 'not-allowed',
          fontSize: 12, padding: '6px 8px', fontFamily: 'monospace',
        }}
      >
        {selected?.input_shape ? `→ set Input to [${selected.input_shape.shape}]` : '→ set Input shape'}
      </button>
    </div>
  )
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
  // show_if — so a field appears once its controlling param matches even before
  // the user has touched it.
  const defaults = Object.fromEntries((params ?? []).map((p) => [p.name, p.default]))
  const effective: Record<string, unknown> = { ...defaults, ...config }
  const source = String(effective.source ?? 'tensors')

  // The variable source renders a dedicated picker for x_var/y_var, so drop those
  // from the generic control list to avoid duplicate (plain text) inputs.
  const genericParams = (params ?? []).filter(
    (p) => paramVisible(p, effective) && !(source === 'variable' && (p.name === 'x_var' || p.name === 'y_var'))
  )

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

        {/* Source selector renders generically below; the picker sits under it. */}
        {genericParams
          .filter((p) => p.name === 'source')
          .map((param) => (
            <div key={param.name} style={{ marginBottom: 14 }}>
              <label style={{ display: 'block', color: 'var(--text-5)', fontSize: 11, marginBottom: 4 }}>
                {param.label}
              </label>
              <ParamControl param={param} value={config[param.name]} nodeColor="var(--accent)" onChange={(next) => setDataParam(param.name, next)} />
            </div>
          ))}

        {source === 'variable' && <VariablePicker />}

        {genericParams
          .filter((p) => p.name !== 'source')
          .map((param) => (
            <div key={param.name} style={{ marginBottom: 14 }}>
              <label style={{ display: 'block', color: 'var(--text-5)', fontSize: 11, marginBottom: 4 }}>
                {param.label}
              </label>
              <ParamControl param={param} value={config[param.name]} nodeColor="var(--accent)" onChange={(next) => setDataParam(param.name, next)} />
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
