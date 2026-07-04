import { useEffect, useState } from 'react'
import { useGraphStore } from '../store/graphStore'
import { useDataParams } from '../hooks/useDataParams'
import { useRecipes } from '../hooks/useRecipes'
import { useDataVariables, type DataVariable } from '../hooks/useDataVariables'
import { paramVisible } from '../lib/paramVisible'
import { OptionalControl, ParamControl } from './Inspector'
import type { ParamDef } from '../types/graph'

interface DiagnosticCheck {
  level: 'ok' | 'warn' | 'error'
  title: string
  detail: string
}

const CHECK_ICON: Record<string, { glyph: string; color: string }> = {
  ok: { glyph: '✓', color: 'var(--accent)' },
  warn: { glyph: '⚠', color: 'var(--warn)' },
  error: { glyph: '✗', color: 'var(--error)' },
}

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

// Picker for the "memory" source: choose from the session's registered data for X
// (and y), pushing the inferred shape into the model's Input node(s). Leaving the
// picks empty is fine — codegen then emits a generic make_dataloaders(X, y).
function VariablePicker({ needsTargets }: { needsTargets: boolean }) {
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

  // Per-input picks (multi-input), persisted in the graph's data config keyed by
  // Input node id — the in-kernel runner resolves them (in forward-arg order) at
  // run time. Single-input keeps x_var (also read by codegen to detect a
  // DataLoader/Dataset pick).
  const picks = (config.x_vars ?? {}) as Record<string, string>
  const setPick = (nodeId: string, varName: string) =>
    setDataParam('x_vars', { ...picks, [nodeId]: varName })

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
        <span style={{ color: 'var(--text-5)', fontSize: 11 }}>Registered data → Input shape</span>
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
          Nothing registered yet — run <span style={{ color: 'var(--accent)' }}>sess.data(X=X, y=y)</span> in
          the notebook, then refresh.
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
                setPick(n.id, v)
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

      {/* An adversarial recipe (needs_targets=false) trains on X alone. */}
      {needsTargets && (
        <>
          {sectionHeader('Target(s)')}
          {varSelect(String(config.y_var ?? ''), (v) => setDataParam('y_var', v), '— none / not needed —')}
        </>
      )}
    </div>
  )
}

export function DataTab() {
  const { data: params } = useDataParams()
  const { data: recipes } = useRecipes()
  const config = useGraphStore((s) => s.data)
  const setDataParam = useGraphStore((s) => s.setDataParam)
  const nodes = useGraphStore((s) => s.nodes)
  const training = useGraphStore((s) => s.training)
  const toProject = useGraphStore((s) => s.toProject)

  // The selected recipe's data contract: an adversarial loop needs no targets
  // and no validation split, so the picker/fields for those are hidden.
  const recipe = recipes?.find((r) => r.name === (training.recipe ?? 'supervised'))
  const needsTargets = recipe?.needs_targets ?? true
  const hasVal = recipe?.has_val ?? true
  // Registry listing — updates live on sess.data(...) pushes, which re-keys the
  // diagnostics below (data changing must re-run the checks).
  const { data: registered } = useDataVariables(true)

  // Live data↔model diagnostics. Debounced POST of the live graph, re-run when
  // the data config, the model (node params), the loss, or the registry change.
  const [checks, setChecks] = useState<DiagnosticCheck[]>([])
  const diagKey = JSON.stringify([
    config,
    training,
    nodes.map((n) => [n.data.nodeType, n.data.params]),
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

  // Effective config (stored value or the param default), used to evaluate
  // show_if — so a field appears once its controlling param matches even before
  // the user has touched it.
  const defaults = Object.fromEntries((params ?? []).map((p) => [p.name, p.default]))
  const effective: Record<string, unknown> = { ...defaults, ...config }
  const source = String(effective.source ?? 'memory')

  // The memory source renders a dedicated picker for x_var/y_var, so drop those
  // from the generic control list to avoid duplicate (plain text) inputs.
  const genericParams = (params ?? []).filter(
    (p) =>
      paramVisible(p, effective) &&
      !(source === 'memory' && (p.name === 'x_var' || p.name === 'y_var')) &&
      // An adversarial recipe has no held-out split.
      !(p.name === 'val_split' && !hasVal)
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

        {source === 'memory' && <VariablePicker needsTargets={needsTargets} />}

        {genericParams.filter((p) => p.name !== 'source').map(field)}
      </div>

      {/* Main pane — live data↔model diagnostics. (The generated
          make_dataloaders() opens via the titlebar's Show code button.) */}
      <div style={{ flex: 1, display: 'flex', flexDirection: 'column', background: 'var(--bg)', minWidth: 0 }}>
        <div
          style={{
            height: 36,
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
          Data ↔ model diagnostics
        </div>
        <div style={{ flex: 1, overflowY: 'auto', padding: '16px 20px', fontFamily: 'monospace', fontSize: 12 }}>
          {checks.length === 0 ? (
            <div style={{ color: 'var(--text-6)', lineHeight: 1.8 }}>
              Register data with <span style={{ color: 'var(--accent)' }}>sess.data(X=X, y=y)</span> and
              pick it on the left — checks against the model appear here.
            </div>
          ) : (
            checks.map((c, i) => {
              const icon = CHECK_ICON[c.level] ?? CHECK_ICON.warn
              return (
                <div key={i} style={{ display: 'flex', gap: 10, marginBottom: 10, lineHeight: 1.5 }}>
                  <span style={{ color: icon.color, flexShrink: 0 }}>{icon.glyph}</span>
                  <div style={{ minWidth: 0 }}>
                    <div style={{ color: c.level === 'error' ? 'var(--error)' : 'var(--text-2)' }}>
                      {c.title}
                    </div>
                    {c.detail && <div style={{ color: 'var(--text-5)', fontSize: 11 }}>{c.detail}</div>}
                  </div>
                </div>
              )
            })
          )}
        </div>
      </div>
    </div>
  )
}
