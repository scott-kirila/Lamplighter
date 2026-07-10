import { Fragment, useEffect, useRef, useState } from 'react'
import { useGraphStore } from '../store/graphStore'
import { useModules } from '../hooks/useModules'
import { formatParamTerms, formatShape } from '../lib/formatShape'
import type { NodeDef, ParamDef } from '../types/graph'

interface InspectorProps {
  registry: Record<string, NodeDef>
}

// Split a stored shape string ("1, 3, 28, 28") into its dimension tokens.
export function parseDims(value: string): string[] {
  return value
    .split(',')
    .map((t) => t.trim())
    .filter((t) => t !== '')
}

// Structured editor for a shape param: a fixed "batch" chip for dim 0 (the
// leading dim is a placeholder — models accept any batch size, so it isn't
// editable) plus one number box per real per-sample dimension. The wire format
// stays a comma-joined string with a leading 1, so the backend parses it
// unchanged.
function ShapeEditor({
  value,
  color,
  onChange,
}: {
  value: string
  color: string
  onChange: (next: string) => void
}) {
  // Per-sample dims only (dim 0 is the locked batch placeholder), held as
  // strings so a box can be transiently empty while editing.
  const [dims, setDims] = useState<string[]>(() => parseDims(value).slice(1))
  // Last value we emitted, so an external change (remote tab, node switch)
  // re-seeds the local boxes but our own edits don't clobber in-progress typing.
  const emitted = useRef(value)

  useEffect(() => {
    if (value !== emitted.current) {
      setDims(parseDims(value).slice(1))
      emitted.current = value
    }
  }, [value])

  const commit = (next: string[]) => {
    setDims(next)
    const serialized = ['1', ...next.map((t) => t.trim()).filter((t) => t !== '')].join(', ')
    emitted.current = serialized
    onChange(serialized)
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
        <span style={{ color: 'var(--text-7)', fontSize: 11, width: 12, textAlign: 'right', flexShrink: 0 }}>
          0
        </span>
        <div
          title="Placeholder batch dimension — the model accepts any batch size; actual batching is set on the data node"
          style={{
            background: 'var(--field)',
            border: '1px dashed var(--border)',
            borderRadius: 4,
            padding: '6px 8px',
            color: 'var(--text-6)',
            fontSize: 12,
            flex: 1,
            minWidth: 0,
            fontFamily: 'monospace',
            cursor: 'help',
          }}
        >
          batch
        </div>
        <span style={{ width: 14, flexShrink: 0 }} />
      </div>
      {dims.map((d, i) => (
        <div key={i} style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <span style={{ color: 'var(--text-7)', fontSize: 11, width: 12, textAlign: 'right', flexShrink: 0 }}>
            {i + 1}
          </span>
          <input
            type="number"
            min={1}
            value={d}
            onChange={(e) => commit(dims.map((v, j) => (j === i ? e.target.value : v)))}
            style={{
              background: 'var(--field)',
              border: '1px solid var(--border)',
              borderRadius: 4,
              padding: '6px 8px',
              color: 'var(--text)',
              fontSize: 13,
              flex: 1,
              minWidth: 0,
              fontFamily: 'monospace',
            }}
          />
          <button
            type="button"
            onClick={() => commit(dims.filter((_, j) => j !== i))}
            title="Remove dimension"
            style={{
              background: 'none',
              border: 'none',
              color: 'var(--text-7)',
              cursor: 'pointer',
              fontSize: 14,
              padding: 0,
              lineHeight: 1,
              flexShrink: 0,
            }}
          >
            ×
          </button>
        </div>
      ))}
      <button
        type="button"
        onClick={() => commit([...dims, '1'])}
        title="Add dimension"
        style={{
          background: 'var(--field)',
          border: `1px dashed color-mix(in srgb, ${color} 33%, transparent)`,
          borderRadius: 4,
          color,
          cursor: 'pointer',
          fontSize: 12,
          padding: '6px 8px',
          width: '100%',
          fontFamily: 'monospace',
        }}
      >
        + dimension
      </button>
    </div>
  )
}

// Editor for an int-or-tuple param (kernel_size, stride, …): a fixed row of
// `arity` number boxes. Emits a scalar when all boxes match (so kernel_size 3
// stays `3` in generated code) and an array otherwise (`(3, 5)`).
function TupleEditor({
  value,
  arity,
  onChange,
}: {
  value: number | number[]
  arity: number
  onChange: (next: number | number[]) => void
}) {
  const normalize = (v: number | number[]): string[] =>
    Array.from({ length: arity }, (_, i) => String(Array.isArray(v) ? v[i] ?? v[0] ?? 0 : v))

  // Local strings so a box can be transiently empty while editing.
  const [boxes, setBoxes] = useState<string[]>(() => normalize(value))
  const emitted = useRef(value)
  useEffect(() => {
    if (value !== emitted.current) {
      setBoxes(normalize(value))
      emitted.current = value
    }
  }, [value]) // eslint-disable-line react-hooks/exhaustive-deps

  const commit = (next: string[]) => {
    setBoxes(next)
    const nums = next.map((t) => {
      const n = parseInt(t, 10)
      return Number.isNaN(n) ? 0 : n
    })
    const out = nums.every((n) => n === nums[0]) ? nums[0] : nums
    emitted.current = out
    onChange(out)
  }

  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
      {boxes.map((b, i) => (
        <Fragment key={i}>
          {i > 0 && (
            <span style={{ color: 'var(--text-6)', fontSize: 12, userSelect: 'none' }}>×</span>
          )}
          <input
            type="number"
            value={b}
            onChange={(e) => commit(boxes.map((v, j) => (j === i ? e.target.value : v)))}
            style={{
              background: 'var(--field)',
              border: '1px solid var(--border)',
              borderRadius: 4,
              padding: '6px 6px',
              color: 'var(--text)',
              fontSize: 13,
              width: 52,
              textAlign: 'center',
              fontFamily: 'monospace',
            }}
          />
        </Fragment>
      ))}
    </div>
  )
}

// Number field with a local draft, so the box can be transiently empty while
// editing (backspacing the last digit must not snap back to the old value —
// the same pattern ShapeEditor/TupleEditor use). Valid parses commit live;
// leaving the field empty/invalid restores the last committed value on blur.
function NumberField({
  value,
  isFloat,
  onChange,
}: {
  value: number
  isFloat: boolean
  onChange: (next: number) => void
}) {
  const [draft, setDraft] = useState(String(value))
  const emitted = useRef(value)
  useEffect(() => {
    if (value !== emitted.current) {
      setDraft(String(value))
      emitted.current = value
    }
  }, [value])

  return (
    <input
      type="number"
      step={isFloat ? 0.05 : 1}
      value={draft}
      onChange={(e) => {
        setDraft(e.target.value)
        const v = isFloat ? parseFloat(e.target.value) : parseInt(e.target.value, 10)
        if (!isNaN(v)) {
          emitted.current = v
          onChange(v)
        }
      }}
      onBlur={() => {
        const v = isFloat ? parseFloat(draft) : parseInt(draft, 10)
        if (isNaN(v)) setDraft(String(emitted.current))
      }}
      style={FIELD_STYLE}
    />
  )
}

const FIELD_STYLE = {
  background: 'var(--field)',
  border: '1px solid var(--border)',
  borderRadius: 4,
  padding: '6px 8px',
  color: 'var(--text)',
  fontSize: 13,
  width: '100%',
  fontFamily: 'monospace',
} as const

// The editor for a single param's base type. `value` is the stored value (may be
// undefined for an unset param); display falls back to the definition's default.
// The Custom node's class selector: a picker over the session's registered
// nn.Modules (sess.modules(Name=Class)), with the same ↻-refresh idiom as the
// data-node variable picker. A previously-picked-but-now-unregistered name is
// kept as a (missing) option so the selection is never silently dropped.
function ModulePicker({ value, onChange }: { value: string; onChange: (next: unknown) => void }) {
  const { data: modules, refetch, isFetching } = useModules(true)
  const names = (modules ?? []).map((m) => m.name)
  const doc = (modules ?? []).find((m) => m.name === value)?.doc
  return (
    <div>
      <div style={{ display: 'flex', gap: 6 }}>
        <select
          value={value}
          onChange={(e) => onChange(e.target.value)}
          style={{ ...FIELD_STYLE, cursor: 'pointer', flex: 1 }}
        >
          <option value="">— pick a module —</option>
          {value && !names.includes(value) && <option value={value}>{value} (missing)</option>}
          {names.map((name) => (
            <option key={name} value={name}>
              {name}
            </option>
          ))}
        </select>
        <button
          onClick={() => refetch()}
          title="Refresh registered modules (sess.modules(...))"
          style={{
            background: 'none', color: 'var(--text-4)', border: '1px solid var(--border)',
            borderRadius: 6, padding: '0 8px', cursor: 'pointer', fontFamily: 'monospace', fontSize: 13,
          }}
        >
          {isFetching ? '…' : '↻'}
        </button>
      </div>
      {names.length === 0 && (
        <div style={{ color: 'var(--text-6)', fontSize: 10.5, marginTop: 5, lineHeight: 1.4 }}>
          Nothing registered — run sess.modules(MyBlock=MyBlock) in the notebook, then ↻.
        </div>
      )}
      {doc && (
        <div style={{ color: 'var(--text-6)', fontSize: 10.5, marginTop: 5, lineHeight: 1.4 }}>{doc}</div>
      )}
    </div>
  )
}

export function ParamControl({
  param,
  value,
  nodeColor,
  onChange,
}: {
  param: ParamDef
  value: unknown
  nodeColor: string
  onChange: (next: unknown) => void
}) {
  if (param.type === 'bool') {
    return (
      <input
        type="checkbox"
        checked={Boolean(value)}
        onChange={(e) => onChange(e.target.checked)}
        style={{ accentColor: nodeColor, width: 16, height: 16, cursor: 'pointer' }}
      />
    )
  }
  if (param.type === 'shape') {
    return <ShapeEditor value={String(value ?? param.default)} color={nodeColor} onChange={onChange} />
  }
  if (param.type === 'string') {
    return (
      <input
        type="text"
        value={String(value ?? param.default ?? '')}
        placeholder="auto"
        onChange={(e) => onChange(e.target.value)}
        style={FIELD_STYLE}
      />
    )
  }
  if (param.type === 'enum') {
    return (
      <select
        value={String(value ?? param.default)}
        onChange={(e) => onChange(e.target.value)}
        style={{ ...FIELD_STYLE, cursor: 'pointer' }}
      >
        {(param.choices ?? []).map((choice) => (
          <option key={choice} value={choice}>
            {choice}
          </option>
        ))}
      </select>
    )
  }
  if (param.type === 'module') {
    return <ModulePicker value={String(value ?? '')} onChange={onChange} />
  }
  if (param.type === 'tuple') {
    return (
      <TupleEditor
        value={(value ?? param.default) as number | number[]}
        arity={param.arity ?? 2}
        onChange={onChange}
      />
    )
  }
  if (param.type === 'multienum') {
    const selected = Array.isArray(value) ? (value as string[]) : []
    return (
      <div style={{ display: 'flex', flexDirection: 'column', gap: 5 }}>
        {(param.choices ?? []).map((choice) => (
          <label key={choice} style={{ display: 'flex', alignItems: 'center', gap: 6, color: 'var(--text-3)', fontSize: 12, cursor: 'pointer' }}>
            <input
              type="checkbox"
              checked={selected.includes(choice)}
              onChange={(e) =>
                onChange(e.target.checked ? [...selected, choice] : selected.filter((c) => c !== choice))
              }
              style={{ accentColor: nodeColor, width: 14, height: 14, cursor: 'pointer' }}
            />
            {choice}
          </label>
        ))}
      </div>
    )
  }
  return (
    <NumberField
      value={Number(value ?? param.default)}
      isFloat={param.type === 'float'}
      onChange={onChange}
    />
  )
}

// Wraps a base control with a None toggle for an optional param. Unchecked = null
// (None); checking it seeds a value so the base control has something to edit.
export function OptionalControl({
  param,
  value,
  nodeColor,
  onChange,
}: {
  param: ParamDef
  value: unknown
  nodeColor: string
  onChange: (next: unknown) => void
}) {
  // An unset param shows its default; an explicit null is the None state.
  const current = value === undefined ? param.default : value
  const isNone = current === null
  const enableSeed = param.default ?? (param.type === 'float' ? 0 : 1)
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
      <input
        type="checkbox"
        checked={!isNone}
        onChange={(e) => onChange(e.target.checked ? enableSeed : null)}
        title="Set a value (otherwise None)"
        style={{ accentColor: nodeColor, width: 16, height: 16, cursor: 'pointer', flexShrink: 0 }}
      />
      {isNone ? (
        <span style={{ color: 'var(--text-6)', fontSize: 12, fontFamily: 'monospace' }}>None</span>
      ) : (
        <div style={{ flex: 1 }}>
          <ParamControl param={param} value={current} nodeColor={nodeColor} onChange={onChange} />
        </div>
      )}
    </div>
  )
}

export function Inspector({ registry }: InspectorProps) {
  const selectedNodeId = useGraphStore((s) => s.selectedNodeId)
  const nodes = useGraphStore((s) => s.nodes)
  const updateNodeParam = useGraphStore((s) => s.updateNodeParam)
  const shape = useGraphStore((s) => (selectedNodeId ? s.shapes[selectedNodeId] : undefined))
  const nodePinShapes = useGraphStore((s) => (selectedNodeId ? s.pinShapes[selectedNodeId] : undefined))
  const paramCount = useGraphStore((s) => (selectedNodeId ? s.paramCounts[selectedNodeId] : undefined))
  const error = useGraphStore((s) => (selectedNodeId ? s.errors[selectedNodeId] : undefined))

  const selectedNode = nodes.find((n) => n.id === selectedNodeId)

  if (!selectedNode) {
    return (
      <div
        style={{
          width: 240,
          background: 'var(--panel)',
          borderLeft: '1px solid var(--border)',
          padding: 20,
          fontFamily: 'monospace',
          color: 'var(--text-8)',
          fontSize: 13,
          flexShrink: 0,
        }}
      >
        Select a node to inspect
      </div>
    )
  }

  const nodeDef = registry[selectedNode.data.nodeType]

  // A multi-output node (LSTM/RNN/GRU) shows a shape per output pin; a single-
  // output node keeps the bare [dims] readout.
  const outputPins = nodeDef?.outputs ?? []
  const perPin =
    !error && outputPins.length > 1 && nodePinShapes
      ? outputPins.filter((p) => nodePinShapes[p.name])
      : []

  return (
    <div
      style={{
        width: 240,
        background: 'var(--panel)',
        borderLeft: '1px solid var(--border)',
        padding: 16,
        fontFamily: 'monospace',
        overflowY: 'auto',
        flexShrink: 0,
      }}
    >
      <div style={{ color: selectedNode.data.color, fontWeight: 700, fontSize: 14, marginBottom: 2 }}>
        {selectedNode.data.label}
      </div>
      <div style={{ color: 'var(--text-8)', fontSize: 11, marginBottom: 16, fontFamily: 'monospace' }}>
        {selectedNode.id.slice(0, 8)}
      </div>

      {(shape || error) && (
        <div
          style={{
            background: 'var(--field)',
            border: `1px solid ${error ? 'var(--error-border-strong)' : 'var(--border)'}`,
            borderRadius: 6,
            padding: '8px 10px',
            marginBottom: 16,
            fontSize: 12,
            color: error ? 'var(--error)' : 'var(--accent)',
          }}
        >
          {error ? (
            error
          ) : perPin.length > 0 ? (
            <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
              {perPin.map((pin) => (
                <div key={pin.name} style={{ display: 'flex', justifyContent: 'space-between', gap: 10 }}>
                  <span style={{ color: 'var(--text-6)' }}>{pin.label}</span>
                  <span>[{formatShape(nodePinShapes![pin.name], ', ', pin.name === 'output')}]</span>
                </div>
              ))}
            </div>
          ) : (
            `[${formatShape(shape!, ', ')}]`
          )}
          {!error && paramCount !== undefined && (
            <div style={{ color: 'var(--text-5)', fontSize: 11, marginTop: 4 }}>
              {paramCount.count.toLocaleString('en-US')} parameters
              {paramCount.terms.length > 0 && (
                <span style={{ color: 'var(--text-7)' }}>
                  {' '}= {formatParamTerms(paramCount.terms)}
                </span>
              )}
            </div>
          )}
        </div>
      )}

      {nodeDef && nodeDef.params.length > 0 && (
        <div>
          <div
            style={{
              fontSize: 10,
              color: 'var(--text-8)',
              textTransform: 'uppercase',
              letterSpacing: 1,
              marginBottom: 10,
            }}
          >
            Parameters
          </div>
          {nodeDef.params.map((param) => (
            <div key={param.name} style={{ marginBottom: 14 }}>
              <label style={{ display: 'block', color: 'var(--text-5)', fontSize: 11, marginBottom: 4 }}>
                {param.label}
              </label>
              {param.optional ? (
                <OptionalControl
                  param={param}
                  value={selectedNode.data.params[param.name]}
                  nodeColor={selectedNode.data.color}
                  onChange={(next) => updateNodeParam(selectedNode.id, param.name, next)}
                />
              ) : (
                <ParamControl
                  param={param}
                  value={selectedNode.data.params[param.name]}
                  nodeColor={selectedNode.data.color}
                  onChange={(next) => updateNodeParam(selectedNode.id, param.name, next)}
                />
              )}
            </div>
          ))}
        </div>
      )}

      {/* Help text, live from the installed torch's docstring (or the authored
          line for Lamplighter-native nodes). Collapsed by default; the body is
          the full Args/Shape text, so it renders pre-wrapped monospace. */}
      {nodeDef?.doc && (
        <details style={{ marginTop: 4 }}>
          <summary
            style={{
              fontSize: 10,
              color: 'var(--text-8)',
              textTransform: 'uppercase',
              letterSpacing: 1,
              cursor: 'pointer',
              userSelect: 'none',
            }}
          >
            PyTorch docs
          </summary>
          <div
            style={{
              marginTop: 8,
              background: 'var(--field)',
              border: '1px solid var(--border)',
              borderRadius: 6,
              padding: '8px 10px',
              fontSize: 11,
              lineHeight: 1.5,
              color: 'var(--text-4)',
              whiteSpace: 'pre-wrap',
              overflowWrap: 'break-word',
              maxHeight: 320,
              overflowY: 'auto',
            }}
          >
            {nodeDef.doc.body || nodeDef.doc.summary}
          </div>
        </details>
      )}
    </div>
  )
}
