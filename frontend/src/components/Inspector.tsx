import { useEffect, useRef, useState } from 'react'
import { useGraphStore } from '../store/graphStore'
import type { NodeDef } from '../types/graph'

interface InspectorProps {
  registry: Record<string, NodeDef>
}

// Split a stored shape string ("1, 3, 28, 28") into its dimension tokens.
function parseDims(value: string): string[] {
  return value
    .split(',')
    .map((t) => t.trim())
    .filter((t) => t !== '')
}

// Structured editor for a shape param: one number box per dimension, with
// add/remove controls. Kills the typo-prone free-text entry while keeping the
// wire format a comma-joined string, so the backend parses it unchanged.
function ShapeEditor({
  value,
  color,
  onChange,
}: {
  value: string
  color: string
  onChange: (next: string) => void
}) {
  // Dims held as strings so a box can be transiently empty while editing.
  const [dims, setDims] = useState<string[]>(() => parseDims(value))
  // Last value we emitted, so an external change (remote tab, node switch)
  // re-seeds the local boxes but our own edits don't clobber in-progress typing.
  const emitted = useRef(value)

  useEffect(() => {
    if (value !== emitted.current) {
      setDims(parseDims(value))
      emitted.current = value
    }
  }, [value])

  const commit = (next: string[]) => {
    setDims(next)
    const serialized = next.map((t) => t.trim()).filter((t) => t !== '').join(', ')
    emitted.current = serialized
    onChange(serialized)
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
      {dims.map((d, i) => (
        <div key={i} style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <span style={{ color: '#555', fontSize: 11, width: 12, textAlign: 'right', flexShrink: 0 }}>
            {i}
          </span>
          <input
            type="number"
            min={1}
            value={d}
            onChange={(e) => commit(dims.map((v, j) => (j === i ? e.target.value : v)))}
            style={{
              background: '#1a1a2e',
              border: '1px solid #2a2a4a',
              borderRadius: 4,
              padding: '6px 8px',
              color: '#e0e0e0',
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
              color: '#555',
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
          background: '#1a1a2e',
          border: `1px dashed ${color}55`,
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

export function Inspector({ registry }: InspectorProps) {
  const selectedNodeId = useGraphStore((s) => s.selectedNodeId)
  const nodes = useGraphStore((s) => s.nodes)
  const updateNodeParam = useGraphStore((s) => s.updateNodeParam)
  const shape = useGraphStore((s) => (selectedNodeId ? s.shapes[selectedNodeId] : undefined))
  const error = useGraphStore((s) => (selectedNodeId ? s.errors[selectedNodeId] : undefined))

  const selectedNode = nodes.find((n) => n.id === selectedNodeId)

  if (!selectedNode) {
    return (
      <div
        style={{
          width: 240,
          background: '#12121f',
          borderLeft: '1px solid #2a2a4a',
          padding: 20,
          fontFamily: 'monospace',
          color: '#444',
          fontSize: 13,
          flexShrink: 0,
        }}
      >
        Select a node to inspect
      </div>
    )
  }

  const nodeDef = registry[selectedNode.data.nodeType]

  return (
    <div
      style={{
        width: 240,
        background: '#12121f',
        borderLeft: '1px solid #2a2a4a',
        padding: 16,
        fontFamily: 'monospace',
        overflowY: 'auto',
        flexShrink: 0,
      }}
    >
      <div style={{ color: selectedNode.data.color, fontWeight: 700, fontSize: 14, marginBottom: 2 }}>
        {selectedNode.data.label}
      </div>
      <div style={{ color: '#444', fontSize: 11, marginBottom: 16, fontFamily: 'monospace' }}>
        {selectedNode.id.slice(0, 8)}
      </div>

      {(shape || error) && (
        <div
          style={{
            background: '#1a1a2e',
            border: `1px solid ${error ? '#ff444444' : '#2a2a4a'}`,
            borderRadius: 6,
            padding: '8px 10px',
            marginBottom: 16,
            fontSize: 12,
            color: error ? '#ff6b6b' : '#4a9eff',
          }}
        >
          {error ?? `[${shape!.join(', ')}]`}
        </div>
      )}

      {nodeDef && nodeDef.params.length > 0 && (
        <div>
          <div
            style={{
              fontSize: 10,
              color: '#444',
              textTransform: 'uppercase',
              letterSpacing: 1,
              marginBottom: 10,
            }}
          >
            Parameters
          </div>
          {nodeDef.params.map((param) => (
            <div key={param.name} style={{ marginBottom: 14 }}>
              <label style={{ display: 'block', color: '#777', fontSize: 11, marginBottom: 4 }}>
                {param.label}
              </label>
              {param.type === 'bool' ? (
                <input
                  type="checkbox"
                  checked={Boolean(selectedNode.data.params[param.name])}
                  onChange={(e) => updateNodeParam(selectedNode.id, param.name, e.target.checked)}
                  style={{ accentColor: selectedNode.data.color, width: 16, height: 16, cursor: 'pointer' }}
                />
              ) : param.type === 'shape' ? (
                <ShapeEditor
                  value={String(selectedNode.data.params[param.name] ?? param.default)}
                  color={selectedNode.data.color}
                  onChange={(next) => updateNodeParam(selectedNode.id, param.name, next)}
                />
              ) : (
                <input
                  type="number"
                  step={param.type === 'float' ? 0.05 : 1}
                  value={String(selectedNode.data.params[param.name] ?? param.default)}
                  onChange={(e) => {
                    const v = param.type === 'float' ? parseFloat(e.target.value) : parseInt(e.target.value, 10)
                    if (!isNaN(v)) updateNodeParam(selectedNode.id, param.name, v)
                  }}
                  style={{
                    background: '#1a1a2e',
                    border: '1px solid #2a2a4a',
                    borderRadius: 4,
                    padding: '6px 8px',
                    color: '#e0e0e0',
                    fontSize: 13,
                    width: '100%',
                    fontFamily: 'monospace',
                  }}
                />
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
