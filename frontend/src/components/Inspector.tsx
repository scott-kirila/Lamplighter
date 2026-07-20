import { useGraphStore } from '../store/graphStore'
import { formatParamTerms, formatShape } from '../lib/formatShape'
import type { NodeDef } from '../types/graph'
import { OptionalControl, ParamControl } from './ParamControl'

interface InspectorProps {
  registry: Record<string, NodeDef>
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
        overflowY: 'auto',
        flexShrink: 0,
      }}
    >
      <div style={{ color: selectedNode.data.color, fontWeight: 700, fontSize: 14, marginBottom: 2 }}>
        {selectedNode.data.label}
      </div>
      <div style={{ color: 'var(--text-8)', fontSize: 11, marginBottom: 16 }}>
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
