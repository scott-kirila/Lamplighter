import { useGraphStore } from '../store/graphStore'
import type { NodeDef } from '../types/graph'

interface InspectorProps {
  registry: Record<string, NodeDef>
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
                <input
                  type="text"
                  value={String(selectedNode.data.params[param.name] ?? param.default)}
                  onChange={(e) => updateNodeParam(selectedNode.id, param.name, e.target.value)}
                  placeholder="e.g. 1, 3, 28, 28"
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
