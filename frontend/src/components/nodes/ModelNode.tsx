import { memo } from 'react'
import { Handle, Position, type NodeProps } from '@xyflow/react'
import { useGraphStore } from '../../store/graphStore'
import type { ModelNode } from '../../store/graphStore'

function ModelNode({ id, data, selected }: NodeProps<ModelNode>) {
  const shape = useGraphStore((s) => s.shapes[id])
  const error = useGraphStore((s) => s.errors[id])

  return (
    <div
      style={{
        background: '#1e1e2e',
        border: `2px solid ${error ? '#ff4444' : selected ? data.color : '#2a2a4a'}`,
        borderRadius: 8,
        minWidth: 160,
        fontFamily: 'monospace',
        fontSize: 12,
        boxShadow: selected ? `0 0 0 1px ${data.color}33` : 'none',
      }}
    >
      <div
        style={{
          background: data.color,
          padding: '6px 12px',
          borderRadius: '6px 6px 0 0',
          fontWeight: 600,
          color: '#fff',
          fontSize: 13,
        }}
      >
        {data.label}
      </div>

      <div style={{ padding: '6px 0' }}>
        {(data.inputPins as Array<{ name: string; label: string }>).map((pin) => (
          <div
            key={pin.name}
            style={{ padding: '3px 12px 3px 18px', position: 'relative', color: '#aaa' }}
          >
            <Handle
              type="target"
              position={Position.Left}
              id={pin.name}
              style={{ background: '#666', width: 10, height: 10, left: -5, border: '2px solid #2a2a4a' }}
            />
            {pin.label}
          </div>
        ))}

        {(data.outputPins as Array<{ name: string; label: string }>).map((pin) => (
          <div
            key={pin.name}
            style={{ padding: '3px 18px 3px 12px', position: 'relative', textAlign: 'right', color: '#aaa' }}
          >
            <Handle
              type="source"
              position={Position.Right}
              id={pin.name}
              style={{ background: '#666', width: 10, height: 10, right: -5, border: '2px solid #2a2a4a' }}
            />
            {pin.label}
          </div>
        ))}
      </div>

      {(shape || error) && (
        <div
          style={{
            borderTop: '1px solid #2a2a4a',
            padding: '4px 10px',
            color: error ? '#ff6b6b' : '#4a9eff',
            fontSize: 11,
          }}
        >
          {error ?? shape.join(' × ')}
        </div>
      )}
    </div>
  )
}

export default memo(ModelNode)
