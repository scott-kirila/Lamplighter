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
        background: 'var(--surface)',
        border: `2px solid ${error ? 'var(--error-bright)' : selected ? data.color : 'var(--border)'}`,
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
          color: 'var(--text-on-accent)',
          fontSize: 13,
        }}
      >
        {data.label}
      </div>

      <div style={{ padding: '6px 0' }}>
        {(data.inputPins as Array<{ name: string; label: string }>).map((pin) => (
          <div
            key={pin.name}
            style={{ padding: '3px 12px 3px 18px', position: 'relative', color: 'var(--text-3)' }}
          >
            <Handle
              type="target"
              position={Position.Left}
              id={pin.name}
              style={{ background: 'var(--text-6)', width: 10, height: 10, left: -5, border: '2px solid var(--border)' }}
            />
            {pin.label}
          </div>
        ))}

        {(data.outputPins as Array<{ name: string; label: string }>).map((pin) => (
          <div
            key={pin.name}
            style={{ padding: '3px 18px 3px 12px', position: 'relative', textAlign: 'right', color: 'var(--text-3)' }}
          >
            <Handle
              type="source"
              position={Position.Right}
              id={pin.name}
              style={{ background: 'var(--text-6)', width: 10, height: 10, right: -5, border: '2px solid var(--border)' }}
            />
            {pin.label}
          </div>
        ))}
      </div>

      {(shape || error) && (
        <div
          style={{
            borderTop: '1px solid var(--border)',
            padding: '4px 10px',
            color: error ? 'var(--error)' : 'var(--accent)',
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
