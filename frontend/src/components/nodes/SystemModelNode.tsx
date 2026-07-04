import { memo } from 'react'
import { Handle, Position, type NodeProps, type Node } from '@xyflow/react'

// A whole model as a node on the system canvas: name, a small subtitle (node
// count), and left/right handles so models can be linked (dataflow claims land
// in a later phase). Double-click opens the model's editing canvas.
export interface SystemModelData extends Record<string, unknown> {
  name: string
  subtitle: string
  active: boolean
}

export type SystemModelNode = Node<SystemModelData>

function SystemModelNode({ data, selected }: NodeProps<SystemModelNode>) {
  return (
    <div
      style={{
        background: 'var(--surface)',
        border: `2px solid ${data.active ? 'var(--accent)' : selected ? 'var(--accent-2)' : 'var(--border)'}`,
        borderRadius: 10,
        minWidth: 170,
        fontFamily: 'monospace',
        boxShadow: data.active
          ? '0 0 0 1px color-mix(in srgb, var(--accent) 25%, transparent)'
          : 'none',
      }}
    >
      <Handle
        type="target"
        position={Position.Left}
        style={{ background: 'var(--text-6)', width: 11, height: 11, left: -6, border: '2px solid var(--border)' }}
      />
      <div
        style={{
          background: 'var(--accent)',
          padding: '8px 14px',
          borderRadius: '8px 8px 0 0',
          fontWeight: 700,
          color: 'var(--text-on-accent)',
          fontSize: 14,
        }}
      >
        {data.name}
      </div>
      <div style={{ padding: '8px 14px', color: 'var(--text-4)', fontSize: 12 }}>
        {data.subtitle}
      </div>
      <Handle
        type="source"
        position={Position.Right}
        style={{ background: 'var(--text-6)', width: 11, height: 11, right: -6, border: '2px solid var(--border)' }}
      />
    </div>
  )
}

export default memo(SystemModelNode)
