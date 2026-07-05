import { memo } from 'react'
import { Handle, Position, type NodeProps, type Node } from '@xyflow/react'

// A data source on the system canvas — a dataset or a noise generator. Its
// right handle wires into a model's input port. No target handle: data has no
// input of its own.
export interface SystemDataData extends Record<string, unknown> {
  name: string
  kind: string // 'dataset' | 'noise'
}

export type SystemDataNode = Node<SystemDataData>

function SystemDataNode({ data, selected }: NodeProps<SystemDataNode>) {
  const isNoise = data.kind === 'noise'
  return (
    <div
      style={{
        background: 'var(--surface)',
        border: `2px solid ${selected ? 'var(--accent-2)' : 'var(--border)'}`,
        borderRadius: 10,
        minWidth: 150,
        fontFamily: 'monospace',
      }}
    >
      <div
        style={{
          background: isNoise ? 'var(--warn)' : 'var(--accent-2)',
          padding: '7px 12px',
          borderRadius: '8px 8px 0 0',
          fontWeight: 700,
          color: 'var(--text-on-accent)',
          fontSize: 13,
        }}
      >
        {data.name}
      </div>
      <div style={{ padding: '6px 12px', color: 'var(--text-4)', fontSize: 11 }}>
        {isNoise ? 'noise source' : 'dataset'}
      </div>
      <Handle
        type="source"
        position={Position.Right}
        style={{ background: 'var(--text-6)', width: 11, height: 11, right: -6, border: '2px solid var(--border)' }}
      />
    </div>
  )
}

export default memo(SystemDataNode)
