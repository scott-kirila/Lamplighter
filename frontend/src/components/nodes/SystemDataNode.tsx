import { memo } from 'react'
import { Handle, Position, type NodeProps, type Node } from '@xyflow/react'

// A data source on the system canvas — a dataset or a noise generator. Its
// right handle(s) wire into a model's input port. No target handle: data has no
// input of its own. A *labeled* dataset (its label conditions a model, e.g. a
// cGAN) splits its output into two named pins — `x` (features) and `y` (label);
// every other data node keeps a single plain output.
export interface SystemDataData extends Record<string, unknown> {
  name: string
  kind: string // 'dataset' | 'noise'
  labeled: boolean // dataset whose y (label) pin is in use → show x/y pins
}

export type SystemDataNode = Node<SystemDataData>

const handleStyle = {
  background: 'var(--text-6)',
  width: 11,
  height: 11,
  border: '2px solid var(--border)',
} as const

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
      {data.labeled ? (
        // Two named output pins: X (features) and y (label).
        <div style={{ borderTop: '1px solid var(--border)', padding: '2px 0' }}>
          {(['x', 'y'] as const).map((pin) => (
            <div
              key={pin}
              style={{ position: 'relative', padding: '3px 12px', fontSize: 11, color: 'var(--text-4)', textAlign: 'right' }}
            >
              {pin}
              <Handle
                id={pin}
                type="source"
                position={Position.Right}
                style={{ ...handleStyle, right: -6, top: '50%', transform: 'translateY(-50%)' }}
              />
            </div>
          ))}
        </div>
      ) : (
        <Handle type="source" position={Position.Right} style={{ ...handleStyle, right: -6 }} />
      )}
    </div>
  )
}

export default memo(SystemDataNode)
