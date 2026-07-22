import { memo } from 'react'
import { Handle, Position, type NodeProps, type Node } from '@xyflow/react'
import { dataKindColor, dataKindTextColor, handleStyle } from './shared'

// A data source on the overview canvas — a dataset or a noise generator. Its
// right handle(s) wire into a model's input port. No target handle: data has no
// input of its own. A *labeled* dataset (its label conditions a model, e.g. a
// cGAN) splits its output into two named pins — `x` (features) and `y` (label);
// every other data node keeps a single plain output.
export interface OverviewDataData extends Record<string, unknown> {
  name: string
  kind: string // 'dataset' | 'noise' | 'env'
  labeled: boolean // dataset whose y (label) pin is in use → show x/y pins
  // Selected on the overview canvas. Driven by the store, not React Flow's
  // internal selection, so it survives the nodes being re-derived each render.
  selected: boolean
}

export type OverviewDataNode = Node<OverviewDataData>

function OverviewDataNode({ data }: NodeProps<OverviewDataNode>) {
  const isNoise = data.kind === 'noise'
  const isEnv = data.kind === 'env'
  // The node's own color carries the selection border + halo, matching how the
  // nn layer nodes highlight: noise = warn, env = a distinct green, else accent.
  const color = dataKindColor(data.kind)
  const subtitle = isNoise ? 'noise source' : isEnv ? 'environment (RL)' : 'dataset'
  return (
    <div
      style={{
        background: 'var(--surface)',
        border: `2px solid ${data.selected ? color : 'var(--border)'}`,
        borderRadius: 10,
        minWidth: 150,
        boxShadow: data.selected ? `0 0 0 1px color-mix(in srgb, ${color} 20%, transparent)` : 'none',
      }}
    >
      <div
        style={{
          background: color,
          padding: '7px 12px',
          borderRadius: '8px 8px 0 0',
          fontWeight: 700,
          color: dataKindTextColor(data.kind),
          fontSize: 13,
        }}
      >
        {data.name}
      </div>
      <div style={{ padding: '6px 12px', color: 'var(--text-4)', fontSize: 11 }}>
        {subtitle}
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

export default memo(OverviewDataNode)
