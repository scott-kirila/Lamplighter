import { memo } from 'react'
import { Handle, Position, type NodeProps, type Node } from '@xyflow/react'

// A whole model as a node on the overview canvas: name, a small subtitle (node
// count), and left/right handles so models can be linked. A model with several
// Input nodes exposes one *named* input port per Input, so a data node can fan
// out different wires to different ports (a cGAN's noise vs. label). A single-
// input model keeps one plain left handle. Double-click opens its editing canvas.
export interface OverviewModelPort {
  id: string // the Input node id — becomes the link's target_input
  name: string
}
export interface OverviewModelData extends Record<string, unknown> {
  name: string
  subtitle: string
  // Selected on the overview canvas (drives the info pane + selection highlight).
  // Driven by the store, not React Flow's internal selection, so it survives the
  // nodes being re-derived on every render.
  selected: boolean
  inputs: OverviewModelPort[]
}

export type OverviewModelNode = Node<OverviewModelData>

const handleStyle = {
  background: 'var(--text-6)',
  width: 11,
  height: 11,
  border: '2px solid var(--border)',
} as const

function OverviewModelNode({ data }: NodeProps<OverviewModelNode>) {
  const multiPort = data.inputs.length > 1
  return (
    <div
      style={{
        background: 'var(--surface)',
        border: `2px solid ${data.selected ? 'var(--accent)' : 'var(--border)'}`,
        borderRadius: 10,
        minWidth: 170,
        fontFamily: 'monospace',
        // Selected: a border + faint halo in the node's own color (accent, its
        // header color) — matching how the nn layer nodes highlight, and keeping
        // it distinct from the data nodes' accent-2/warn.
        boxShadow: data.selected
          ? '0 0 0 1px color-mix(in srgb, var(--accent) 20%, transparent)'
          : 'none',
      }}
    >
      {/* Single-input (or none): one plain target handle, unchanged. */}
      {!multiPort && (
        <Handle type="target" position={Position.Left} style={{ ...handleStyle, left: -6 }} />
      )}
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
      {/* Multi-input: one named port row per Input, each with its own handle. */}
      {multiPort && (
        <div style={{ borderTop: '1px solid var(--border)', padding: '2px 0' }}>
          {data.inputs.map((p) => (
            <div
              key={p.id}
              style={{
                position: 'relative',
                padding: '3px 14px',
                fontSize: 11,
                color: 'var(--text-4)',
                overflow: 'hidden',
                textOverflow: 'ellipsis',
                whiteSpace: 'nowrap',
              }}
            >
              <Handle
                id={p.id}
                type="target"
                position={Position.Left}
                style={{ ...handleStyle, left: -6, top: '50%', transform: 'translateY(-50%)' }}
              />
              {p.name}
            </div>
          ))}
        </div>
      )}
      <Handle type="source" position={Position.Right} style={{ ...handleStyle, right: -6 }} />
    </div>
  )
}

export default memo(OverviewModelNode)
