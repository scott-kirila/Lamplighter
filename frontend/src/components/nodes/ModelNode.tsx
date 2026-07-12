import { memo } from 'react'
import { Handle, Position, type NodeProps } from '@xyflow/react'
import { useGraphStore } from '../../store/graphStore'
import type { ModelNode } from '../../store/graphStore'
import { formatShape } from '../../lib/formatShape'
import { concernColor, useNodeHealth } from '../../hooks/useTrainingHealth'

function ModelNode({ id, data, selected, dragging }: NodeProps<ModelNode>) {
  const shape = useGraphStore((s) => s.shapes[id])
  const error = useGraphStore((s) => s.errors[id])
  const connected = useGraphStore((s) =>
    s.edges.some((e) => e.source === id || e.target === id)
  )
  // Training-health dot from the last run — a colour (amber→red) that evokes a
  // reading, not a labelled verdict. Only surfaced once there's real concern, so
  // a healthy graph stays uncluttered; hover gives the factual context.
  const nh = useNodeHealth(id)
  const health = nh && nh.concern >= 0.35 ? nh : null

  // Ghost an unconnected node while dragging so the splice-target wire beneath it
  // (edges render under nodes) stays visible. Matches the palette drag preview.
  const ghosted = dragging && !connected

  // Show a user-given name in the title (e.g. "Input (myData)") so named IO nodes
  // are identifiable on the canvas without opening the Inspector.
  const name = typeof data.params.name === 'string' ? data.params.name.trim() : ''

  return (
    <div
      style={{
        position: 'relative',
        background: 'var(--surface)',
        border: `2px solid ${error ? 'var(--error-bright)' : selected ? data.color : 'var(--border)'}`,
        borderRadius: 8,
        minWidth: 160,
        fontFamily: 'monospace',
        fontSize: 12,
        boxShadow: selected ? `0 0 0 1px color-mix(in srgb, ${data.color} 20%, transparent)` : 'none',
        opacity: ghosted ? 0.6 : 1,
        transition: 'opacity 0.1s',
      }}
    >
      {health && (
        <div
          title={`Training health — ${health.note}`}
          style={{
            position: 'absolute',
            top: -7,
            right: -7,
            zIndex: 1,
            width: 14,
            height: 14,
            borderRadius: '50%',
            background: concernColor(health.concern),
            border: '2px solid var(--surface)',
          }}
        />
      )}
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
        {name && <span style={{ opacity: 0.75, fontWeight: 400 }}> ({name})</span>}
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
          {error ?? formatShape(shape, ' × ')}
        </div>
      )}
    </div>
  )
}

export default memo(ModelNode)
