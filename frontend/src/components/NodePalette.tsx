import type { NodeDef } from '../types/graph'

const CATEGORIES = ['io', 'layers', 'activations', 'ops']
const CATEGORY_LABELS: Record<string, string> = {
  io: 'I / O',
  layers: 'Layers',
  activations: 'Activations',
  ops: 'Ops',
}

interface NodePaletteProps {
  registry: Record<string, NodeDef>
}

export function NodePalette({ registry }: NodePaletteProps) {
  const byCategory = Object.values(registry).reduce<Record<string, NodeDef[]>>((acc, def) => {
    ;(acc[def.category] ??= []).push(def)
    return acc
  }, {})

  const onDragStart = (e: React.DragEvent, nodeType: string) => {
    e.dataTransfer.setData('application/lamplighter-node', nodeType)
    e.dataTransfer.effectAllowed = 'move'
  }

  return (
    <div
      style={{
        width: 200,
        background: '#12121f',
        borderRight: '1px solid #2a2a4a',
        padding: '12px 0',
        fontFamily: 'monospace',
        overflowY: 'auto',
        flexShrink: 0,
      }}
    >
      <div
        style={{
          padding: '0 12px 12px',
          fontSize: 10,
          color: '#555',
          textTransform: 'uppercase',
          letterSpacing: 2,
        }}
      >
        Nodes
      </div>

      {CATEGORIES.map((cat) => {
        const nodes = byCategory[cat]
        if (!nodes?.length) return null
        return (
          <div key={cat} style={{ marginBottom: 8 }}>
            <div
              style={{
                padding: '4px 12px',
                fontSize: 10,
                color: '#444',
                textTransform: 'uppercase',
                letterSpacing: 1,
              }}
            >
              {CATEGORY_LABELS[cat] ?? cat}
            </div>
            {nodes.map((def) => (
              <PaletteItem key={def.type} def={def} onDragStart={onDragStart} />
            ))}
          </div>
        )
      })}
    </div>
  )
}

function PaletteItem({
  def,
  onDragStart,
}: {
  def: NodeDef
  onDragStart: (e: React.DragEvent, type: string) => void
}) {
  return (
    <div
      draggable
      onDragStart={(e) => onDragStart(e, def.type)}
      style={{
        padding: '7px 12px',
        cursor: 'grab',
        display: 'flex',
        alignItems: 'center',
        gap: 8,
        color: '#ccc',
        fontSize: 13,
        userSelect: 'none',
        transition: 'background 0.1s',
      }}
      onMouseEnter={(e) => (e.currentTarget.style.background = '#1e1e2e')}
      onMouseLeave={(e) => (e.currentTarget.style.background = 'transparent')}
    >
      <div
        style={{
          width: 10,
          height: 10,
          borderRadius: 3,
          background: def.color,
          flexShrink: 0,
        }}
      />
      {def.label}
    </div>
  )
}
