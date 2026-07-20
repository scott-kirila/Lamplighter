import { useRef, useState } from 'react'
import { useGraphStore } from '../store/graphStore'
import { nodeColor } from '../lib/nodeColor'
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

// Split an ordered node list into contiguous subcategory runs. The registry
// keeps same-subcategory nodes together, so a run is one palette section; nodes
// with no subcategory form a null-headed run rendered without a sub-header.
// Section order follows the registry; nodes *within* a section are sorted
// alphabetically by label.
function groupBySubcategory(nodes: NodeDef[]): { subcategory: string | null; nodes: NodeDef[] }[] {
  const groups: { subcategory: string | null; nodes: NodeDef[] }[] = []
  for (const def of nodes) {
    const sub = def.subcategory ?? null
    const last = groups[groups.length - 1]
    if (last && last.subcategory === sub) last.nodes.push(def)
    else groups.push({ subcategory: sub, nodes: [def] })
  }
  for (const g of groups) {
    g.nodes.sort((a, b) => a.label.localeCompare(b.label, undefined, { numeric: true }))
  }
  return groups
}

export function NodePalette({ registry }: NodePaletteProps) {
  const setPaletteDragType = useGraphStore((s) => s.setPaletteDragType)
  const setSpliceTarget = useGraphStore((s) => s.setSpliceTarget)
  const [query, setQuery] = useState('')

  const q = query.trim().toLowerCase()
  const matches = (def: NodeDef) =>
    !q || def.label.toLowerCase().includes(q) || def.type.toLowerCase().includes(q)

  const byCategory = Object.values(registry).reduce<Record<string, NodeDef[]>>((acc, def) => {
    ;(acc[def.category] ??= []).push(def)
    return acc
  }, {})

  const onDragStart = (e: React.DragEvent, nodeType: string) => {
    e.dataTransfer.setData('application/lamplighter-node', nodeType)
    e.dataTransfer.effectAllowed = 'move'
    setPaletteDragType(nodeType)
  }

  // Clear the drag/highlight state when the drag ends (drop or cancel).
  const onDragEnd = () => {
    setPaletteDragType(null)
    setSpliceTarget(null)
  }

  return (
    <div
      style={{
        width: 200,
        background: 'var(--panel)',
        borderRight: '1px solid var(--border)',
        padding: '12px 0',
        overflowY: 'auto',
        flexShrink: 0,
      }}
    >
      <div
        style={{
          padding: '0 12px 8px',
          fontSize: 10,
          color: 'var(--text-7)',
          textTransform: 'uppercase',
          letterSpacing: 2,
        }}
      >
        Nodes
      </div>
      <div style={{ padding: '0 12px 10px' }}>
        <input
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Search nodes…"
          style={{
            width: '100%',
            background: 'var(--field)',
            color: 'var(--text)',
            border: '1px solid var(--border)',
            borderRadius: 5,
            padding: '5px 8px',
            fontSize: 12,
          }}
        />
      </div>

      {CATEGORIES.map((cat) => {
        const nodes = (byCategory[cat] ?? []).filter(matches)
        if (!nodes.length) return null
        return (
          <div key={cat} style={{ marginBottom: 8 }}>
            <div
              style={{
                padding: '4px 12px',
                fontSize: 10,
                color: 'var(--text-8)',
                textTransform: 'uppercase',
                letterSpacing: 1,
              }}
            >
              {CATEGORY_LABELS[cat] ?? cat}
            </div>
            {groupBySubcategory(nodes).map((group, gi) => (
              <div key={gi}>
                {group.subcategory && (
                  <div
                    style={{
                      padding: '3px 12px 3px 20px',
                      fontSize: 9.5,
                      color: 'var(--text-7)',
                      letterSpacing: 0.5,
                    }}
                  >
                    {group.subcategory}
                  </div>
                )}
                {group.nodes.map((def) => (
                  <PaletteItem
                    key={def.type}
                    def={def}
                    indent={!!group.subcategory}
                    onDragStart={onDragStart}
                    onDragEnd={onDragEnd}
                  />
                ))}
              </div>
            ))}
          </div>
        )
      })}

      {q && !CATEGORIES.some((cat) => (byCategory[cat] ?? []).some(matches)) && (
        <div style={{ padding: '2px 12px', fontSize: 12, color: 'var(--text-6)' }}>
          No nodes match “{query}”.
        </div>
      )}
    </div>
  )
}

function PaletteItem({
  def,
  indent = false,
  onDragStart,
  onDragEnd,
}: {
  def: NodeDef
  indent?: boolean
  onDragStart: (e: React.DragEvent, type: string) => void
  onDragEnd: () => void
}) {
  // Hidden, node-shaped element handed to setDragImage so the drag preview looks
  // like the node being placed rather than this list row.
  const previewRef = useRef<HTMLDivElement>(null)
  // Docstring tooltip: appears after a short hover (so scanning the list stays
  // quiet), fixed-positioned right of the row to escape the sidebar's overflow.
  const [tip, setTip] = useState<{ top: number; left: number } | null>(null)
  const tipTimer = useRef<number | null>(null)
  const hideTip = () => {
    if (tipTimer.current !== null) window.clearTimeout(tipTimer.current)
    tipTimer.current = null
    setTip(null)
  }

  return (
    <div
      draggable
      onDragStart={(e) => {
        const el = previewRef.current
        if (el) {
          // Center the preview on the cursor so the hit point matches where the
          // node visually sits — drop-to-insert tests the cursor as the center.
          const r = el.getBoundingClientRect()
          e.dataTransfer.setDragImage(el, r.width / 2, r.height / 2)
          e.dataTransfer.setData(
            'application/lamplighter-offset',
            JSON.stringify({ x: r.width / 2, y: r.height / 2 })
          )
        }
        hideTip()
        onDragStart(e, def.type)
      }}
      onDragEnd={onDragEnd}
      style={{
        padding: indent ? '7px 12px 7px 20px' : '7px 12px',
        cursor: 'grab',
        display: 'flex',
        alignItems: 'center',
        gap: 8,
        color: 'var(--text-2)',
        fontSize: 13,
        userSelect: 'none',
        transition: 'background 0.1s',
      }}
      onMouseEnter={(e) => {
        e.currentTarget.style.background = 'var(--surface)'
        if (def.doc) {
          const r = e.currentTarget.getBoundingClientRect()
          tipTimer.current = window.setTimeout(() => setTip({ top: r.top, left: r.right }), 350)
        }
      }}
      onMouseLeave={(e) => {
        e.currentTarget.style.background = 'transparent'
        hideTip()
      }}
    >
      <div
        style={{
          width: 10,
          height: 10,
          borderRadius: 3,
          background: nodeColor(def.category, def.type),
          flexShrink: 0,
        }}
      />
      {def.label}
      {tip && def.doc && (
        <div
          style={{
            position: 'fixed',
            left: tip.left + 8,
            top: tip.top - 4,
            maxWidth: 300,
            zIndex: 100,
            background: 'var(--surface)',
            border: '1px solid var(--border)',
            borderRadius: 6,
            padding: '8px 10px',
            fontSize: 11.5,
            lineHeight: 1.45,
            color: 'var(--text-3)',
            boxShadow: 'var(--shadow-md)',
            pointerEvents: 'none',
            whiteSpace: 'normal',
          }}
        >
          {def.doc.summary}
        </div>
      )}
      <NodePreview def={def} ref={previewRef} />
    </div>
  )
}

// A static visual stand-in for the canvas node, mirroring ModelNode's shell
// (colored header + pin rows). Rendered off-screen purely to serve as the drag
// image; it must stay painted (not display:none) for the browser to snapshot it.
function NodePreview({ def, ref }: { def: NodeDef; ref: React.Ref<HTMLDivElement> }) {
  const pinRow = (label: string, side: 'in' | 'out') => (
    <div
      key={`${side}-${label}`}
      style={{
        padding: side === 'in' ? '3px 12px 3px 18px' : '3px 18px 3px 12px',
        textAlign: side === 'in' ? 'left' : 'right',
        color: 'var(--text-3)',
      }}
    >
      {label}
    </div>
  )

  return (
    <div
      ref={ref}
      style={{
        position: 'fixed',
        top: 0,
        left: -9999,
        pointerEvents: 'none',
        // Match the ghosted look of an unconnected node dragged on-canvas.
        opacity: 0.6,
        background: 'var(--surface)',
        border: '2px solid var(--border)',
        borderRadius: 8,
        minWidth: 160,
        fontSize: 12,
      }}
    >
      <div
        style={{
          background: nodeColor(def.category, def.type),
          padding: '6px 12px',
          borderRadius: '6px 6px 0 0',
          fontWeight: 600,
          color: 'var(--text-on-accent)',
          fontSize: 13,
        }}
      >
        {def.label}
      </div>
      <div style={{ padding: '6px 0' }}>
        {def.inputs.map((p) => pinRow(p.label, 'in'))}
        {def.outputs.map((p) => pinRow(p.label, 'out'))}
      </div>
    </div>
  )
}
