import type { ReactNode } from 'react'
import { useGraphStore } from '../store/graphStore'

// Deleting a model drops it and all its layers — confirm first (mirrors the
// overview canvas and sidebar).
const confirmModelDelete = (name: string) =>
  window.confirm(`Delete the model "${name}"? This removes the model and all its layers from the project.`)

const sectionHeader = (text: string) => (
  <div
    style={{
      fontSize: 10, color: 'var(--text-8)', textTransform: 'uppercase',
      letterSpacing: 1, margin: '14px 0 8px',
    }}
  >
    {text}
  </div>
)

// The info pane for a model selected on the overview: its name (editable),
// a shape/wiring summary, and quick actions (open its canvas, delete it). Reads
// everything from the store so it stays live as the project changes.
export function ModelInspector({ modelId }: { modelId: string }) {
  const models = useGraphStore((s) => s.models)
  const activeModelId = useGraphStore((s) => s.activeModelId)
  const activeNodes = useGraphStore((s) => s.nodes)
  const modelGraphs = useGraphStore((s) => s.modelGraphs)
  const paramCounts = useGraphStore((s) => s.paramCounts)
  const modelResults = useGraphStore((s) => s.modelResults)
  const dataNodes = useGraphStore((s) => s.dataNodes)
  const links = useGraphStore((s) => s.links)
  const renameModel = useGraphStore((s) => s.renameModel)
  const openModel = useGraphStore((s) => s.openModel)
  const deleteModel = useGraphStore((s) => s.deleteModel)

  const model = models.find((m) => m.id === modelId)
  if (!model) return null

  // The model's own graph (active model lives in `nodes`; others are stashed).
  const nodes = modelId === activeModelId ? activeNodes : modelGraphs[modelId]?.nodes ?? []
  // Input nodes as named ports, in forward()-arg order (canvas position).
  const inputs = nodes
    .filter((n) => n.data.nodeType === 'Input')
    .slice()
    .sort((a, b) => a.position.y - b.position.y || a.position.x - b.position.x || a.id.localeCompare(b.id))
    .map((n, i) => String(n.data.params.name ?? '').trim() || `in ${i}`)
  const outputs = nodes.filter((n) => n.data.nodeType === 'Output').length

  // Total trainable parameters: the active model's counts live in the flat
  // paramCounts map; a stashed model's are in its modelResults entry.
  const counts = modelId === activeModelId ? paramCounts : modelResults[modelId]?.paramCounts ?? {}
  const totalParams = Object.values(counts).reduce((sum, p) => sum + p.count, 0)

  // Resolve a link endpoint (data node or model) to its display name.
  const nameOf = (id: string | null | undefined) =>
    dataNodes.find((d) => d.id === id)?.name ?? models.find((m) => m.id === id)?.name ?? '?'
  const fedBy = links.filter((l) => l.target_model === modelId).map((l) => nameOf(l.source_data ?? l.source_model))
  const feeds = links.filter((l) => l.source_model === modelId).map((l) => nameOf(l.target_model))

  const row = (label: string, value: ReactNode) => (
    <div style={{ display: 'flex', justifyContent: 'space-between', gap: 12, fontSize: 12.5, padding: '3px 0' }}>
      <span style={{ color: 'var(--text-5)' }}>{label}</span>
      <span style={{ textAlign: 'right', color: 'var(--text-2)' }}>{value}</span>
    </div>
  )

  return (
    <div
      style={{
        width: 300, flexShrink: 0, borderLeft: '1px solid var(--border)', background: 'var(--panel)',
        padding: 20, overflowY: 'auto', fontFamily: 'monospace',
      }}
    >
      <div
        style={{ color: 'var(--text-6)', fontSize: 10, letterSpacing: 1, textTransform: 'uppercase', marginBottom: 8 }}
      >
        model
      </div>
      <input
        value={model.name}
        onChange={(e) => renameModel(model.id, e.target.value)}
        style={{
          width: '100%', background: 'var(--field)', color: 'var(--text)', border: '1px solid var(--border)',
          borderRadius: 5, padding: '6px 8px', fontFamily: 'monospace', fontSize: 14, fontWeight: 700,
        }}
      />

      {sectionHeader('Summary')}
      {row('Layers', nodes.length)}
      {row('Inputs', inputs.length)}
      {row('Outputs', outputs)}
      {row('Parameters', totalParams > 0 ? totalParams.toLocaleString('en-US') : '—')}

      {inputs.length > 0 && (
        <>
          {sectionHeader('Input ports')}
          {inputs.map((name, i) => (
            <div key={i} style={{ fontSize: 12.5, color: 'var(--node-io)', padding: '2px 0' }}>
              • {name}
            </div>
          ))}
        </>
      )}

      {sectionHeader('Wiring')}
      {fedBy.length === 0 && feeds.length === 0 ? (
        <div style={{ fontSize: 12, color: 'var(--text-6)' }}>Not wired yet.</div>
      ) : (
        <>
          {fedBy.length > 0 && row('Fed by', fedBy.join(', '))}
          {feeds.length > 0 && row('Feeds', feeds.join(', '))}
        </>
      )}

      <button
        onClick={() => openModel(model.id)}
        style={{
          marginTop: 18, width: '100%', background: 'var(--accent)', color: 'var(--text-on-accent)',
          border: 'none', borderRadius: 6, padding: '8px 12px', fontFamily: 'monospace', fontSize: 13,
          fontWeight: 600, cursor: 'pointer',
        }}
      >
        Open in canvas ›
      </button>
      {models.length > 1 && (
        <button
          onClick={() => confirmModelDelete(model.name) && deleteModel(model.id)}
          style={{
            marginTop: 8, width: '100%', background: 'none', color: 'var(--text-5)',
            border: '1px solid var(--border)', borderRadius: 6, padding: '7px 12px', fontFamily: 'monospace',
            fontSize: 12, cursor: 'pointer',
          }}
        >
          Delete model
        </button>
      )}
    </div>
  )
}
