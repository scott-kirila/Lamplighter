import { useCallback, useMemo, useState } from 'react'
import {
  Background,
  Controls,
  ReactFlow,
  ReactFlowProvider,
  type NodeTypes,
  type Node,
} from '@xyflow/react'
import '@xyflow/react/dist/style.css'
import { useGraphStore } from '../store/graphStore'
import SystemModelNode, { type SystemModelData } from './nodes/SystemModelNode'

const nodeTypes: NodeTypes = { systemModel: SystemModelNode }

// The high-level view: every model as a node you can arrange and open, plus a
// sidebar to jump between and rename them. Single-model projects show one model
// here (and land on its canvas by default) — the system view is the hub the
// multi-model workflows (GAN, …) build on.
function SystemCanvas() {
  const models = useGraphStore((s) => s.models)
  const activeModelId = useGraphStore((s) => s.activeModelId)
  const nodeCount = useGraphStore((s) => s.nodes.length)
  const openModel = useGraphStore((s) => s.openModel)
  const setModelSysPosition = useGraphStore((s) => s.setModelSysPosition)

  const nodes: Node<SystemModelData>[] = useMemo(
    () =>
      models.map((m) => ({
        id: m.id,
        type: 'systemModel',
        position: m.sysPosition,
        data: {
          name: m.name,
          // Only the active model's graph is loaded, so only it has a live count.
          subtitle: m.id === activeModelId ? `${nodeCount} node${nodeCount === 1 ? '' : 's'}` : 'model',
          active: m.id === activeModelId,
        },
      })),
    [models, activeModelId, nodeCount]
  )

  const onNodeDoubleClick = useCallback(
    (_e: React.MouseEvent, node: Node) => openModel(node.id),
    [openModel]
  )
  const onNodeDragStop = useCallback(
    (_e: MouseEvent | TouchEvent, node: Node) => setModelSysPosition(node.id, node.position),
    [setModelSysPosition]
  )

  return (
    <ReactFlow
      nodes={nodes}
      edges={[]}
      nodeTypes={nodeTypes}
      onNodeDoubleClick={onNodeDoubleClick}
      onNodeDragStop={onNodeDragStop}
      fitView
      style={{ background: 'var(--bg)' }}
    >
      <Background color="var(--canvas-dots)" gap={24} size={1} />
      <Controls style={{ background: 'var(--surface)', border: '1px solid var(--border)' }} />
    </ReactFlow>
  )
}

function Sidebar() {
  const models = useGraphStore((s) => s.models)
  const activeModelId = useGraphStore((s) => s.activeModelId)
  const openModel = useGraphStore((s) => s.openModel)
  const renameModel = useGraphStore((s) => s.renameModel)
  const [editing, setEditing] = useState<string | null>(null)

  return (
    <div
      style={{
        width: 200,
        flexShrink: 0,
        borderRight: '1px solid var(--border)',
        background: 'var(--panel)',
        padding: '10px 8px',
        display: 'flex',
        flexDirection: 'column',
        gap: 4,
      }}
    >
      <div style={{ fontFamily: 'monospace', fontSize: 11, color: 'var(--text-6)', padding: '2px 6px 6px' }}>
        MODELS
      </div>
      {models.map((m) => (
        <div key={m.id} style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
          {editing === m.id ? (
            <input
              autoFocus
              defaultValue={m.name}
              onBlur={(e) => {
                const v = e.target.value.trim()
                if (v) renameModel(m.id, v)
                setEditing(null)
              }}
              onKeyDown={(e) => {
                if (e.key === 'Enter') (e.target as HTMLInputElement).blur()
                if (e.key === 'Escape') setEditing(null)
              }}
              style={{
                flex: 1,
                background: 'var(--surface)',
                color: 'var(--text)',
                border: '1px solid var(--accent)',
                borderRadius: 5,
                padding: '5px 8px',
                fontFamily: 'monospace',
                fontSize: 13,
              }}
            />
          ) : (
            <button
              onClick={() => openModel(m.id)}
              onDoubleClick={() => setEditing(m.id)}
              title="Click to open · double-click to rename"
              style={{
                flex: 1,
                textAlign: 'left',
                background: m.id === activeModelId ? 'var(--surface)' : 'none',
                color: m.id === activeModelId ? 'var(--text)' : 'var(--text-4)',
                border: `1px solid ${m.id === activeModelId ? 'var(--accent)' : 'transparent'}`,
                borderRadius: 5,
                padding: '5px 8px',
                fontFamily: 'monospace',
                fontSize: 13,
                cursor: 'pointer',
              }}
            >
              {m.name}
            </button>
          )}
        </div>
      ))}
    </div>
  )
}

export function SystemView() {
  return (
    <div style={{ display: 'flex', flex: 1, overflow: 'hidden' }}>
      <Sidebar />
      <ReactFlowProvider>
        <div style={{ flex: 1, height: '100%' }}>
          <SystemCanvas />
        </div>
      </ReactFlowProvider>
    </div>
  )
}
