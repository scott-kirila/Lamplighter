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
import type { NodeDef, NodeMove } from '../types/graph'
import SystemModelNode, { type SystemModelData } from './nodes/SystemModelNode'

const nodeTypes: NodeTypes = { systemModel: SystemModelNode }

interface SystemViewProps {
  registry: Record<string, NodeDef>
  onModelMove?: (moves: NodeMove[]) => void
}

// The high-level view: every model as a node you can arrange and open, plus a
// sidebar to add, jump between, rename, and remove them. Single-model projects
// show one model here (and land on its canvas by default) — the system view is
// the hub the multi-model workflows (GAN, …) build on.
function SystemCanvas({ onModelMove }: { onModelMove?: (moves: NodeMove[]) => void }) {
  const models = useGraphStore((s) => s.models)
  const activeModelId = useGraphStore((s) => s.activeModelId)
  const activeCount = useGraphStore((s) => s.nodes.length)
  const modelGraphs = useGraphStore((s) => s.modelGraphs)
  const openModel = useGraphStore((s) => s.openModel)
  const setModelSysPosition = useGraphStore((s) => s.setModelSysPosition)

  const nodeCountFor = useCallback(
    (id: string) => (id === activeModelId ? activeCount : modelGraphs[id]?.nodes.length ?? 0),
    [activeModelId, activeCount, modelGraphs]
  )

  const nodes: Node<SystemModelData>[] = useMemo(
    () =>
      models.map((m) => {
        const count = nodeCountFor(m.id)
        return {
          id: m.id,
          type: 'systemModel',
          position: m.sysPosition,
          data: {
            name: m.name,
            subtitle: `${count} node${count === 1 ? '' : 's'}`,
            active: m.id === activeModelId,
          },
        }
      }),
    [models, activeModelId, nodeCountFor]
  )

  const onNodeDoubleClick = useCallback(
    (_e: React.MouseEvent, node: Node) => openModel(node.id),
    [openModel]
  )
  const onNodeDragStop = useCallback(
    (_e: MouseEvent | TouchEvent, node: Node) => {
      setModelSysPosition(node.id, node.position)
      onModelMove?.([{ id: node.id, position: node.position }])
    },
    [setModelSysPosition, onModelMove]
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

function Sidebar({ registry }: { registry: Record<string, NodeDef> }) {
  const models = useGraphStore((s) => s.models)
  const activeModelId = useGraphStore((s) => s.activeModelId)
  const openModel = useGraphStore((s) => s.openModel)
  const renameModel = useGraphStore((s) => s.renameModel)
  const addModel = useGraphStore((s) => s.addModel)
  const deleteModel = useGraphStore((s) => s.deleteModel)
  const [editing, setEditing] = useState<string | null>(null)

  return (
    <div
      style={{
        width: 210,
        flexShrink: 0,
        borderRight: '1px solid var(--border)',
        background: 'var(--panel)',
        padding: '10px 8px',
        display: 'flex',
        flexDirection: 'column',
        gap: 4,
      }}
    >
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          padding: '2px 6px 6px',
        }}
      >
        <span style={{ fontFamily: 'monospace', fontSize: 11, color: 'var(--text-6)' }}>MODELS</span>
        <button
          onClick={() => addModel(registry)}
          title="Add a model"
          style={{
            background: 'none',
            color: 'var(--accent)',
            border: '1px solid var(--border)',
            borderRadius: 5,
            padding: '1px 8px',
            fontFamily: 'monospace',
            fontSize: 14,
            cursor: 'pointer',
            lineHeight: 1.2,
          }}
        >
          ＋
        </button>
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
            <>
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
                  overflow: 'hidden',
                  textOverflow: 'ellipsis',
                  whiteSpace: 'nowrap',
                }}
              >
                {m.name}
              </button>
              {models.length > 1 && (
                <button
                  onClick={() => deleteModel(m.id)}
                  title={`Delete ${m.name}`}
                  style={{
                    background: 'none',
                    color: 'var(--text-6)',
                    border: 'none',
                    cursor: 'pointer',
                    fontFamily: 'monospace',
                    fontSize: 13,
                    padding: '2px 4px',
                  }}
                >
                  ✕
                </button>
              )}
            </>
          )}
        </div>
      ))}
    </div>
  )
}

export function SystemView({ registry, onModelMove }: SystemViewProps) {
  return (
    <div style={{ display: 'flex', flex: 1, overflow: 'hidden' }}>
      <Sidebar registry={registry} />
      <ReactFlowProvider>
        <div style={{ flex: 1, height: '100%' }}>
          <SystemCanvas onModelMove={onModelMove} />
        </div>
      </ReactFlowProvider>
    </div>
  )
}
