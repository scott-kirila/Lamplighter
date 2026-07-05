import { useCallback, useMemo, useState } from 'react'
import {
  Background,
  Controls,
  ReactFlow,
  ReactFlowProvider,
  type Connection,
  type Edge,
  type NodeChange,
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
  const links = useGraphStore((s) => s.links)
  const linkResults = useGraphStore((s) => s.linkResults)
  const addLink = useGraphStore((s) => s.addLink)
  const removeLink = useGraphStore((s) => s.removeLink)

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

  // Links → styled edges: the backend's shape-check drives the color (accent
  // when compatible, error when not) and the evidence label.
  const edges: Edge[] = useMemo(
    () =>
      links.map((l) => {
        const res = linkResults[l.id]
        const ok = res?.ok ?? true
        return {
          id: l.id,
          // The edge's source node is a data node (data→model) or a model
          // (model→model) — whichever this link carries.
          source: l.source_data ?? l.source_model ?? '',
          target: l.target_model,
          animated: true,
          label: res?.message,
          labelStyle: { fill: ok ? 'var(--text-3)' : 'var(--error)', fontFamily: 'monospace', fontSize: 11 },
          labelBgStyle: { fill: 'var(--panel)', fillOpacity: 0.9 },
          labelBgPadding: [6, 3] as [number, number],
          labelBgBorderRadius: 4,
          style: { stroke: ok ? 'var(--accent-2)' : 'var(--error-bright)', strokeWidth: 2 },
        }
      }),
    [links, linkResults]
  )

  // Apply live position changes to the model's sys_position on every drag tick,
  // so a model node follows the cursor (same as the model canvas) rather than
  // jumping only on drop. Non-position changes (selection/dimensions) are
  // React Flow's to track internally.
  const onNodesChange = useCallback(
    (changes: NodeChange<Node<SystemModelData>>[]) => {
      for (const c of changes) {
        if (c.type === 'position' && c.position) setModelSysPosition(c.id, c.position)
      }
    },
    [setModelSysPosition]
  )

  const onNodeDoubleClick = useCallback(
    (_e: React.MouseEvent, node: Node) => openModel(node.id),
    [openModel]
  )
  // The live positions are already in the store (onNodesChange); persist/broadcast
  // the final one on drop.
  const onNodeDragStop = useCallback(
    (_e: MouseEvent | TouchEvent, node: Node) => onModelMove?.([{ id: node.id, position: node.position }]),
    [onModelMove]
  )
  const onConnect = useCallback(
    (c: Connection) => {
      if (c.source && c.target) addLink(c.source, c.target)
    },
    [addLink]
  )
  const onEdgesDelete = useCallback(
    (eds: Edge[]) => eds.forEach((e) => removeLink(e.id)),
    [removeLink]
  )

  return (
    <ReactFlow
      nodes={nodes}
      edges={edges}
      nodeTypes={nodeTypes}
      onNodesChange={onNodesChange}
      onNodeDoubleClick={onNodeDoubleClick}
      onNodeDragStop={onNodeDragStop}
      onConnect={onConnect}
      onEdgesDelete={onEdgesDelete}
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
              <div
                onDoubleClick={() => setEditing(m.id)}
                title="Double-click to rename"
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
                  cursor: 'default',
                  userSelect: 'none',
                  overflow: 'hidden',
                  textOverflow: 'ellipsis',
                  whiteSpace: 'nowrap',
                }}
              >
                {m.name}
              </div>
              <button
                onClick={() => openModel(m.id)}
                title={`Open ${m.name}`}
                style={{
                  background: 'none',
                  color: 'var(--text-4)',
                  border: 'none',
                  cursor: 'pointer',
                  fontFamily: 'monospace',
                  fontSize: 15,
                  padding: '2px 4px',
                  lineHeight: 1,
                }}
              >
                ›
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
