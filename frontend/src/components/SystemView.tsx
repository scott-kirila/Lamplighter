import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import {
  Background,
  Controls,
  ReactFlow,
  ReactFlowProvider,
  useReactFlow,
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
import SystemDataNode, { type SystemDataData } from './nodes/SystemDataNode'
import { DataNodeInspector } from './DataNodeInspector'

const nodeTypes: NodeTypes = { systemModel: SystemModelNode, systemData: SystemDataNode }

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
  const dataNodes = useGraphStore((s) => s.dataNodes)
  const setDataNodeSysPosition = useGraphStore((s) => s.setDataNodeSysPosition)
  const setSelectedDataNode = useGraphStore((s) => s.setSelectedDataNode)
  const links = useGraphStore((s) => s.links)
  const linkResults = useGraphStore((s) => s.linkResults)
  const addLink = useGraphStore((s) => s.addLink)
  const removeLink = useGraphStore((s) => s.removeLink)

  const nodeCountFor = useCallback(
    (id: string) => (id === activeModelId ? activeCount : modelGraphs[id]?.nodes.length ?? 0),
    [activeModelId, activeCount, modelGraphs]
  )
  const isDataNode = useCallback((id: string) => dataNodes.some((d) => d.id === id), [dataNodes])

  const nodes: Node<SystemModelData | SystemDataData>[] = useMemo(
    () => [
      ...models.map((m) => {
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
      ...dataNodes.map((d) => ({
        id: d.id,
        type: 'systemData',
        position: d.sysPosition,
        data: { name: d.name, kind: d.kind },
      })),
    ],
    [models, activeModelId, nodeCountFor, dataNodes]
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
    (changes: NodeChange<Node<SystemModelData | SystemDataData>>[]) => {
      for (const c of changes) {
        if (c.type === 'position' && c.position) {
          if (isDataNode(c.id)) setDataNodeSysPosition(c.id, c.position)
          else setModelSysPosition(c.id, c.position)
        }
      }
    },
    [setModelSysPosition, setDataNodeSysPosition, isDataNode]
  )

  // Double-clicking a *model* drills into it; data nodes aren't drillable.
  const onNodeDoubleClick = useCallback(
    (_e: React.MouseEvent, node: Node) => {
      if (!isDataNode(node.id)) openModel(node.id)
    },
    [openModel, isDataNode]
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
  // Clicking a data node opens its Inspector; clicking a model or the pane clears it.
  const onNodeClick = useCallback(
    (_e: React.MouseEvent, node: Node) => setSelectedDataNode(isDataNode(node.id) ? node.id : null),
    [setSelectedDataNode, isDataNode]
  )
  const onPaneClick = useCallback(() => setSelectedDataNode(null), [setSelectedDataNode])

  // Re-frame the canvas when a node is added, so a new model/data node lands in
  // view (fitView otherwise only runs once, on mount).
  const { fitView } = useReactFlow()
  const count = models.length + dataNodes.length
  const prev = useRef(count)
  useEffect(() => {
    if (count > prev.current) fitView({ duration: 200, padding: 0.2 })
    prev.current = count
  }, [count, fitView])

  return (
    <ReactFlow
      nodes={nodes}
      edges={edges}
      nodeTypes={nodeTypes}
      onNodesChange={onNodesChange}
      onNodeClick={onNodeClick}
      onPaneClick={onPaneClick}
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
  const dataNodes = useGraphStore((s) => s.dataNodes)
  const addDataNode = useGraphStore((s) => s.addDataNode)
  const removeDataNode = useGraphStore((s) => s.removeDataNode)
  const [editing, setEditing] = useState<string | null>(null)

  const addBtn: React.CSSProperties = {
    background: 'none', color: 'var(--accent)', border: '1px solid var(--border)',
    borderRadius: 5, padding: '1px 7px', fontFamily: 'monospace', fontSize: 11,
    cursor: 'pointer', lineHeight: 1.4,
  }

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

      {/* Data sources — wire these into a model's input on the canvas. */}
      <div
        style={{
          display: 'flex', alignItems: 'center', justifyContent: 'space-between',
          padding: '2px 6px 6px', marginTop: 12,
        }}
      >
        <span style={{ fontFamily: 'monospace', fontSize: 11, color: 'var(--text-6)' }}>DATA</span>
        <div style={{ display: 'flex', gap: 4 }}>
          <button onClick={() => addDataNode('dataset')} title="Add a dataset" style={addBtn}>＋ set</button>
          <button onClick={() => addDataNode('noise')} title="Add a noise source" style={addBtn}>＋ noise</button>
        </div>
      </div>
      {dataNodes.map((d) => (
        <div key={d.id} style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
          <div
            style={{
              flex: 1, padding: '5px 8px', borderRadius: 5, fontFamily: 'monospace', fontSize: 13,
              color: d.kind === 'noise' ? 'var(--warn)' : 'var(--accent-2)',
              overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
            }}
          >
            {d.name}
          </div>
          <button
            onClick={() => removeDataNode(d.id)}
            title={`Delete ${d.name}`}
            style={{
              background: 'none', color: 'var(--text-6)', border: 'none', cursor: 'pointer',
              fontFamily: 'monospace', fontSize: 13, padding: '2px 4px',
            }}
          >
            ✕
          </button>
        </div>
      ))}
    </div>
  )
}

export function SystemView({ registry, onModelMove }: SystemViewProps) {
  const selectedDataNodeId = useGraphStore((s) => s.selectedDataNodeId)
  const dataNodes = useGraphStore((s) => s.dataNodes)
  const selected = dataNodes.find((d) => d.id === selectedDataNodeId)
  return (
    <div style={{ display: 'flex', flex: 1, overflow: 'hidden' }}>
      <Sidebar registry={registry} />
      <ReactFlowProvider>
        <div style={{ flex: 1, height: '100%' }}>
          <SystemCanvas onModelMove={onModelMove} />
        </div>
      </ReactFlowProvider>
      {selected && <DataNodeInspector node={selected} />}
    </div>
  )
}
