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
import OverviewModelNode, { type OverviewModelData, type OverviewModelPort } from './nodes/OverviewModelNode'
import OverviewDataNode, { type OverviewDataData } from './nodes/OverviewDataNode'
import { DataNodeInspector } from './DataNodeInspector'
import { ModelInspector } from './ModelInspector'
import { useModelDeleteConfirm } from '../hooks/useModelDeleteConfirm'

const nodeTypes: NodeTypes = { overviewModel: OverviewModelNode, overviewData: OverviewDataNode }

interface OverviewViewProps {
  registry: Record<string, NodeDef>
  onModelMove?: (moves: NodeMove[]) => void
}

// The high-level view: every model as a node you can arrange and open, plus a
// sidebar to add, jump between, rename, and remove them. Single-model projects
// show one model here (and land on its canvas by default) — the overview is
// the hub the multi-model workflows (GAN, …) build on.
function OverviewCanvas({ onModelMove }: { onModelMove?: (moves: NodeMove[]) => void }) {
  const models = useGraphStore((s) => s.models)
  const activeModelId = useGraphStore((s) => s.activeModelId)
  const activeNodes = useGraphStore((s) => s.nodes)
  const activeCount = activeNodes.length
  const modelGraphs = useGraphStore((s) => s.modelGraphs)
  const openModel = useGraphStore((s) => s.openModel)
  const setModelSysPosition = useGraphStore((s) => s.setModelSysPosition)
  const dataNodes = useGraphStore((s) => s.dataNodes)
  const setDataNodeSysPosition = useGraphStore((s) => s.setDataNodeSysPosition)
  const setSelectedDataNode = useGraphStore((s) => s.setSelectedDataNode)
  const setSelectedOverviewModel = useGraphStore((s) => s.setSelectedOverviewModel)
  const selectedDataNodeId = useGraphStore((s) => s.selectedDataNodeId)
  const selectedOverviewModelId = useGraphStore((s) => s.selectedOverviewModelId)
  const links = useGraphStore((s) => s.links)
  const linkResults = useGraphStore((s) => s.linkResults)
  const addLink = useGraphStore((s) => s.addLink)
  const removeLink = useGraphStore((s) => s.removeLink)
  const removeDataNode = useGraphStore((s) => s.removeDataNode)
  const deleteModel = useGraphStore((s) => s.deleteModel)
  const { requestDelete, modal: deleteModal } = useModelDeleteConfirm(deleteModel)
  // The selected node (model or data) — drives the selection ring and Delete-key
  // removal. Derived from the store's two (mutually exclusive) selection fields,
  // so a click on the canvas, the sidebar, or the info pane all stay in sync.
  const selectedId = selectedOverviewModelId ?? selectedDataNodeId
  // The selected link (edge). Edges are derived from `links`, so React Flow has
  // no change handler to track their selection — we track it ourselves: clicking
  // a link selects it for the highlight + Delete, and clears any node selection.
  const [selectedLinkId, setSelectedLinkId] = useState<string | null>(null)

  const nodeCountFor = useCallback(
    (id: string) => (id === activeModelId ? activeCount : modelGraphs[id]?.nodes.length ?? 0),
    [activeModelId, activeCount, modelGraphs]
  )
  const isDataNode = useCallback((id: string) => dataNodes.some((d) => d.id === id), [dataNodes])
  // A model's Input nodes as named ports, ordered to match forward()'s args
  // (top-to-bottom by canvas position). A data node fans out to these ports.
  const inputsFor = useCallback(
    (id: string): OverviewModelPort[] =>
      (id === activeModelId ? activeNodes : modelGraphs[id]?.nodes ?? [])
        .filter((n) => n.data.nodeType === 'Input')
        .slice()
        .sort((a, b) => a.position.y - b.position.y || a.position.x - b.position.x || a.id.localeCompare(b.id))
        .map((n, i) => ({ id: n.id, name: String(n.data.params.name ?? '').trim() || `in ${i}` })),
    [activeModelId, activeNodes, modelGraphs]
  )
  // Datasets whose label (y) pin is wired → they render split x/y output pins.
  const labeledDatasetIds = useMemo(
    () => new Set(links.filter((l) => l.source_pin === 'y' && l.source_data).map((l) => l.source_data as string)),
    [links]
  )

  const nodes: Node<OverviewModelData | OverviewDataData>[] = useMemo(
    () => [
      ...models.map((m) => {
        const count = nodeCountFor(m.id)
        return {
          id: m.id,
          type: 'overviewModel',
          position: m.sysPosition,
          data: {
            name: m.name,
            subtitle: `${count} node${count === 1 ? '' : 's'}`,
            selected: m.id === selectedId,
            inputs: inputsFor(m.id),
          },
        }
      }),
      ...dataNodes.map((d) => ({
        id: d.id,
        type: 'overviewData',
        position: d.sysPosition,
        data: { name: d.name, kind: d.kind, labeled: labeledDatasetIds.has(d.id), selected: d.id === selectedId },
      })),
    ],
    [models, selectedId, nodeCountFor, dataNodes, inputsFor, labeledDatasetIds]
  )

  // Links → styled edges: the backend's shape-check drives the color (accent
  // when compatible, error when not). Only a *failing* link is labelled — its
  // mismatch text is the useful bit; a healthy link's "src → tgt" is redundant
  // with the wire itself.
  const edges: Edge[] = useMemo(
    () =>
      links.map((l) => {
        const res = linkResults[l.id]
        const ok = res?.ok ?? true
        const selected = l.id === selectedLinkId
        return {
          id: l.id,
          // The edge's source node is a data node (data→model) or a model
          // (model→model) — whichever this link carries.
          source: l.source_data ?? l.source_model ?? '',
          target: l.target_model,
          // Leave from the named source pin (a labeled dataset's x/y) and land on
          // the named input port when the link specifies one, else single handles.
          sourceHandle: l.source_pin ?? (l.source_data && labeledDatasetIds.has(l.source_data) ? 'x' : undefined),
          targetHandle: l.target_input ?? undefined,
          animated: true,
          label: ok ? undefined : res?.message,
          labelStyle: { fill: 'var(--error)', fontFamily: 'monospace', fontSize: 11 },
          labelBgStyle: { fill: 'var(--panel)', fillOpacity: 0.9 },
          labelBgPadding: [6, 3] as [number, number],
          labelBgBorderRadius: 4,
          style: {
            // Selected: our edge-highlight (a failing link keeps its error red so
            // it still reads as broken), plus a thicker line.
            stroke: selected ? (ok ? 'var(--edge-highlight)' : 'var(--error-bright)') : ok ? 'var(--accent-2)' : 'var(--error-bright)',
            strokeWidth: selected ? 3.5 : 2,
          },
        }
      }),
    [links, linkResults, labeledDatasetIds, selectedLinkId]
  )

  // Apply live position changes to the model's sys_position on every drag tick,
  // so a model node follows the cursor (same as the model canvas) rather than
  // jumping only on drop. Non-position changes (selection/dimensions) are
  // React Flow's to track internally.
  const onNodesChange = useCallback(
    (changes: NodeChange<Node<OverviewModelData | OverviewDataData>>[]) => {
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
      // c.targetHandle is the Input node id when the wire lands on a named port
      // (multi-input model), else null → the sole input. c.sourceHandle is a
      // labeled dataset's pin ('x'/'y'), else null.
      if (c.source && c.target) addLink(c.source, c.target, c.targetHandle ?? null, c.sourceHandle ?? null)
    },
    [addLink]
  )
  // Delete a canvas node: data/noise nodes go freely; a model confirms first (and
  // never the last one).
  const deleteNode = useCallback(
    (id: string) => {
      if (isDataNode(id)) {
        removeDataNode(id)
      } else if (models.length > 1) {
        const model = models.find((m) => m.id === id)
        if (model) requestDelete(model.id, model.name)
      }
    },
    [isDataNode, removeDataNode, requestDelete, models]
  )
  // Delete/Backspace removes the selected node (unless a text field has focus, so
  // editing the Inspector never nukes the node). Edge deletion stays React Flow's.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key !== 'Delete' && e.key !== 'Backspace') return
      const t = e.target as HTMLElement | null
      if (t && (t.tagName === 'INPUT' || t.tagName === 'TEXTAREA' || t.isContentEditable)) return
      if (selectedLinkId) {
        removeLink(selectedLinkId)
        setSelectedLinkId(null)
      } else if (selectedId) {
        deleteNode(selectedId)
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [selectedId, selectedLinkId, deleteNode, removeLink])
  // Clicking a node selects it (for Delete) and opens the matching right-hand
  // pane — the data node's Inspector or the model's info pane (mutually
  // exclusive). Clicking the empty pane clears both.
  const onNodeClick = useCallback(
    (_e: React.MouseEvent, node: Node) => {
      const data = isDataNode(node.id)
      setSelectedDataNode(data ? node.id : null)
      setSelectedOverviewModel(data ? null : node.id)
      setSelectedLinkId(null)
    },
    [setSelectedDataNode, setSelectedOverviewModel, isDataNode]
  )
  const onPaneClick = useCallback(() => {
    setSelectedDataNode(null)
    setSelectedOverviewModel(null)
    setSelectedLinkId(null)
  }, [setSelectedDataNode, setSelectedOverviewModel])
  // Clicking a link selects it (for the highlight + Delete); clear any node
  // selection so the two stay mutually exclusive.
  const onEdgeClick = useCallback(
    (_e: React.MouseEvent, edge: Edge) => {
      setSelectedLinkId(edge.id)
      setSelectedDataNode(null)
      setSelectedOverviewModel(null)
    },
    [setSelectedDataNode, setSelectedOverviewModel]
  )

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
      onEdgeClick={onEdgeClick}
      onPaneClick={onPaneClick}
      onNodeDoubleClick={onNodeDoubleClick}
      onNodeDragStop={onNodeDragStop}
      onConnect={onConnect}
      // Deletion (nodes and links) is handled by the keydown effect above, the
      // single path — so React Flow's own delete keybinding stays off.
      deleteKeyCode={null}
      // Attribution moved to a header credit (see Toolbar); hide the canvas badge.
      proOptions={{ hideAttribution: true }}
      fitView
      style={{ background: 'var(--bg)' }}
    >
      <Background color="var(--canvas-dots)" gap={24} size={1} />
      <Controls style={{ background: 'var(--surface)', border: '1px solid var(--border)' }} />
      {deleteModal}
    </ReactFlow>
  )
}

function Sidebar({ registry }: { registry: Record<string, NodeDef> }) {
  const models = useGraphStore((s) => s.models)
  const selectedModelId = useGraphStore((s) => s.selectedOverviewModelId)
  const setSelectedModel = useGraphStore((s) => s.setSelectedOverviewModel)
  const setSelectedDataNode = useGraphStore((s) => s.setSelectedDataNode)
  const openModel = useGraphStore((s) => s.openModel)
  const renameModel = useGraphStore((s) => s.renameModel)
  const addModel = useGraphStore((s) => s.addModel)
  const deleteModel = useGraphStore((s) => s.deleteModel)
  const { requestDelete, modal: deleteModal } = useModelDeleteConfirm(deleteModel)
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
                onClick={() => {
                  setSelectedModel(m.id)
                  setSelectedDataNode(null)
                }}
                onDoubleClick={() => setEditing(m.id)}
                title="Click to select · double-click to rename"
                style={{
                  flex: 1,
                  textAlign: 'left',
                  background: m.id === selectedModelId ? 'var(--surface)' : 'none',
                  color: m.id === selectedModelId ? 'var(--text)' : 'var(--text-4)',
                  border: `1px solid ${m.id === selectedModelId ? 'var(--accent)' : 'transparent'}`,
                  borderRadius: 5,
                  padding: '5px 8px',
                  fontFamily: 'monospace',
                  fontSize: 13,
                  cursor: 'pointer',
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
                  onClick={() => requestDelete(m.id, m.name)}
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
          <button onClick={() => addDataNode('env')} title="Add a Gymnasium environment (RL)" style={addBtn}>＋ env</button>
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
      {deleteModal}
    </div>
  )
}

export function OverviewView({ registry, onModelMove }: OverviewViewProps) {
  const selectedDataNodeId = useGraphStore((s) => s.selectedDataNodeId)
  const selectedOverviewModelId = useGraphStore((s) => s.selectedOverviewModelId)
  const dataNodes = useGraphStore((s) => s.dataNodes)
  const selected = dataNodes.find((d) => d.id === selectedDataNodeId)
  return (
    <div style={{ display: 'flex', flex: 1, overflow: 'hidden' }}>
      <Sidebar registry={registry} />
      <ReactFlowProvider>
        <div style={{ flex: 1, height: '100%' }}>
          <OverviewCanvas onModelMove={onModelMove} />
        </div>
      </ReactFlowProvider>
      {/* The right pane: a data node's Inspector, else the selected model's info
          pane (the two selections are mutually exclusive). */}
      {selected ? (
        <DataNodeInspector node={selected} />
      ) : selectedOverviewModelId ? (
        <ModelInspector modelId={selectedOverviewModelId} />
      ) : null}
    </div>
  )
}
