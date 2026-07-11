import { useCallback, useMemo } from 'react'
import {
  Background,
  Controls,
  MiniMap,
  ReactFlow,
  ReactFlowProvider,
  useReactFlow,
  type NodeTypes,
  type OnSelectionChangeParams,
} from '@xyflow/react'
import '@xyflow/react/dist/style.css'
import { useGraphStore } from '../store/graphStore'
import type { ModelNode as ModelNodeType } from '../store/graphStore'
import ModelNode from './nodes/ModelNode'
import type { NodeDef, NodeMove } from '../types/graph'

const nodeTypes: NodeTypes = { modelNode: ModelNode }

// How close (screen px) the test point must come to a wire to splice into it.
// A forgiving radius so "near/over the wire" activates, rather than requiring a
// pixel-perfect hit on the thin edge.
const SPLICE_RADIUS = 44

// The id of the edge whose rendered path comes closest to a screen point, if any
// is within `threshold` px. Measures actual path geometry (sampled via the SVG
// path) rather than hit-testing a single pixel, so it doesn't depend on the thin
// interaction band or on what's painted on top.
function nearestEdgeId(x: number, y: number, threshold: number): string | null {
  let bestId: string | null = null
  let bestDist = threshold
  for (const edgeEl of document.querySelectorAll('.react-flow__edge')) {
    const path = edgeEl.querySelector<SVGPathElement>('.react-flow__edge-path')
    const ctm = path?.getScreenCTM()
    if (!path || !ctm) continue
    const len = path.getTotalLength()
    if (!len) continue
    const steps = Math.min(60, Math.max(4, Math.round(len / 10)))
    let minDist = Infinity
    for (let i = 0; i <= steps; i++) {
      const p = path.getPointAtLength((len * i) / steps).matrixTransform(ctm)
      const d = Math.hypot(p.x - x, p.y - y)
      if (d < minDist) minDist = d
    }
    if (minDist < bestDist) {
      bestDist = minDist
      bestId = edgeEl.getAttribute('data-id')
    }
  }
  return bestId
}

interface CanvasProps {
  registry: Record<string, NodeDef>
  onNodeMove?: (moves: NodeMove[]) => void
}

function DropCanvas({ registry, onNodeMove }: CanvasProps) {
  const { screenToFlowPosition, flowToScreenPosition } = useReactFlow()
  const nodes = useGraphStore((s) => s.nodes)
  const edges = useGraphStore((s) => s.edges)
  const onNodesChange = useGraphStore((s) => s.onNodesChange)
  const onEdgesChange = useGraphStore((s) => s.onEdgesChange)
  const onConnect = useGraphStore((s) => s.onConnect)
  const addNode = useGraphStore((s) => s.addNode)
  const insertNodeOnEdge = useGraphStore((s) => s.insertNodeOnEdge)
  const spliceNodeIntoEdge = useGraphStore((s) => s.spliceNodeIntoEdge)
  const setSelectedNode = useGraphStore((s) => s.setSelectedNode)
  const spliceTargetId = useGraphStore((s) => s.spliceTargetId)
  const setSpliceTarget = useGraphStore((s) => s.setSpliceTarget)

  const onSelectionChange = useCallback(
    ({ nodes }: OnSelectionChangeParams) => {
      setSelectedNode(nodes.length === 1 ? nodes[0].id : null)
    },
    [setSelectedNode]
  )

  // The edge an unconnected, splice-capable node is currently centered over —
  // shared by the drag highlight and the commit on drag-stop.
  const spliceTargetForNode = useCallback(
    (n: ModelNodeType): string | null => {
      const def = registry[n.data.nodeType]
      if (!def || def.inputs.length === 0 || def.outputs.length === 0) return null
      const connected = useGraphStore
        .getState()
        .edges.some((e) => e.source === n.id || e.target === n.id)
      if (connected) return null
      const w = n.measured?.width ?? 0
      const h = n.measured?.height ?? 0
      const c = flowToScreenPosition({ x: n.position.x + w / 2, y: n.position.y + h / 2 })
      // Reach to the node's own extent so a wire passing under the node body
      // activates, even when its center isn't right on the line.
      return nearestEdgeId(c.x, c.y, Math.max(SPLICE_RADIUS, h / 2 + 8))
    },
    [registry, flowToScreenPosition]
  )

  const onDragOver = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault()
      e.dataTransfer.dropEffect = 'move'
      // Highlight the edge under the cursor while dragging a splice-capable
      // palette node. dataTransfer is unreadable mid-drag, so eligibility comes
      // from the node type stashed at drag start.
      const type = useGraphStore.getState().paletteDragType
      const def = type ? registry[type] : undefined
      const eligible = !!def && def.inputs.length > 0 && def.outputs.length > 0
      setSpliceTarget(eligible ? nearestEdgeId(e.clientX, e.clientY, SPLICE_RADIUS) : null)
    },
    [registry, setSpliceTarget]
  )

  const onDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault()
      setSpliceTarget(null)
      const nodeType = e.dataTransfer.getData('application/lamplighter-node')
      const nodeDef = registry[nodeType]
      if (!nodeDef) return

      // The cursor sits at the centered drag image's middle; offset back to the
      // node's top-left for placement, and hit-test the cursor as the center.
      let off = { x: 0, y: 0 }
      try {
        const raw = e.dataTransfer.getData('application/lamplighter-offset')
        if (raw) off = JSON.parse(raw)
      } catch {
        // missing/malformed offset — fall back to placing at the cursor
      }
      const position = screenToFlowPosition({ x: e.clientX - off.x, y: e.clientY - off.y })

      // If dropped onto an existing edge, splice the node into it. Only nodes
      // with both an input and an output can be spliced (Input/Output can't).
      const canSplice = nodeDef.inputs.length > 0 && nodeDef.outputs.length > 0
      if (canSplice) {
        const edgeId = nearestEdgeId(e.clientX, e.clientY, SPLICE_RADIUS)
        if (edgeId) {
          insertNodeOnEdge(nodeDef, position, edgeId)
          return
        }
      }
      addNode(nodeDef, position)
    },
    [registry, addNode, insertNodeOnEdge, screenToFlowPosition, setSpliceTarget]
  )

  // Highlight the wire a dragged node would splice into, live as it moves.
  const onNodeDrag = useCallback(
    (_e: MouseEvent | TouchEvent, _node: ModelNodeType, dragged: ModelNodeType[]) => {
      setSpliceTarget(dragged.length === 1 ? spliceTargetForNode(dragged[0]) : null)
    },
    [spliceTargetForNode, setSpliceTarget]
  )

  const onNodeDragStop = useCallback(
    (_e: MouseEvent | TouchEvent, _node: ModelNodeType, dragged: ModelNodeType[]) => {
      setSpliceTarget(null)
      // Dropping a single, unconnected, splice-capable node onto an edge inserts
      // it there — the on-canvas equivalent of dropping a palette node on a wire.
      if (dragged.length === 1) {
        const target = spliceTargetForNode(dragged[0])
        if (target) {
          spliceNodeIntoEdge(dragged[0].id, target)
          return
        }
      }
      // Otherwise just persist the final positions (covers multi-select drags).
      onNodeMove?.(dragged.map((n) => ({ id: n.id, position: n.position })))
    },
    [spliceTargetForNode, spliceNodeIntoEdge, setSpliceTarget, onNodeMove]
  )

  // Restyle the splice-target edge so it reads as the active drop target.
  const styledEdges = useMemo(() => {
    if (!spliceTargetId) return edges
    return edges.map((e) =>
      e.id === spliceTargetId
        ? { ...e, animated: true, style: { ...e.style, stroke: 'var(--accent-2)', strokeWidth: 3 } }
        : e
    )
  }, [edges, spliceTargetId])

  return (
    <ReactFlow
      nodes={nodes}
      edges={styledEdges}
      onNodesChange={onNodesChange}
      onEdgesChange={onEdgesChange}
      onConnect={onConnect}
      onSelectionChange={onSelectionChange}
      onNodeDrag={onNodeDrag}
      onNodeDragStop={onNodeDragStop}
      onDragOver={onDragOver}
      onDrop={onDrop}
      nodeTypes={nodeTypes}
      deleteKeyCode={['Backspace', 'Delete']}
      fitView
      style={{ background: 'var(--bg)' }}
    >
      <Background color="var(--canvas-dots)" gap={24} size={1} />
      <Controls style={{ background: 'var(--surface)', border: '1px solid var(--border)' }} />
      <MiniMap
        pannable
        zoomable
        nodeColor={(n) => String(n.data?.color ?? 'var(--text-6)')}
        nodeStrokeColor="var(--border)"
        maskColor="var(--overlay)"
        style={{ background: 'var(--panel)', border: '1px solid var(--border)' }}
      />
    </ReactFlow>
  )
}

export function Canvas({ registry, onNodeMove }: CanvasProps) {
  return (
    <ReactFlowProvider>
      <div style={{ flex: 1, height: '100%' }}>
        <DropCanvas registry={registry} onNodeMove={onNodeMove} />
      </div>
    </ReactFlowProvider>
  )
}
