import { useCallback } from 'react'
import {
  Background,
  Controls,
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

interface CanvasProps {
  registry: Record<string, NodeDef>
  onNodeMove?: (moves: NodeMove[]) => void
}

function DropCanvas({ registry, onNodeMove }: CanvasProps) {
  const { screenToFlowPosition } = useReactFlow()
  const nodes = useGraphStore((s) => s.nodes)
  const edges = useGraphStore((s) => s.edges)
  const onNodesChange = useGraphStore((s) => s.onNodesChange)
  const onEdgesChange = useGraphStore((s) => s.onEdgesChange)
  const onConnect = useGraphStore((s) => s.onConnect)
  const addNode = useGraphStore((s) => s.addNode)
  const setSelectedNode = useGraphStore((s) => s.setSelectedNode)

  const onSelectionChange = useCallback(
    ({ nodes }: OnSelectionChangeParams) => {
      setSelectedNode(nodes.length === 1 ? nodes[0].id : null)
    },
    [setSelectedNode]
  )

  const onDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    e.dataTransfer.dropEffect = 'move'
  }, [])

  const onDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault()
      const nodeType = e.dataTransfer.getData('application/lamplighter-node')
      const nodeDef = registry[nodeType]
      if (!nodeDef) return
      const position = screenToFlowPosition({ x: e.clientX, y: e.clientY })
      addNode(nodeDef, position)
    },
    [registry, addNode, screenToFlowPosition]
  )

  // Broadcast final positions once a drag ends (covers multi-select drags).
  const onNodeDragStop = useCallback(
    (_e: MouseEvent | TouchEvent, _node: ModelNodeType, dragged: ModelNodeType[]) => {
      onNodeMove?.(dragged.map((n) => ({ id: n.id, position: n.position })))
    },
    [onNodeMove]
  )

  return (
    <ReactFlow
      nodes={nodes}
      edges={edges}
      onNodesChange={onNodesChange}
      onEdgesChange={onEdgesChange}
      onConnect={onConnect}
      onSelectionChange={onSelectionChange}
      onNodeDragStop={onNodeDragStop}
      onDragOver={onDragOver}
      onDrop={onDrop}
      nodeTypes={nodeTypes}
      fitView
      style={{ background: 'var(--bg)' }}
    >
      <Background color="var(--canvas-dots)" gap={24} size={1} />
      <Controls style={{ background: 'var(--surface)', border: '1px solid var(--border)' }} />
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
