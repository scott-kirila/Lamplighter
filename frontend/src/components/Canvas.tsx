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
import ModelNode from './nodes/ModelNode'
import type { NodeDef } from '../types/graph'

const nodeTypes: NodeTypes = { modelNode: ModelNode }

function DropCanvas({ registry }: { registry: Record<string, NodeDef> }) {
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
      const nodeType = e.dataTransfer.getData('application/scorch-node')
      const nodeDef = registry[nodeType]
      if (!nodeDef) return
      const position = screenToFlowPosition({ x: e.clientX, y: e.clientY })
      addNode(nodeDef, position)
    },
    [registry, addNode, screenToFlowPosition]
  )

  return (
    <ReactFlow
      nodes={nodes}
      edges={edges}
      onNodesChange={onNodesChange}
      onEdgesChange={onEdgesChange}
      onConnect={onConnect}
      onSelectionChange={onSelectionChange}
      onDragOver={onDragOver}
      onDrop={onDrop}
      nodeTypes={nodeTypes}
      fitView
      style={{ background: '#0d0d1a' }}
    >
      <Background color="#1e1e30" gap={24} size={1} />
      <Controls style={{ background: '#1e1e2e', border: '1px solid #2a2a4a' }} />
    </ReactFlow>
  )
}

export function Canvas({ registry }: { registry: Record<string, NodeDef> }) {
  return (
    <ReactFlowProvider>
      <div style={{ flex: 1, height: '100%' }}>
        <DropCanvas registry={registry} />
      </div>
    </ReactFlowProvider>
  )
}
