import { create } from 'zustand'
import {
  addEdge,
  applyEdgeChanges,
  applyNodeChanges,
  type Connection,
  type Edge,
  type EdgeChange,
  type Node,
  type NodeChange,
} from '@xyflow/react'
import type { DomainGraph, NodeDef } from '../types/graph'

export interface ModelNodeData extends Record<string, unknown> {
  nodeType: string
  label: string
  color: string
  inputPins: Array<{ name: string; label: string }>
  outputPins: Array<{ name: string; label: string }>
  params: Record<string, unknown>
}

export type ModelNode = Node<ModelNodeData>

interface GraphState {
  nodes: ModelNode[]
  edges: Edge[]
  onNodesChange: (changes: NodeChange<ModelNode>[]) => void
  onEdgesChange: (changes: EdgeChange[]) => void
  onConnect: (connection: Connection) => void

  selectedNodeId: string | null
  setSelectedNode: (id: string | null) => void

  addNode: (nodeDef: NodeDef, position: { x: number; y: number }) => void
  updateNodeParam: (nodeId: string, key: string, value: unknown) => void
  loadGraph: (domain: DomainGraph, registry: Record<string, NodeDef>) => void

  shapes: Record<string, number[]>
  errors: Record<string, string>
  setValidationResult: (shapes: Record<string, number[]>, errors: Record<string, string>) => void

  toDomainGraph: () => DomainGraph
}

export const useGraphStore = create<GraphState>((set, get) => ({
  nodes: [],
  edges: [],

  onNodesChange: (changes) =>
    set((s) => ({ nodes: applyNodeChanges(changes, s.nodes) })),

  onEdgesChange: (changes) =>
    set((s) => ({ edges: applyEdgeChanges(changes, s.edges) })),

  onConnect: (conn) =>
    set((s) => {
      // A target input handle accepts a single edge — a new connection
      // replaces any existing one rather than silently fanning in.
      const cleared = s.edges.filter(
        (e) => !(e.target === conn.target && e.targetHandle === conn.targetHandle)
      )
      return { edges: addEdge(conn, cleared) }
    }),

  selectedNodeId: null,
  setSelectedNode: (id) => set({ selectedNodeId: id }),

  addNode: (nodeDef, position) => {
    const id = crypto.randomUUID()
    const defaultParams = Object.fromEntries(
      nodeDef.params.map((p) => [p.name, p.default])
    )
    const node: ModelNode = {
      id,
      type: 'modelNode',
      position,
      data: {
        nodeType: nodeDef.type,
        label: nodeDef.label,
        color: nodeDef.color,
        inputPins: nodeDef.inputs,
        outputPins: nodeDef.outputs,
        params: defaultParams,
      },
    }
    set((s) => ({ nodes: [...s.nodes, node] }))
  },

  updateNodeParam: (nodeId, key, value) =>
    set((s) => ({
      nodes: s.nodes.map((n) =>
        n.id === nodeId
          ? { ...n, data: { ...n.data, params: { ...n.data.params, [key]: value } } }
          : n
      ),
    })),

  loadGraph: (domain, registry) => {
    const nodes: ModelNode[] = domain.nodes.map((dn) => {
      const def = registry[dn.type]
      return {
        id: dn.id,
        type: 'modelNode',
        position: dn.position,
        data: {
          nodeType: dn.type,
          label: def?.label ?? dn.type,
          color: def?.color ?? '#888888',
          inputPins: def?.inputs ?? [],
          outputPins: def?.outputs ?? [],
          params: dn.params,
        },
      }
    })
    const edges: Edge[] = domain.edges.map((de) => ({
      id: de.id,
      source: de.source,
      sourceHandle: de.sourceHandle,
      target: de.target,
      targetHandle: de.targetHandle,
    }))
    set({ nodes, edges, selectedNodeId: null })
  },

  shapes: {},
  errors: {},
  setValidationResult: (shapes, errors) => set({ shapes, errors }),

  toDomainGraph: () => {
    const { nodes, edges } = get()
    return {
      nodes: nodes.map((n) => ({
        id: n.id,
        type: n.data.nodeType,
        position: n.position,
        params: n.data.params,
      })),
      edges: edges.map((e) => ({
        id: e.id,
        source: e.source,
        sourceHandle: e.sourceHandle ?? 'output',
        target: e.target,
        targetHandle: e.targetHandle ?? 'input',
      })),
    }
  },
}))
