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
import type { DomainGraph, NodeDef, NodeMove } from '../types/graph'
import { nodeColor } from '../lib/nodeColor'

export interface ModelNodeData extends Record<string, unknown> {
  nodeType: string
  label: string
  color: string
  inputPins: Array<{ name: string; label: string }>
  outputPins: Array<{ name: string; label: string }>
  params: Record<string, unknown>
}

export type ModelNode = Node<ModelNodeData>

// One epoch of a streamed in-kernel training run.
export interface RunEpoch {
  epoch: number
  epochs: number
  metrics: Record<string, number>
}

// Rewire edge A→B into A→N→B, splicing node N (via the given handles) in place
// of the original edge. Returns the new edge list.
function splicedEdges(
  edges: Edge[],
  edge: Edge,
  nodeId: string,
  inHandle: string,
  outHandle: string
): Edge[] {
  return edges
    .filter((e) => e.id !== edge.id)
    .concat(
      {
        id: crypto.randomUUID(),
        source: edge.source,
        sourceHandle: edge.sourceHandle,
        target: nodeId,
        targetHandle: inHandle,
      },
      {
        id: crypto.randomUUID(),
        source: nodeId,
        sourceHandle: outHandle,
        target: edge.target,
        targetHandle: edge.targetHandle,
      }
    )
}

// Build a canvas node from a registry definition, seeded with default params.
function buildNode(nodeDef: NodeDef, position: { x: number; y: number }): ModelNode {
  return {
    id: crypto.randomUUID(),
    type: 'modelNode',
    position,
    data: {
      nodeType: nodeDef.type,
      label: nodeDef.label,
      color: nodeColor(nodeDef.category, nodeDef.type),
      inputPins: nodeDef.inputs,
      outputPins: nodeDef.outputs,
      params: Object.fromEntries(nodeDef.params.map((p) => [p.name, p.default])),
    },
  }
}

interface GraphState {
  nodes: ModelNode[]
  edges: Edge[]
  onNodesChange: (changes: NodeChange<ModelNode>[]) => void
  onEdgesChange: (changes: EdgeChange[]) => void
  onConnect: (connection: Connection) => void

  selectedNodeId: string | null
  setSelectedNode: (id: string | null) => void

  addNode: (nodeDef: NodeDef, position: { x: number; y: number }) => void
  insertNodeOnEdge: (
    nodeDef: NodeDef,
    position: { x: number; y: number },
    edgeId: string
  ) => void
  spliceNodeIntoEdge: (nodeId: string, edgeId: string) => void
  updateNodeParam: (nodeId: string, key: string, value: unknown) => void

  // Which top-level tab is active (model canvas vs data / training config).
  activeTab: 'model' | 'data' | 'training'
  setActiveTab: (tab: 'model' | 'data' | 'training') => void

  // Graph-global training config (loss/optimizer/hyperparams). Rides the design.
  training: Record<string, unknown>
  setTrainingParam: (key: string, value: unknown) => void

  // Data-pipeline config (source, batching) driving the Data panel. Rides the design.
  data: Record<string, unknown>
  setDataParam: (key: string, value: unknown) => void

  // Transient drag state for the drop-to-insert highlight: the edge a splice
  // would land on, and the node type being dragged from the palette (so a
  // dragover — where dataTransfer is unreadable — can still check eligibility).
  spliceTargetId: string | null
  setSpliceTarget: (edgeId: string | null) => void
  paletteDragType: string | null
  setPaletteDragType: (nodeType: string | null) => void
  loadGraph: (domain: DomainGraph, registry: Record<string, NodeDef>) => void
  seedDefault: (registry: Record<string, NodeDef>) => void
  setNodePositions: (moves: NodeMove[]) => void

  // In-kernel training run (triggered from the Training tab, streamed over WS).
  runState: 'idle' | 'running' | 'done' | 'stopped' | 'failed'
  runEpochs: RunEpoch[]
  runError: string | null
  setRunStatus: (state: GraphState['runState'], error: string | null) => void
  appendRunEpoch: (epoch: RunEpoch) => void

  shapes: Record<string, number[]>
  // Per-output-pin shapes ({ nodeId: { pin: dims } }) — powers the Inspector's
  // per-pin readout for multi-output nodes (LSTM's output / h_n / c_n).
  pinShapes: Record<string, Record<string, number[]>>
  errors: Record<string, string>
  graphIssues: string[]
  code: string | null
  setValidationResult: (
    shapes: Record<string, number[]>,
    pinShapes: Record<string, Record<string, number[]>>,
    errors: Record<string, string>,
    graphIssues: string[],
    code: string | null
  ) => void
  setCode: (code: string | null) => void

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

  addNode: (nodeDef, position) =>
    set((s) => ({ nodes: [...s.nodes, buildNode(nodeDef, position)] })),

  // Splice a node into an existing edge A→B: drop the original edge and rewire
  // A→N→B through the new node's first input/output handles. Caller ensures the
  // node has both (Input/Output can't be spliced). Falls back to a plain add if
  // the edge has since vanished.
  insertNodeOnEdge: (nodeDef, position, edgeId) =>
    set((s) => {
      const node = buildNode(nodeDef, position)
      const edge = s.edges.find((e) => e.id === edgeId)
      if (!edge) return { nodes: [...s.nodes, node] }
      return {
        nodes: [...s.nodes, node],
        edges: splicedEdges(
          s.edges,
          edge,
          node.id,
          nodeDef.inputs[0]?.name ?? 'input',
          nodeDef.outputs[0]?.name ?? 'output'
        ),
      }
    }),

  // Splice an existing (unconnected) node into an edge — same rewiring as
  // insertNodeOnEdge, but for a node already on the canvas. The drag handler
  // gates this to unconnected, splice-capable nodes.
  spliceNodeIntoEdge: (nodeId, edgeId) =>
    set((s) => {
      const node = s.nodes.find((n) => n.id === nodeId)
      const edge = s.edges.find((e) => e.id === edgeId)
      if (!node || !edge) return {}
      return {
        edges: splicedEdges(
          s.edges,
          edge,
          nodeId,
          node.data.inputPins[0]?.name ?? 'input',
          node.data.outputPins[0]?.name ?? 'output'
        ),
      }
    }),

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
          color: nodeColor(def?.category, dn.type),
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
    set({
      nodes,
      edges,
      selectedNodeId: null,
      training: domain.training ?? {},
      data: domain.data ?? {},
    })
  },

  // Seed a fresh canvas with an Input → Output scaffold (unconnected, so adding
  // a layer between them needs no edge deletion). Ordinary deletable nodes —
  // correctness is enforced by validation, not by locking these in place.
  seedDefault: (registry) => {
    const make = (type: string, position: { x: number; y: number }): ModelNode | null => {
      const def = registry[type]
      if (!def) return null
      return {
        id: crypto.randomUUID(),
        type: 'modelNode',
        position,
        data: {
          nodeType: def.type,
          label: def.label,
          color: nodeColor(def.category, def.type),
          inputPins: def.inputs,
          outputPins: def.outputs,
          params: Object.fromEntries(def.params.map((p) => [p.name, p.default])),
        },
      }
    }
    const seeded = [make('Input', { x: 80, y: 200 }), make('Output', { x: 520, y: 200 })]
    set({ nodes: seeded.filter((n): n is ModelNode => n !== null), edges: [], selectedNodeId: null })
  },

  setNodePositions: (moves) =>
    set((s) => {
      const byId = new Map(moves.map((m) => [m.id, m.position]))
      return {
        nodes: s.nodes.map((n) => {
          const pos = byId.get(n.id)
          return pos ? { ...n, position: pos } : n
        }),
      }
    }),

  spliceTargetId: null,
  // No-op when unchanged so the frequent dragover/drag updates don't re-render.
  setSpliceTarget: (edgeId) =>
    set((s) => (s.spliceTargetId === edgeId ? {} : { spliceTargetId: edgeId })),
  paletteDragType: null,
  setPaletteDragType: (nodeType) => set({ paletteDragType: nodeType }),

  runState: 'idle',
  runEpochs: [],
  runError: null,
  // Entering "running" clears the previous run's lines so the panel starts fresh.
  setRunStatus: (state, error) =>
    set((s) => ({
      runState: state,
      runError: error,
      runEpochs: state === 'running' && s.runState !== 'running' ? [] : s.runEpochs,
    })),
  appendRunEpoch: (epoch) => set((s) => ({ runEpochs: [...s.runEpochs, epoch] })),

  shapes: {},
  pinShapes: {},
  errors: {},
  graphIssues: [],
  code: null,
  setValidationResult: (shapes, pinShapes, errors, graphIssues, code) =>
    set({ shapes, pinShapes, errors, graphIssues, code }),
  setCode: (code) => set({ code }),

  activeTab: 'model',
  setActiveTab: (tab) => set({ activeTab: tab }),

  training: {},
  setTrainingParam: (key, value) =>
    set((s) => ({ training: { ...s.training, [key]: value } })),

  data: {},
  setDataParam: (key, value) =>
    set((s) => ({ data: { ...s.data, [key]: value } })),

  toDomainGraph: () => {
    const { nodes, edges, training, data } = get()
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
      training,
      data,
    }
  },
}))
