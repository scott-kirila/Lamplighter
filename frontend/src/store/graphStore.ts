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

// A layer's parameter count and its factorization (parameter tensor shapes).
export interface ParamCount {
  count: number
  terms: number[][]
}

// One epoch of a streamed in-kernel training run.
export interface RunEpoch {
  epoch: number
  epochs: number
  metrics: Record<string, number>
}

// Rebuild the per-epoch stream from a run's history dict (metric name → series),
// for tabs that join mid-run or after it — GET /api/run/status returns the full
// history, and the dashboard renders RunEpoch[]. A metric appears in an epoch's
// metrics only when its series reaches that epoch (e.g. no val without a
// val_loader).
export function epochsFromHistory(
  history: Record<string, number[]> | null | undefined,
  plannedEpochs: number
): RunEpoch[] {
  if (!history) return []
  const n = Math.max(0, ...Object.values(history).map((v) => v.length))
  return Array.from({ length: n }, (_, i) => ({
    epoch: i + 1,
    epochs: plannedEpochs,
    metrics: Object.fromEntries(
      Object.entries(history)
        .filter(([, v]) => i < v.length)
        .map(([k, v]) => [k, v[i]])
    ),
  }))
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

// Approximate rendered node footprint, for making room on insert.
const NODE_WIDTH = 190
const NODE_HEIGHT = 110
const INSERT_GAP = 40
const PITCH = NODE_WIDTH + INSERT_GAP // one node column, with breathing room

// Fit a node spliced onto the edge source→target at `pos` without overlaps:
// nudge it right until it clears the source (only when they'd vertically
// overlap — a drop below the wire keeps its x), then, if the gap to the target
// can't fit it, slide every node from the target's column rightward by the
// shortfall — the minimum move, applied uniformly so all relative arrangement
// (parallel branches included) is preserved. Only for left-to-right edges;
// free-form/vertical layouts are left alone. Returns the adjusted drop
// position and the (possibly shifted) node list; `skipId` pins the spliced
// node itself.
function placeAndMakeRoom(
  nodes: ModelNode[],
  sourceId: string,
  targetId: string,
  pos: { x: number; y: number },
  skipId?: string
): { position: { x: number; y: number }; nodes: ModelNode[] } {
  const source = nodes.find((n) => n.id === sourceId)
  const target = nodes.find((n) => n.id === targetId)
  if (!source || !target || target.position.x <= source.position.x) {
    return { position: pos, nodes }
  }

  // Clear the left neighbor.
  const overlapsSourceRow = Math.abs(pos.y - source.position.y) < NODE_HEIGHT
  const x = overlapsSourceRow ? Math.max(pos.x, source.position.x + PITCH) : pos.x
  const position = { x, y: pos.y }

  // Make room before the right neighbor.
  const delta = PITCH - (target.position.x - x)
  if (delta <= 0) return { position, nodes }
  const threshold = target.position.x
  return {
    position,
    nodes: nodes.map((n) =>
      n.id !== skipId && n.position.x >= threshold
        ? { ...n, position: { ...n.position, x: n.position.x + delta } }
        : n
    ),
  }
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
  runSeed: number | null
  setRunStatus: (state: GraphState['runState'], error: string | null, seed?: number | null) => void
  appendRunEpoch: (epoch: RunEpoch) => void
  // Seed run state from GET /api/run/status on (re)connect, so a tab that joins
  // mid-run (or after) shows the run instead of waiting for the next WS event.
  hydrateRun: (
    state: GraphState['runState'],
    error: string | null,
    epochs: RunEpoch[],
    seed?: number | null
  ) => void

  shapes: Record<string, number[]>
  // Per-output-pin shapes ({ nodeId: { pin: dims } }) — powers the Inspector's
  // per-pin readout for multi-output nodes (LSTM's output / h_n / c_n).
  pinShapes: Record<string, Record<string, number[]>>
  // Per-node parameter counts + the parameter tensors' shapes (the count's
  // factorization), from the meta-instantiated modules.
  paramCounts: Record<string, ParamCount>
  errors: Record<string, string>
  graphIssues: string[]
  code: string | null
  setValidationResult: (
    shapes: Record<string, number[]>,
    pinShapes: Record<string, Record<string, number[]>>,
    paramCounts: Record<string, ParamCount>,
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
      const edge = s.edges.find((e) => e.id === edgeId)
      if (!edge) return { nodes: [...s.nodes, buildNode(nodeDef, position)] }
      // Clear the left neighbor and slide the right-hand side over if needed.
      const fitted = placeAndMakeRoom(s.nodes, edge.source, edge.target, position)
      const node = buildNode(nodeDef, fitted.position)
      return {
        nodes: [...fitted.nodes, node],
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
      // Same fitting as palette inserts, applied to the already-placed node.
      const fitted = placeAndMakeRoom(s.nodes, edge.source, edge.target, node.position, nodeId)
      return {
        nodes: fitted.nodes.map((n) =>
          n.id === nodeId ? { ...n, position: fitted.position } : n
        ),
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
  runSeed: null,
  // Entering "running" clears the previous run's lines so the panel starts fresh.
  setRunStatus: (state, error, seed) =>
    set((s) => ({
      runState: state,
      runError: error,
      runSeed: seed !== undefined ? seed : s.runSeed,
      runEpochs: state === 'running' && s.runState !== 'running' ? [] : s.runEpochs,
    })),
  // Ignore epochs at/behind the newest one — protects against the hydration
  // fetch racing a live run_epoch event (which could otherwise duplicate a line).
  appendRunEpoch: (epoch) =>
    set((s) => {
      const last = s.runEpochs[s.runEpochs.length - 1]
      if (last && epoch.epoch <= last.epoch) return {}
      return { runEpochs: [...s.runEpochs, epoch] }
    }),

  // Conservative merge: live WS events win. State applies only when this tab
  // hasn't seen a transition yet (a late joiner misses the "running" broadcast);
  // the fetched epoch list applies only when it's more complete than ours.
  hydrateRun: (state, error, epochs, seed = null) =>
    set((s) => ({
      runState: s.runState === 'idle' ? state : s.runState,
      runError: s.runError ?? error,
      runSeed: s.runSeed ?? seed,
      runEpochs: epochs.length > s.runEpochs.length ? epochs : s.runEpochs,
    })),

  shapes: {},
  pinShapes: {},
  paramCounts: {},
  errors: {},
  graphIssues: [],
  code: null,
  setValidationResult: (shapes, pinShapes, paramCounts, errors, graphIssues, code) =>
    set({ shapes, pinShapes, paramCounts, errors, graphIssues, code }),
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
