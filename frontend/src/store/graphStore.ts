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
import type {
  DomainGraph,
  DomainLink,
  DomainProject,
  NodeDef,
  NodeMove,
} from '../types/graph'
import { nodeColor } from '../lib/nodeColor'
import { useRunStore } from './runStore'

// The id/name of the sole model in a single-model project — matches the
// backend's SOLE_MODEL_ID, so the compat get_graph/set_graph path lines up.
export const SOLE_MODEL_ID = 'model'

// One model's identity + its place on the overview canvas. The active model's
// graph lives in the top-level nodes/edges (the editing surface); this tracks
// the metadata every model carries. Multi-model graph stashing arrives in a
// later phase — for now a project holds exactly one model.
export interface ModelMeta {
  id: string
  name: string
  sysPosition: { x: number; y: number }
}

// One undo step: the project slice of the store (never run state, selection,
// or derived results — those refresh via the validate round-trip).
interface ProjectSnapshot {
  nodes: ModelNode[]
  edges: Edge[]
  models: ModelMeta[]
  activeModelId: string
  modelGraphs: Record<string, { nodes: ModelNode[]; edges: Edge[] }>
  links: DomainLink[]
  dataNodes: DataNodeMeta[]
  training: Record<string, unknown>
}

const HISTORY_LIMIT = 50

function defaultModels(): ModelMeta[] {
  return [{ id: SOLE_MODEL_ID, name: 'Model', sysPosition: { x: 0, y: 0 } }]
}

// A data source on the overview canvas — a dataset (→ a DataLoader) or noise (→ an
// in-loop sampler). ``config`` mirrors the backend DataNode.config (a dataset's
// Data-panel form, or a noise node's dims/distribution).
export interface DataNodeMeta {
  id: string
  kind: 'dataset' | 'noise'
  name: string
  sysPosition: { x: number; y: number }
  config: Record<string, unknown>
}

// A model's canvas contents, stashed while another model is being edited (the
// active model's graph lives in the top-level nodes/edges).
export interface StashedGraph {
  nodes: ModelNode[]
  edges: Edge[]
}

// The last inference result for one model — kept per model so switching the
// active model shows its shapes immediately, without a validation round-trip.
export interface ModelResult {
  shapes: Record<string, number[]>
  pinShapes: Record<string, Record<string, number[]>>
  paramCounts: Record<string, ParamCount>
  errors: Record<string, string>
  graphIssues: string[]
  code: string | null
}

const EMPTY_RESULT: ModelResult = {
  shapes: {},
  pinShapes: {},
  paramCounts: {},
  errors: {},
  graphIssues: [],
  code: null,
}

// One model's inference result as it arrives on the wire (backend snake_case),
// before it's mapped into the camelCase ModelResult the store holds.
export interface WireModelResult {
  shapes: Record<string, number[]>
  pin_shapes?: Record<string, Record<string, number[]>>
  params?: Record<string, ParamCount>
  errors?: Record<string, string>
  graph_issues?: string[]
}

// A ReactFlow node/edge pair from a domain graph, using the registry to fill in
// each node's pins/label/color. Shared by loadGraph and loadProject.
function nodesFromDomain(
  domain: { nodes: DomainGraph['nodes']; edges: DomainGraph['edges'] },
  registry: Record<string, NodeDef>
): StashedGraph {
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
  return { nodes, edges }
}

// Serialize a ReactFlow node/edge pair back to a domain graph (nodes + edges).
function domainFromNodes(nodes: ModelNode[], edges: Edge[]): { nodes: DomainGraph['nodes']; edges: DomainGraph['edges'] } {
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
}

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

// An Input → Output scaffold (unconnected) for a fresh model canvas. Shared by
// seedDefault and addModel so a new model opens with the happy-path skeleton.
function seedGraph(registry: Record<string, NodeDef>): StashedGraph {
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
  return { nodes: seeded.filter((n): n is ModelNode => n !== null), edges: [] }
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
  // Update a node param in a specific model — the active canvas or a stashed
  // one. Used by the Data tab to auto-fill the data-fed model's Input shape even
  // when it isn't the model currently open.
  updateNodeParamInModel: (modelId: string, nodeId: string, key: string, value: unknown) => void

  // Which top-level view is active: the high-level overview, a model's
  // canvas, or the data / training config. Single-model use lands on 'model'
  // (the classic canvas) — the overview is one click away.
  activeTab: 'overview' | 'model' | 'training'
  setActiveTab: (tab: 'overview' | 'model' | 'training') => void

  // The models in the project and which one the canvas edits. The active
  // model's graph is the top-level nodes/edges; the rest are stashed in
  // modelGraphs. links are dataflow claims between models (drawn in the overview
  // view). modelResults holds each model's last inference result, so switching
  // the active model shows its shapes without a round-trip.
  models: ModelMeta[]
  activeModelId: string
  modelGraphs: Record<string, StashedGraph>
  links: DomainLink[]
  modelResults: Record<string, ModelResult>
  // Data sources on the overview canvas (dataset / noise), wired into model inputs.
  dataNodes: DataNodeMeta[]
  addDataNode: (kind: 'dataset' | 'noise') => void
  removeDataNode: (id: string) => void
  renameDataNode: (id: string, name: string) => void
  setDataNodeSysPosition: (id: string, position: { x: number; y: number }) => void
  setDataNodeConfigParam: (id: string, key: string, value: unknown) => void
  // Ensure a GAN's generator has a noise node wired into it (recipe-provisioned
  // but explicit) — a no-op if one already exists. Its dims start from the
  // generator's Input.
  ensureGanNoise: (generatorModelId: string) => void
  // Ensure the data-fed model has a dataset node wired into it — a no-op if one
  // already exists. Carries over an existing project.data form so old projects
  // keep their picks.
  ensureDatasetFor: (modelId: string) => void
  // Provision a conditional-GAN's wiring — a noise node into the generator, and a
  // dataset whose X feeds the discriminator and whose label (y) conditions both
  // models — a no-op if a dataset is already wired to the discriminator. Ports are
  // picked by Input name ("label" → the label port) else canvas position (last =
  // label), matching the recipe's own resolution.
  ensureCganWiring: (generatorModelId: string, discriminatorModelId: string) => void
  // The data node selected on the overview canvas — drives its Inspector panel.
  selectedDataNodeId: string | null
  setSelectedDataNode: (id: string | null) => void
  // The model selected on the overview canvas — drives its info pane. Mutually
  // exclusive with selectedDataNodeId (a click sets one and clears the other).
  selectedOverviewModelId: string | null
  setSelectedOverviewModel: (id: string | null) => void
  // Per-link shape-check results from the backend (id → {ok, message}); drives
  // the overview canvas's link styling and evidence labels.
  linkResults: Record<string, { ok: boolean; message: string }>
  // Draw a dataflow link between two models (overview canvas onConnect); a no-op
  // for a self-link or a duplicate. Seeds the target model's (sole) Input shape
  // from the source model's output, so the discriminator's input auto-matches
  // the generator's output the moment they're linked.
  addLink: (
    sourceModel: string,
    targetModel: string,
    targetInput?: string | null,
    sourcePin?: string | null
  ) => void
  removeLink: (id: string) => void
  setLinkResults: (links: Array<{ id: string; ok: boolean; message: string }>) => void
  // Open a model's canvas (from the overview or a model tab): stash the
  // current model's graph, load the target's, and show it.
  openModel: (id: string) => void
  // Add a new (seeded) model and open it. Returns nothing; the new model's id
  // is derived internally.
  addModel: (registry: Record<string, NodeDef>) => void
  // Remove a model (refused for the last one); switches away if it was active.
  deleteModel: (id: string) => void
  renameModel: (id: string, name: string) => void
  setModelSysPosition: (id: string, position: { x: number; y: number }) => void
  // Load a whole project (hydration / remote sync): populate all models, open
  // the first, stash the rest.
  loadProject: (project: DomainProject, registry: Record<string, NodeDef>) => void
  // Apply per-model inference results from a sync/shapes message (raw wire shape,
  // snake_case); the active model's result flows into the flat shapes/errors/…
  // maps the canvas reads.
  setProjectResults: (
    models: Record<string, WireModelResult>,
    code: Record<string, string | null> | null
  ) => void

  // Project-level training config: the recipe, its loop params, role→model
  // assignment (`roles`), and per-role params (`per_role`). Rides the project.
  training: Record<string, unknown>
  setTrainingParam: (key: string, value: unknown) => void
  // Set one per-role param, e.g. the generator's learning rate:
  // training.per_role.generator.lr.
  setTrainingRoleParam: (role: string, key: string, value: unknown) => void

  // Transient drag state for the drop-to-insert highlight: the edge a splice
  // would land on, and the node type being dragged from the palette (so a
  // dragover — where dataTransfer is unreadable — can still check eligibility).
  spliceTargetId: string | null
  setSpliceTarget: (edgeId: string | null) => void
  paletteDragType: string | null
  setPaletteDragType: (nodeType: string | null) => void
  seedDefault: (registry: Record<string, NodeDef>) => void
  // Undo/redo over the project (not runs/checkpoints/selection). capture() is
  // called at the START of each destructive action; a repeated `key` coalesces
  // (typing in one param field is a single undo step). Undoing is itself a
  // structural change, so the normal validate push persists + syncs it.
  past: ProjectSnapshot[]
  future: ProjectSnapshot[]
  _lastCaptureKey: string | null
  capture: (key?: string) => void
  undo: () => void
  redo: () => void
  // Reset the undo history and the run dashboard — the view state a "new
  // project" (blank or from a template) discards along with the old design.
  freshStart: () => void
  // The param patch without a history capture — the shared core for
  // updateNodeParamInModel and the internal seeding calls (addLink,
  // setDataNodeConfigParam), so one user gesture stays one undo step.
  _patchParamInModel: (modelId: string, nodeId: string, key: string, value: unknown) => void
  // A clean slate: the whole project back to its first-open state (one model,
  // the Input → Output scaffold, no wiring/data nodes/training config). The
  // structural change triggers the normal validate push, which overwrites the
  // autosave and syncs other tabs. Checkpoints and the run dashboard are KEPT.
  resetProject: (registry: Record<string, NodeDef>) => void
  setNodePositions: (moves: NodeMove[]) => void
  // Apply drag-end positions from a remote tab to a specific model — the active
  // model's live nodes, or an inactive model's stashed graph.
  applyModelMoves: (modelId: string | null, moves: NodeMove[]) => void
  // Apply drag-end positions on the overview canvas (model sys_positions).
  applyOverviewMoves: (moves: NodeMove[]) => void
  // Merge per-model generated code from a 'code' push into the model results.
  setProjectCode: (code: Record<string, string | null>) => void

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

  toDomainGraph: () => DomainGraph
  // The whole project (Phase B: the one model's graph + project training/data).
  toProject: () => DomainProject
}

export const useGraphStore = create<GraphState>((set, get) => ({
  nodes: [],
  edges: [],

  models: defaultModels(),
  activeModelId: SOLE_MODEL_ID,
  modelGraphs: {},
  links: [],
  modelResults: {},
  linkResults: {},

  dataNodes: [],
  selectedDataNodeId: null,
  setSelectedDataNode: (id) => set({ selectedDataNodeId: id }),
  selectedOverviewModelId: null,
  setSelectedOverviewModel: (id) => set({ selectedOverviewModelId: id }),
  addDataNode: (kind) => {
    get().capture()
    set((s) => {
      const id = crypto.randomUUID()
      const same = s.dataNodes.filter((d) => d.kind === kind)
      const base = kind === 'noise' ? 'Noise' : 'Data'
      const name = same.length === 0 ? base : `${base} ${same.length + 1}`
      // Place a new node just left of the models (near the leftmost one).
      const minX = Math.min(0, ...s.models.map((m) => m.sysPosition.x), ...s.dataNodes.map((d) => d.sysPosition.x))
      const maxY = Math.max(0, ...s.dataNodes.map((d) => d.sysPosition.y))
      // dataset: the Data-panel form defaults; noise: a per-sample latent shape
      // ("100" like an Input shape, batch excluded) + distribution.
      const config = kind === 'noise' ? { dims: '100', distribution: 'normal' } : { source: 'memory' }
      return {
        dataNodes: [...s.dataNodes, { id, kind, name, sysPosition: { x: minX - 260, y: maxY + 120 }, config }],
        selectedDataNodeId: id,
        selectedOverviewModelId: null, // keep the two selections mutually exclusive
      }
    })
  },
  removeDataNode: (id) => {
    get().capture()
    set((s) => ({
      dataNodes: s.dataNodes.filter((d) => d.id !== id),
      links: s.links.filter((l) => l.source_data !== id),
      selectedDataNodeId: s.selectedDataNodeId === id ? null : s.selectedDataNodeId,
    }))
  },
  renameDataNode: (id, name) => {
    get().capture(`rdn:${id}`)
    set((s) => ({ dataNodes: s.dataNodes.map((d) => (d.id === id ? { ...d, name } : d)) }))
  },
  setDataNodeSysPosition: (id, position) =>
    set((s) => ({ dataNodes: s.dataNodes.map((d) => (d.id === id ? { ...d, sysPosition: position } : d)) })),
  setDataNodeConfigParam: (id, key, value) => {
    get().capture(`dc:${id}:${key}`)
    const s = get()
    const dataNodes = s.dataNodes.map((d) => (d.id === id ? { ...d, config: { ...d.config, [key]: value } } : d))
    set({ dataNodes })
    // Editing a noise node's dims re-seeds the wired model's (sole) Input — the
    // noise node is the latent source of truth.
    const dn = dataNodes.find((d) => d.id === id)
    if (dn?.kind === 'noise' && key === 'dims') {
      const link = s.links.find((l) => l.source_data === id)
      if (link) {
        const nodesOf = (mid: string) =>
          mid === get().activeModelId ? get().nodes : get().modelGraphs[mid]?.nodes ?? []
        const inputs = nodesOf(link.target_model).filter((n) => n.data.nodeType === 'Input')
        if (inputs.length === 1) {
          get()._patchParamInModel(link.target_model, inputs[0].id, 'shape', `1, ${String(value)}`)
        }
      }
    }
  },
  ensureGanNoise: (generatorModelId) =>
    set((s) => {
      const wired = s.links.some(
        (l) =>
          l.target_model === generatorModelId &&
          s.dataNodes.some((d) => d.id === l.source_data && d.kind === 'noise')
      )
      if (wired) return {}
      // Seed the noise dims from the generator's current Input (batch dropped).
      const genNodes = generatorModelId === s.activeModelId ? s.nodes : s.modelGraphs[generatorModelId]?.nodes ?? []
      const input = genNodes.find((n) => n.data.nodeType === 'Input')
      const shape = String(input?.data.params.shape ?? '1, 100')
      const dims = shape.split(',').map((t) => t.trim()).filter(Boolean).slice(1).join(', ') || '100'
      const id = crypto.randomUUID()
      const gen = s.models.find((m) => m.id === generatorModelId)
      const minX = Math.min(0, ...s.models.map((m) => m.sysPosition.x), ...s.dataNodes.map((d) => d.sysPosition.x))
      const noise: DataNodeMeta = {
        id, kind: 'noise', name: 'Noise',
        sysPosition: { x: minX - 260, y: gen?.sysPosition.y ?? 0 },
        config: { dims, distribution: 'normal' },
      }
      const link: DomainLink = {
        id: crypto.randomUUID(), source_data: id, target_model: generatorModelId, target_input: null,
      }
      return { dataNodes: [...s.dataNodes, noise], links: [...s.links, link] }
    }),
  ensureDatasetFor: (modelId) =>
    set((s) => {
      const wired = s.links.some(
        (l) =>
          l.target_model === modelId &&
          s.dataNodes.some((d) => d.id === l.source_data && d.kind === 'dataset')
      )
      if (wired) return {}
      const id = crypto.randomUUID()
      const model = s.models.find((m) => m.id === modelId)
      const minX = Math.min(0, ...s.models.map((m) => m.sysPosition.x), ...s.dataNodes.map((d) => d.sysPosition.x))
      const dataset: DataNodeMeta = {
        id, kind: 'dataset', name: 'Data',
        sysPosition: { x: minX - 260, y: model?.sysPosition.y ?? 0 },
        config: { source: 'memory' },
      }
      const link: DomainLink = {
        id: crypto.randomUUID(), source_data: id, target_model: modelId, target_input: null,
      }
      return { dataNodes: [...s.dataNodes, dataset], links: [...s.links, link] }
    }),
  ensureCganWiring: (generatorId, discriminatorId) =>
    set((s) => {
      // Idempotent once the conditional wiring exists — its signature is a
      // y-pinned link into either model, which only the fan-out below creates.
      // A plain dataset link (no source_pin, planted by ensureDatasetFor when a
      // role is assigned before its partner) does NOT count as wired, so a
      // half-provisioned — or restored-broken — project still gets completed.
      const conditioned = s.links.some(
        (l) => l.source_pin === 'y' && (l.target_model === generatorId || l.target_model === discriminatorId)
      )
      if (conditioned) return {}

      // Heal any plain, port-less dataset link a prior ensureDatasetFor planted
      // into either model: reuse that dataset node (don't spawn a second) and
      // drop its links — the pinned x/y fan-out below replaces them.
      const datasetKindIds = new Set(s.dataNodes.filter((d) => d.kind === 'dataset').map((d) => d.id))
      const isStalePlain = (l: DomainLink) =>
        l.source_pin == null &&
        l.source_data != null &&
        datasetKindIds.has(l.source_data) &&
        (l.target_model === generatorId || l.target_model === discriminatorId)
      const reusableDatasetId = s.links.find(isStalePlain)?.source_data ?? null
      const keptLinks = s.links.filter((l) => !isStalePlain(l))

      const nodesOf = (id: string): ModelNode[] =>
        id === s.activeModelId ? s.nodes : s.modelGraphs[id]?.nodes ?? []
      // A model's (primary, label) input ports: the label is the Input named
      // "label" (case-insensitive), else the last by canvas position; the primary
      // is the first remaining input. target_input is null for a lone input (which
      // renders one plain handle), else the port's node id.
      const ports = (id: string) => {
        const inputs = nodesOf(id)
          .filter((n) => n.data.nodeType === 'Input')
          .slice()
          .sort((a, b) => a.position.y - b.position.y || a.position.x - b.position.x || a.id.localeCompare(b.id))
        const label =
          inputs.find((n) => String(n.data.params.name ?? '').trim().toLowerCase() === 'label') ??
          (inputs.length > 1 ? inputs[inputs.length - 1] : undefined)
        const primary = inputs.find((n) => n.id !== label?.id)
        const multi = inputs.length > 1
        return { primary, label, primaryId: multi ? primary?.id ?? null : null, labelId: label?.id ?? null }
      }
      const gp = ports(generatorId)
      const dp = ports(discriminatorId)

      const gen = s.models.find((m) => m.id === generatorId)
      const disc = s.models.find((m) => m.id === discriminatorId)
      const minX = Math.min(0, ...s.models.map((m) => m.sysPosition.x), ...s.dataNodes.map((d) => d.sysPosition.x))

      // Noise dims from the generator's noise (primary) Input, batch dropped.
      const noiseShape = String(gp.primary?.data.params.shape ?? '1, 100')
      const dims = noiseShape.split(',').map((t) => t.trim()).filter(Boolean).slice(1).join(', ') || '100'
      const noiseId = crypto.randomUUID()
      const datasetId = reusableDatasetId ?? crypto.randomUUID()
      const noise: DataNodeMeta = {
        id: noiseId, kind: 'noise', name: 'Noise',
        sysPosition: { x: minX - 260, y: (gen?.sysPosition.y ?? 0) - 80 },
        config: { dims, distribution: 'normal' },
      }
      const dataset: DataNodeMeta = {
        id: datasetId, kind: 'dataset', name: 'Data',
        sysPosition: { x: minX - 260, y: (disc?.sysPosition.y ?? 0) + 80 },
        config: { source: 'memory' },
      }
      const uid = () => crypto.randomUUID()
      const links: DomainLink[] = [...keptLinks]
      // noise → generator's noise port; dataset X → discriminator image; the
      // label (y) conditions both models.
      if (gp.primary)
        links.push({ id: uid(), source_data: noiseId, source_pin: null, target_model: generatorId, target_input: gp.primaryId })
      if (dp.primary)
        links.push({ id: uid(), source_data: datasetId, source_pin: 'x', target_model: discriminatorId, target_input: dp.primaryId })
      if (dp.label)
        links.push({ id: uid(), source_data: datasetId, source_pin: 'y', target_model: discriminatorId, target_input: dp.labelId })
      if (gp.label)
        links.push({ id: uid(), source_data: datasetId, source_pin: 'y', target_model: generatorId, target_input: gp.labelId })
      const dataNodes = reusableDatasetId ? [...s.dataNodes, noise] : [...s.dataNodes, noise, dataset]
      return { dataNodes, links }
    }),

  // Draw a wire on the overview canvas into a model's input — from another model's
  // output, or from a data node. The target is always a model (data has no input).
  addLink: (sourceId, targetId, targetInput, sourcePin) => {
    const s = get()
    if (sourceId === targetId) return
    if (s.dataNodes.some((d) => d.id === targetId)) return // can't wire *into* a data node
    const fromData = s.dataNodes.some((d) => d.id === sourceId)
    // A port is claimed once: dedupe by (source, target, target port) so a data
    // node can still fan out to *different* input ports of the same model.
    const port = targetInput ?? null
    const pin = sourcePin ?? null
    const dup = s.links.some(
      (l) =>
        l.target_model === targetId &&
        (l.target_input ?? null) === port &&
        (fromData ? l.source_data === sourceId : l.source_model === sourceId)
    )
    if (dup) return
    get().capture()
    const link: DomainLink = fromData
      ? { id: crypto.randomUUID(), source_data: sourceId, source_pin: pin, target_model: targetId, target_input: port }
      : { id: crypto.randomUUID(), source_model: sourceId, source_pin: pin, target_model: targetId, target_input: port }
    set({ links: [...s.links, link] })
    if (fromData) return // data→model: no output-shape seeding (noise seeding lands in G5)
    // Model→model: seed the wired Input's shape from the source's output — the
    // named target port when the wire lands on one, else the sole input.
    const nodesOf = (id: string): ModelNode[] =>
      id === s.activeModelId ? s.nodes : s.modelGraphs[id]?.nodes ?? []
    const outNode = nodesOf(sourceId).find((n) => n.data.nodeType === 'Output')
    const outShape = outNode ? s.modelResults[sourceId]?.shapes[outNode.id] : undefined
    if (!outShape || outShape.length === 0) return
    const inputs = nodesOf(targetId).filter((n) => n.data.nodeType === 'Input')
    const seedTarget = port ? inputs.find((n) => n.id === port) : inputs.length === 1 ? inputs[0] : undefined
    if (!seedTarget) return // ambiguous / none — leave it to the user
    get()._patchParamInModel(targetId, seedTarget.id, 'shape', outShape.join(', '))
  },
  removeLink: (id) => {
    get().capture()
    set((s) => ({ links: s.links.filter((l) => l.id !== id) }))
  },
  setLinkResults: (links) =>
    set({ linkResults: Object.fromEntries(links.map((l) => [l.id, { ok: l.ok, message: l.message }])) }),

  openModel: (id) =>
    set((s) => {
      if (id === s.activeModelId) return { activeTab: 'model' }
      if (!s.models.some((m) => m.id === id)) return {}
      // Stash the current model's graph; pop the target's out of the stash.
      const stashed = { ...s.modelGraphs, [s.activeModelId]: { nodes: s.nodes, edges: s.edges } }
      const target = stashed[id] ?? { nodes: [], edges: [] }
      delete stashed[id]
      const r = s.modelResults[id] ?? EMPTY_RESULT
      return {
        modelGraphs: stashed,
        nodes: target.nodes,
        edges: target.edges,
        selectedNodeId: null,
        activeModelId: id,
        activeTab: 'model',
        shapes: r.shapes,
        pinShapes: r.pinShapes,
        paramCounts: r.paramCounts,
        errors: r.errors,
        graphIssues: r.graphIssues,
        code: r.code,
      }
    }),

  addModel: (registry) => {
    get().capture()
    set((s) => {
      const id = crypto.randomUUID()
      // A unique display name: Model 2, Model 3, …
      const taken = new Set(s.models.map((m) => m.name))
      let n = s.models.length + 1
      let name = `Model ${n}`
      while (taken.has(name)) name = `Model ${++n}`
      const maxY = Math.max(0, ...s.models.map((m) => m.sysPosition.y))
      const meta: ModelMeta = { id, name, sysPosition: { x: 40, y: maxY + 130 } }
      const seed = seedGraph(registry)
      // Stash the current model, make the new (seeded) one active.
      const stashed = { ...s.modelGraphs, [s.activeModelId]: { nodes: s.nodes, edges: s.edges } }
      return {
        models: [...s.models, meta],
        modelGraphs: stashed,
        nodes: seed.nodes,
        edges: seed.edges,
        selectedNodeId: null,
        activeModelId: id,
        activeTab: 'model',
        shapes: {},
        pinShapes: {},
        paramCounts: {},
        errors: {},
        graphIssues: [],
        code: null,
      }
    })
  },

  deleteModel: (id) => {
    get().capture()
    set((s) => {
      if (s.models.length <= 1) return {} // never delete the last model
      const models = s.models.filter((m) => m.id !== id)
      const stashed = { ...s.modelGraphs }
      delete stashed[id]
      const results = { ...s.modelResults }
      delete results[id]
      // Drop any link touching the gone model (a data feed into it, or a
      // model→model wire) so nothing dangles.
      const linkKept = (l: DomainLink) => l.source_model !== id && l.target_model !== id
      const links = s.links.filter(linkKept)
      const selectedOverviewModelId = s.selectedOverviewModelId === id ? null : s.selectedOverviewModelId
      if (id !== s.activeModelId) {
        return { models, modelGraphs: stashed, modelResults: results, links, selectedOverviewModelId }
      }
      // Deleting the active model — open the first remaining one.
      const next = models[0]
      const target = stashed[next.id] ?? { nodes: [], edges: [] }
      delete stashed[next.id]
      const r = results[next.id] ?? EMPTY_RESULT
      return {
        models,
        modelGraphs: stashed,
        modelResults: results,
        links,
        selectedOverviewModelId,
        nodes: target.nodes,
        edges: target.edges,
        selectedNodeId: null,
        activeModelId: next.id,
        shapes: r.shapes,
        pinShapes: r.pinShapes,
        paramCounts: r.paramCounts,
        errors: r.errors,
        graphIssues: r.graphIssues,
        code: r.code,
      }
    })
  },

  renameModel: (id, name) => {
    get().capture(`rm:${id}`)
    set((s) => ({ models: s.models.map((m) => (m.id === id ? { ...m, name } : m)) }))
  },
  setModelSysPosition: (id, position) =>
    set((s) => ({
      models: s.models.map((m) => (m.id === id ? { ...m, sysPosition: position } : m)),
    })),

  loadProject: (project, registry) =>
    set((s) => {
      if (project.models.length === 0) return {}
      const models: ModelMeta[] = project.models.map((m) => ({
        id: m.id,
        name: m.name,
        sysPosition: m.sys_position ?? { x: 0, y: 0 },
      }))
      // Keep editing the same model across a remote sync when it still exists;
      // otherwise open the first.
      const activeId = models.some((m) => m.id === s.activeModelId) ? s.activeModelId : models[0].id
      const graphs: Record<string, StashedGraph> = {}
      for (const m of project.models) graphs[m.id] = nodesFromDomain(m.graph, registry)
      const active = graphs[activeId]
      const stashed = { ...graphs }
      delete stashed[activeId]
      const dataNodes: DataNodeMeta[] = (project.data_nodes ?? []).map((d) => ({
        id: d.id,
        kind: d.kind === 'noise' ? 'noise' : 'dataset',
        name: d.name,
        sysPosition: d.sys_position ?? { x: 0, y: 0 },
        config: d.config ?? {},
      }))
      return {
        models,
        activeModelId: activeId,
        modelGraphs: stashed,
        dataNodes,
        links: project.links ?? [],
        nodes: active.nodes,
        edges: active.edges,
        selectedNodeId: null,
        training: project.training ?? {},
      }
    }),

  setProjectResults: (models, code) =>
    set((s) => {
      const results: Record<string, ModelResult> = { ...s.modelResults }
      for (const [id, r] of Object.entries(models)) {
        results[id] = {
          shapes: r.shapes ?? {},
          pinShapes: r.pin_shapes ?? {},
          paramCounts: r.params ?? {},
          errors: r.errors ?? {},
          graphIssues: r.graph_issues ?? [],
          code: code ? code[id] ?? null : s.modelResults[id]?.code ?? null,
        }
      }
      // The active model's result flows into the flat maps the canvas reads.
      const active = results[s.activeModelId] ?? EMPTY_RESULT
      return {
        modelResults: results,
        shapes: active.shapes,
        pinShapes: active.pinShapes,
        paramCounts: active.paramCounts,
        errors: active.errors,
        graphIssues: active.graphIssues,
        // Preserve existing code when this payload didn't carry any (e.g. a
        // shapes-only update while the panel is closed).
        code: code ? active.code : s.code,
      }
    }),

  onNodesChange: (changes) => {
    // Drag ticks/selection stream through here constantly — only a removal is
    // a destructive edit worth an undo step.
    if (changes.some((c) => c.type === 'remove')) get().capture()
    set((s) => ({ nodes: applyNodeChanges(changes, s.nodes) }))
  },

  onEdgesChange: (changes) => {
    if (changes.some((c) => c.type === 'remove')) get().capture()
    set((s) => ({ edges: applyEdgeChanges(changes, s.edges) }))
  },

  onConnect: (conn) => {
    get().capture()
    set((s) => {
      // A target input handle accepts a single edge — a new connection
      // replaces any existing one rather than silently fanning in.
      const cleared = s.edges.filter(
        (e) => !(e.target === conn.target && e.targetHandle === conn.targetHandle)
      )
      return { edges: addEdge(conn, cleared) }
    })
  },

  selectedNodeId: null,
  setSelectedNode: (id) => set({ selectedNodeId: id }),

  addNode: (nodeDef, position) => {
    get().capture()
    set((s) => ({ nodes: [...s.nodes, buildNode(nodeDef, position)] }))
  },

  // Splice a node into an existing edge A→B: drop the original edge and rewire
  // A→N→B through the new node's first input/output handles. Caller ensures the
  // node has both (Input/Output can't be spliced). Falls back to a plain add if
  // the edge has since vanished.
  insertNodeOnEdge: (nodeDef, position, edgeId) => {
    get().capture()
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
    })
  },

  // Splice an existing (unconnected) node into an edge — same rewiring as
  // insertNodeOnEdge, but for a node already on the canvas. The drag handler
  // gates this to unconnected, splice-capable nodes.
  spliceNodeIntoEdge: (nodeId, edgeId) => {
    get().capture()
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
    })
  },

  updateNodeParam: (nodeId, key, value) => {
    get().capture(`p:${nodeId}:${key}`)
    set((s) => ({
      nodes: s.nodes.map((n) =>
        n.id === nodeId
          ? { ...n, data: { ...n.data, params: { ...n.data.params, [key]: value } } }
          : n
      ),
    }))
  },

  updateNodeParamInModel: (modelId, nodeId, key, value) => {
    get().capture(`pm:${modelId}:${nodeId}:${key}`)
    get()._patchParamInModel(modelId, nodeId, key, value)
  },

  _patchParamInModel: (modelId, nodeId, key, value) =>
    set((s) => {
      const patch = (ns: ModelNode[]) =>
        ns.map((n) =>
          n.id === nodeId
            ? { ...n, data: { ...n.data, params: { ...n.data.params, [key]: value } } }
            : n
        )
      if (modelId === s.activeModelId) return { nodes: patch(s.nodes) }
      const stash = s.modelGraphs[modelId]
      if (!stash) return {}
      return { modelGraphs: { ...s.modelGraphs, [modelId]: { ...stash, nodes: patch(stash.nodes) } } }
    }),

  // Seed a fresh canvas with an Input → Output scaffold (unconnected, so adding
  // a layer between them needs no edge deletion). Ordinary deletable nodes —
  // correctness is enforced by validation, not by locking these in place.
  past: [],
  future: [],
  _lastCaptureKey: null,
  capture: (key) =>
    set((s) => {
      if (key !== undefined && key === s._lastCaptureKey) return {} // coalesce
      const snap: ProjectSnapshot = {
        nodes: s.nodes, edges: s.edges, models: s.models, activeModelId: s.activeModelId,
        modelGraphs: s.modelGraphs, links: s.links, dataNodes: s.dataNodes, training: s.training,
      }
      return {
        past: [...s.past.slice(-(HISTORY_LIMIT - 1)), snap],
        future: [], // a new edit forks history — redo no longer applies
        _lastCaptureKey: key ?? null,
      }
    }),
  undo: () =>
    set((s) => {
      const prev = s.past[s.past.length - 1]
      if (!prev) return {}
      const snap: ProjectSnapshot = {
        nodes: s.nodes, edges: s.edges, models: s.models, activeModelId: s.activeModelId,
        modelGraphs: s.modelGraphs, links: s.links, dataNodes: s.dataNodes, training: s.training,
      }
      return {
        ...prev,
        past: s.past.slice(0, -1),
        future: [...s.future, snap],
        _lastCaptureKey: null,
        selectedNodeId: null,
        selectedDataNodeId: null,
        selectedOverviewModelId: null,
      }
    }),
  redo: () =>
    set((s) => {
      const next = s.future[s.future.length - 1]
      if (!next) return {}
      const snap: ProjectSnapshot = {
        nodes: s.nodes, edges: s.edges, models: s.models, activeModelId: s.activeModelId,
        modelGraphs: s.modelGraphs, links: s.links, dataNodes: s.dataNodes, training: s.training,
      }
      return {
        ...next,
        past: [...s.past, snap],
        future: s.future.slice(0, -1),
        _lastCaptureKey: null,
        selectedNodeId: null,
        selectedDataNodeId: null,
        selectedOverviewModelId: null,
      }
    }),

  seedDefault: (registry) => {
    const seed = seedGraph(registry)
    set((s) => ({
      nodes: seed.nodes,
      edges: seed.edges,
      selectedNodeId: null,
      models: s.models.length > 0 ? [s.models[0]] : defaultModels(),
      activeModelId: s.models[0]?.id ?? SOLE_MODEL_ID,
      modelGraphs: {},
      links: [],
    }))
  },

  // The view resets a fresh project entails, shared by the blank reset and a
  // template load: undo history and the last run's dashboard both belong to the
  // project being replaced, so a "new project" starts clean on both. (The
  // kernel's trained model + checkpoints are untouched — a canvas action doesn't
  // reach across and destroy them.)
  freshStart: () => {
    set({ past: [], future: [], _lastCaptureKey: null })
    useRunStore.getState().reset() // the run dashboard is the run store's to clear
  },

  resetProject: (registry) => {
    // "New project" is a history BOUNDARY, not an undoable edit — File→New
    // starts fresh (the confirm dialog is the guard), so no capture() here.
    const seed = seedGraph(registry)
    set({
      nodes: seed.nodes,
      edges: seed.edges,
      selectedNodeId: null,
      models: defaultModels(),
      activeModelId: SOLE_MODEL_ID,
      modelGraphs: {},
      modelResults: {},
      links: [],
      linkResults: {},
      dataNodes: [],
      selectedDataNodeId: null,
      selectedOverviewModelId: null,
      training: {},
      // Stale readouts cleared now; the validate reply repopulates in a beat.
      shapes: {},
      pinShapes: {},
      paramCounts: {},
      errors: {},
      graphIssues: [],
      // Land on the fresh scaffold, ready to build.
      activeTab: 'model',
    })
    get().freshStart()
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

  applyModelMoves: (modelId, moves) =>
    set((s) => {
      const byId = new Map(moves.map((m) => [m.id, m.position]))
      // The move targets the active model (or an unspecified one) → live nodes.
      if (modelId === null || modelId === s.activeModelId) {
        return { nodes: s.nodes.map((n) => (byId.has(n.id) ? { ...n, position: byId.get(n.id)! } : n)) }
      }
      // Otherwise patch the target model's stashed graph.
      const stash = s.modelGraphs[modelId]
      if (!stash) return {}
      return {
        modelGraphs: {
          ...s.modelGraphs,
          [modelId]: {
            ...stash,
            nodes: stash.nodes.map((n) => (byId.has(n.id) ? { ...n, position: byId.get(n.id)! } : n)),
          },
        },
      }
    }),

  applyOverviewMoves: (moves) =>
    set((s) => {
      const byId = new Map(moves.map((m) => [m.id, m.position]))
      return { models: s.models.map((m) => (byId.has(m.id) ? { ...m, sysPosition: byId.get(m.id)! } : m)) }
    }),

  setProjectCode: (code) =>
    set((s) => {
      const results = { ...s.modelResults }
      for (const [id, src] of Object.entries(code)) {
        results[id] = { ...(results[id] ?? EMPTY_RESULT), code: src }
      }
      return {
        modelResults: results,
        code: s.activeModelId in code ? code[s.activeModelId] : s.code,
      }
    }),

  spliceTargetId: null,
  // No-op when unchanged so the frequent dragover/drag updates don't re-render.
  setSpliceTarget: (edgeId) =>
    set((s) => (s.spliceTargetId === edgeId ? {} : { spliceTargetId: edgeId })),
  paletteDragType: null,
  setPaletteDragType: (nodeType) => set({ paletteDragType: nodeType }),

  shapes: {},
  pinShapes: {},
  paramCounts: {},
  errors: {},
  graphIssues: [],
  code: null,

  activeTab: 'model',
  setActiveTab: (tab) => set({ activeTab: tab }),

  training: {},
  setTrainingParam: (key, value) => {
    get().capture(`t:${key}`)
    set((s) => ({ training: { ...s.training, [key]: value } }))
  },
  setTrainingRoleParam: (role, key, value) => {
    get().capture(`tr:${role}:${key}`)
    set((s) => {
      const per = (s.training.per_role as Record<string, Record<string, unknown>>) ?? {}
      return {
        training: {
          ...s.training,
          per_role: { ...per, [role]: { ...(per[role] ?? {}), [key]: value } },
        },
      }
    })
  },

  toDomainGraph: () => {
    const { nodes, edges, training } = get()
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
    }
  },

  // Assemble the whole project: the active model's graph is the top-level
  // nodes/edges; every other model comes from its stash. training is
  // project-level; data lives on the wired data nodes.
  toProject: () => {
    const { models, activeModelId, modelGraphs, dataNodes, links, training, nodes, edges } = get()
    return {
      version: 3,
      models: models.map((m) => {
        const stash = m.id === activeModelId ? { nodes, edges } : modelGraphs[m.id]
        return {
          id: m.id,
          name: m.name,
          graph: stash ? domainFromNodes(stash.nodes, stash.edges) : { nodes: [], edges: [] },
          sys_position: m.sysPosition,
        }
      }),
      data_nodes: dataNodes.map((d) => ({
        id: d.id,
        kind: d.kind,
        name: d.name,
        sys_position: d.sysPosition,
        config: d.config,
      })),
      links,
      training,
    }
  },
}))
