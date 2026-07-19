import type { Edge, Node } from '@xyflow/react'
import type { DomainLink } from '../types/graph'

// Naming convention across the store's data shapes:
//   • The `Domain*` types (../types/graph) mirror the backend wire format
//     field-for-field, so they stay snake_case. Types that are pure pass-through
//     data — `links: DomainLink[]` — are held as the Domain type directly, since
//     load/save is then a straight JSON copy with no remapping.
//   • The store's own working types (`ModelMeta`, `DataNodeMeta`, `ModelResult`)
//     use camelCase. They carry UI/working state and are mapped from the wire
//     shape at the load boundary (e.g. sys_position → sysPosition), so they're
//     deliberately allowed to diverge from the serialized form.

// The id of the sole model in a single-model project — matches the backend's
// SOLE_MODEL_ID, so a single-model project round-trips with a stable model id.
export const SOLE_MODEL_ID = 'model'

export const HISTORY_LIMIT = 50

// One model's identity + its place on the overview canvas. The active model's
// graph lives in the top-level nodes/edges (the editing surface); this tracks
// the metadata every model carries.
export interface ModelMeta {
  id: string
  name: string
  sysPosition: { x: number; y: number }
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

export const EMPTY_RESULT: ModelResult = {
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

// One undo step: the project slice of the store (never run state, selection,
// or derived results — those refresh via the validate round-trip).
export interface ProjectSnapshot {
  nodes: ModelNode[]
  edges: Edge[]
  models: ModelMeta[]
  activeModelId: string
  modelGraphs: Record<string, { nodes: ModelNode[]; edges: Edge[] }>
  links: DomainLink[]
  dataNodes: DataNodeMeta[]
  training: Record<string, unknown>
}
