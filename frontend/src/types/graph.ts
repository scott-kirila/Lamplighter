export interface PinDef {
  name: string
  label: string
}

export interface ParamDef {
  name: string
  label: string
  // 'module' renders as a picker over the session's registered nn.Modules
  // (sess.modules(Name=Class)) — the Custom node's class selector.
  type: 'int' | 'float' | 'bool' | 'shape' | 'enum' | 'tuple' | 'string' | 'multienum' | 'module'
  default: number | boolean | string | string[]
  choices?: string[] // allowed values for an 'enum' param
  arity?: number // element count for a 'tuple' param
  optional?: boolean // may also be null (None)
  show_if?: Record<string, unknown> | null // show only when other params match
  // One line explaining the field, shown as the label's tooltip. The rule it
  // enforces: labels carry the name and the UNIT, explanations live here.
  help?: string | null
  choice_labels?: Record<string, string> | null // display text per enum choice
  placeholder?: string | null // example input for a free-text field
}

export interface NodeDef {
  type: string
  label: string
  category: string // drives the display color via nodeColor()
  // Optional second-level grouping within a category (e.g. layers →
  // Convolution / Pooling); drives the palette's sub-headers.
  subcategory?: string | null
  inputs: PinDef[]
  outputs: PinDef[]
  params: ParamDef[]
  // Help text: an nn-backed node's live torch docstring (summary = first prose
  // paragraph; body = the full cleaned text), or an authored line (body: '').
  doc?: { summary: string; body: string } | null
}

export interface DomainNode {
  id: string
  type: string
  position: { x: number; y: number }
  params: Record<string, unknown>
}

export interface DomainEdge {
  id: string
  source: string
  sourceHandle: string
  target: string
  targetHandle: string
}

export interface DomainGraph {
  nodes: DomainNode[]
  edges: DomainEdge[]
}

// One model in a project: a named graph plus a spot on the overview canvas. Mirrors
// the backend ModelDef; the inner graph carries only nodes/edges (training/data
// are project-level).
export interface DomainModel {
  id: string
  name: string
  graph: { nodes: DomainNode[]; edges: DomainEdge[] }
  sys_position: { x: number; y: number }
}

// A dataflow claim into a model's input on the overview canvas (mirrors ModelLink).
// The source is another model's output (source_model) or a data node
// (source_data) — exactly one is set.
export interface DomainLink {
  id: string
  source_model?: string | null
  source_pin?: string | null
  source_data?: string | null
  target_model: string
  target_input?: string | null
}

// A data source on the overview canvas (mirrors the backend DataNode). A dataset
// node maps to a DataLoader; a noise node to an in-loop sampler.
export interface DomainDataNode {
  id: string
  kind: string // 'dataset' | 'noise'
  name: string
  sys_position: { x: number; y: number }
  config: Record<string, unknown>
}

// The whole project: models + data sources + how they connect + shared training
// config (mirrors the backend Project). A single-model project is just one
// model, no data nodes, no links.
export interface DomainProject {
  version: number
  models: DomainModel[]
  data_nodes: DomainDataNode[]
  links: DomainLink[]
  training?: Record<string, unknown>
}

export interface NodeMove {
  id: string
  position: { x: number; y: number }
}
