export interface PinDef {
  name: string
  label: string
}

export interface ParamDef {
  name: string
  label: string
  type: 'int' | 'float' | 'bool' | 'shape' | 'enum' | 'tuple' | 'string' | 'multienum'
  default: number | boolean | string | string[]
  choices?: string[] // allowed values for an 'enum' param
  arity?: number // element count for a 'tuple' param
  optional?: boolean // may also be null (None)
  show_if?: Record<string, unknown> | null // show only when other params match
}

export interface NodeDef {
  type: string
  label: string
  category: string // drives the display color via nodeColor()
  inputs: PinDef[]
  outputs: PinDef[]
  params: ParamDef[]
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
  training?: Record<string, unknown>
  data?: Record<string, unknown>
}

// One model in a project: a named graph plus a spot on the system canvas. Mirrors
// the backend ModelDef; the inner graph carries only nodes/edges (training/data
// are project-level).
export interface DomainModel {
  id: string
  name: string
  graph: { nodes: DomainNode[]; edges: DomainEdge[] }
  sys_position: { x: number; y: number }
}

// A dataflow claim into a model's input on the system canvas (mirrors ModelLink).
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

// The whole design: models + how they connect + shared training/data (mirrors
// the backend Project). A single-model project is just one model, no links.
export interface DomainProject {
  version: number
  models: DomainModel[]
  links: DomainLink[]
  training?: Record<string, unknown>
  data?: Record<string, unknown>
}

export interface NodeMove {
  id: string
  position: { x: number; y: number }
}
