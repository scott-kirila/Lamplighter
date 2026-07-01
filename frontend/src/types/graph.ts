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
  category: string
  color: string
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

export interface NodeMove {
  id: string
  position: { x: number; y: number }
}
