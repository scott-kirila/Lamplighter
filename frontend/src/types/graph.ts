export interface PinDef {
  name: string
  label: string
}

export interface ParamDef {
  name: string
  label: string
  type: 'int' | 'float' | 'bool' | 'shape' | 'enum'
  default: number | boolean | string
  choices?: string[] // allowed values for an 'enum' param
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
}

export interface NodeMove {
  id: string
  position: { x: number; y: number }
}
