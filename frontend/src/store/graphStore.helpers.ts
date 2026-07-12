import type { Edge } from '@xyflow/react'
import type { DomainGraph, NodeDef } from '../types/graph'
import { nodeColor } from '../lib/nodeColor'
import { SOLE_MODEL_ID, type ModelMeta, type ModelNode, type StashedGraph } from './graphStore.types'

export function defaultModels(): ModelMeta[] {
  return [{ id: SOLE_MODEL_ID, name: 'Model', sysPosition: { x: 0, y: 0 } }]
}

// A ReactFlow node/edge pair from a domain graph, using the registry to fill in
// each node's pins/label/color. Shared by loadGraph and loadProject.
export function nodesFromDomain(
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
export function domainFromNodes(
  nodes: ModelNode[],
  edges: Edge[]
): { nodes: DomainGraph['nodes']; edges: DomainGraph['edges'] } {
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

// Rewire edge A→B into A→N→B, splicing node N (via the given handles) in place
// of the original edge. Returns the new edge list.
export function splicedEdges(
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
export function placeAndMakeRoom(
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
export function seedGraph(registry: Record<string, NodeDef>): StashedGraph {
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
export function buildNode(nodeDef: NodeDef, position: { x: number; y: number }): ModelNode {
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
