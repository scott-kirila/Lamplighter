/**
 * Is this project still the untouched scaffold a fresh session seeds?
 *
 * `seedGraph` lays down a bare Input and Output with no wire between them, and
 * nothing else. That state is indistinguishable from "the user is midway
 * through building" by node count alone, so the test is deliberately strict:
 * one model, only I/O nodes, no wires, no data. The moment any of those change
 * the user has made a decision, and offering to replace their work would be
 * rude — the Start panel exists for the state where there is nothing to lose.
 */
export function isPristineProject(input: {
  models: { id: string }[]
  dataNodes: { id: string }[]
  nodes: { data: { nodeType: string } }[]
  edges: unknown[]
}): boolean {
  if (input.models.length !== 1) return false
  if (input.dataNodes.length > 0) return false
  if (input.edges.length > 0) return false
  if (input.nodes.length > 2) return false
  return input.nodes.every((n) => n.data.nodeType === 'Input' || n.data.nodeType === 'Output')
}
