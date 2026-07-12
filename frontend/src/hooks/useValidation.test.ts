import { describe, expect, it } from 'vitest'
import { keyFromProject } from './useValidation'
import type { DomainProject } from '../types/graph'

// The structural signature that drives re-validation and the cross-tab echo
// guard. Two rules must hold or the app misbehaves:
//   • it MUST ignore positions — else every drag re-validates (and a drag echoed
//     from another tab would rebuild the canvas).
//   • it MUST change on any structural edit (params/edges/wiring/config) — else a
//     real change is silently skipped and shapes go stale.
function baseProject(): DomainProject {
  return {
    version: 3,
    models: [
      {
        id: 'm1',
        name: 'Model',
        sys_position: { x: 0, y: 0 },
        graph: {
          nodes: [
            { id: 'in', type: 'Input', position: { x: 0, y: 0 }, params: { shape: '1, 784' } },
            { id: 'out', type: 'Output', position: { x: 300, y: 0 }, params: {} },
          ],
          edges: [
            { id: 'e1', source: 'in', sourceHandle: 'output', target: 'out', targetHandle: 'input' },
          ],
        },
      },
    ],
    data_nodes: [
      { id: 'd1', kind: 'dataset', name: 'Data', sys_position: { x: -200, y: 0 }, config: { source: 'memory' } },
    ],
    links: [{ id: 'l1', source_data: 'd1', target_model: 'm1', target_input: null }],
    training: { loss: 'CrossEntropyLoss', epochs: 10 },
  }
}

const clone = (p: DomainProject): DomainProject => structuredClone(p)
const key = keyFromProject

describe('keyFromProject — stable identity', () => {
  it('is deterministic: a freshly-built equal project yields the same key', () => {
    expect(key(baseProject())).toBe(key(baseProject()))
  })
})

describe('keyFromProject — positions are excluded (dragging must not re-validate)', () => {
  it('ignores a node position', () => {
    const p = clone(baseProject())
    p.models[0].graph.nodes[0].position = { x: 999, y: 999 }
    expect(key(p)).toBe(key(baseProject()))
  })

  it('ignores a model sys_position', () => {
    const p = clone(baseProject())
    p.models[0].sys_position = { x: 500, y: 500 }
    expect(key(p)).toBe(key(baseProject()))
  })

  it('ignores a data-node sys_position', () => {
    const p = clone(baseProject())
    p.data_nodes[0].sys_position = { x: 42, y: 42 }
    expect(key(p)).toBe(key(baseProject()))
  })
})

describe('keyFromProject — structural edits change the key', () => {
  const changed = (mutate: (p: DomainProject) => void): boolean => {
    const p = clone(baseProject())
    mutate(p)
    return key(p) !== key(baseProject())
  }

  it('a node param change', () => {
    expect(changed((p) => { p.models[0].graph.nodes[0].params = { shape: '1, 32' } })).toBe(true)
  })

  it('a node type change', () => {
    expect(changed((p) => { p.models[0].graph.nodes[0].type = 'Embedding' })).toBe(true)
  })

  it('adding a node', () => {
    expect(changed((p) => {
      p.models[0].graph.nodes.push({ id: 'l', type: 'Linear', position: { x: 1, y: 1 }, params: {} })
    })).toBe(true)
  })

  it('a rewired edge (endpoint)', () => {
    expect(changed((p) => { p.models[0].graph.edges[0].target = 'somewhere' })).toBe(true)
  })

  it('an edge handle change', () => {
    expect(changed((p) => { p.models[0].graph.edges[0].targetHandle = 'in1' })).toBe(true)
  })

  it('a link retarget / pin change', () => {
    expect(changed((p) => { p.links[0].source_pin = 'y' })).toBe(true)
    expect(changed((p) => { p.links[0].target_input = 'in' })).toBe(true)
  })

  it('a data-node config change', () => {
    expect(changed((p) => { p.data_nodes[0].config = { source: 'memory', X: 'features' } })).toBe(true)
  })

  it('adding a data node', () => {
    expect(changed((p) => {
      p.data_nodes.push({ id: 'noise', kind: 'noise', name: 'Noise', sys_position: { x: 0, y: 0 }, config: {} })
    })).toBe(true)
  })

  it('a training-config change', () => {
    expect(changed((p) => { p.training = { loss: 'MSELoss', epochs: 10 } })).toBe(true)
  })

  it('adding a second model', () => {
    expect(changed((p) => {
      p.models.push({ id: 'm2', name: 'Second', sys_position: { x: 0, y: 0 }, graph: { nodes: [], edges: [] } })
    })).toBe(true)
  })
})
