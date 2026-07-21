import { describe, expect, it } from 'vitest'
import { adoptBestParams } from './adoptBest'
import type { SweepParamSpec } from './sweepScript'

const record = () => {
  const training: Record<string, unknown> = {}
  const nodes: Array<[string, string, string, unknown]> = []
  const data: Array<[string, string, unknown]> = []
  return {
    training,
    nodes,
    data,
    targets: {
      setTrainingParam: (k: string, v: unknown) => { training[k] = v },
      patchNodeParam: (m: string, n: string, p: string, v: unknown) => { nodes.push([m, n, p, v]) },
      patchDataParam: (n: string, p: string, v: unknown) => { data.push([n, p, v]) },
    },
  }
}

describe('adoptBestParams (draft the next run from the winner)', () => {
  it('merges loop knobs into training and patches node-targeted specs onto the canvas', () => {
    const specs: SweepParamSpec[] = [
      { name: 'lr', type: 'float', low: 0.001, high: 0.1 },
      { name: 'l1.out_features', type: 'int', low: 8, high: 64,
        node: { model: 'm1', node: 'l1', param: 'out_features' } },
    ]
    const { training, nodes, targets } = record()
    const applied = adoptBestParams({ lr: 0.0122, 'l1.out_features': 48 }, specs, targets)
    expect(applied).toBe(2)
    expect(training).toEqual({ lr: 0.0122 })
    expect(nodes).toEqual([['m1', 'l1', 'out_features', 48]])
  })

  it('adopts plain loop keys even without a surviving spec (the key IS the training key)', () => {
    const { training, targets } = record()
    expect(adoptBestParams({ optimizer: 'SGD' }, [], targets)).toBe(1)
    expect(training).toEqual({ optimizer: 'SGD' })
  })

  it('skips a dotted key whose node spec is gone — never pollutes training', () => {
    const { training, nodes, targets } = record()
    expect(adoptBestParams({ 'l1.out_features': 48 }, [], targets)).toBe(0)
    expect(training).toEqual({})
    expect(nodes).toEqual([])
  })

  it('patches a data-targeted spec onto its data node, not into training', () => {
    // A swept batch_size lives on the wired dataset node — adopting it must
    // reach the node's config (training has no batch_size to hold).
    const specs: SweepParamSpec[] = [
      { name: 'd1.batch_size', type: 'int', low: 16, high: 64, data: { node: 'd1', param: 'batch_size' } },
    ]
    const { training, data, targets } = record()
    expect(adoptBestParams({ 'd1.batch_size': 64 }, specs, targets)).toBe(1)
    expect(data).toEqual([['d1', 'batch_size', 64]])
    expect(training).toEqual({})
  })
})
