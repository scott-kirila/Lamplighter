import { describe, expect, it } from 'vitest'
import { buildHealth, layerVerdict, nodeVerdicts } from './useTrainingHealth'
import type { RunEpoch } from '../store/runStore'

describe('layerVerdict (the update-ratio → verdict)', () => {
  const s = (dw: number[], extra: { w?: number[]; g?: number[] } = {}) => ({
    w: extra.w ?? dw.map(() => 1),
    dw,
    g: extra.g ?? [],
  })

  it('healthy in the ~1e-3 band', () => {
    expect(layerVerdict(s([1.2e-3, 1.1e-3, 9e-4])).level).toBe('ok')
    expect(layerVerdict(s([1.2e-3, 1.1e-3, 9e-4])).label).toBe('healthy')
  })

  it('stalled when the update ratio is ~0', () => {
    const v = layerVerdict(s([1e-7, 1e-8, 1e-8]))
    expect(v.level).toBe('warn')
    expect(v.label).toBe('stalled')
  })

  it('flags vanishing gradients in the stalled note when grad ≈ 0', () => {
    expect(layerVerdict(s([1e-8, 1e-8], { g: [1e-9] })).note).toMatch(/vanishing/)
  })

  it('exploding when updates exceed the weights', () => {
    expect(layerVerdict(s([2, 3, 5])).level).toBe('error')
    expect(layerVerdict(s([2, 3, 5])).label).toBe('exploding')
  })

  it('diverged on a non-finite weight norm', () => {
    expect(layerVerdict(s([1e-3], { w: [NaN] })).label).toBe('diverged')
  })

  it('neutral before there is any update ratio (epoch 1)', () => {
    expect(layerVerdict(s([])).label).toBe('—')
    expect(layerVerdict(s([])).level).toBe('ok')
  })

  it('judges on the RECENT window, not the whole run (recovered from an early spike)', () => {
    // an early exploding ratio, then settled healthy — the last 3 win
    expect(layerVerdict(s([5, 1e-3, 1.1e-3, 1e-3])).level).toBe('ok')
  })
})

describe('buildHealth (pivot snapshots → per-role/layer series)', () => {
  const ep = (n: number, health: RunEpoch['health']): RunEpoch => ({
    epoch: n,
    epochs: 3,
    metrics: {},
    health,
  })

  it('returns [] when no epoch carries health', () => {
    expect(buildHealth([ep(1, undefined)])).toEqual([])
  })

  it('pivots per role/layer, in the latest snapshot order, with verdicts', () => {
    const epochs: RunEpoch[] = [
      ep(1, { model: { layer_0: { node: 'Conv2d', w: 1.0, g: 0.5 } } }),
      ep(2, { model: { layer_0: { node: 'Conv2d', w: 1.1, dw: 1e-3, g: 0.4 } } }),
    ]
    const [role] = buildHealth(epochs)
    expect(role.role).toBe('model')
    expect(role.layers).toHaveLength(1)
    const l = role.layers[0]
    expect(l.node).toBe('Conv2d')
    expect(l.w).toEqual([1.0, 1.1]) // series across epochs
    expect(l.dw).toEqual([1e-3]) // dw only from epoch 2
    expect(l.verdict.level).toBe('ok')
  })

  it('keeps roles separate (a GAN)', () => {
    const roles = buildHealth([
      ep(1, {
        Generator: { layer_0: { node: 'Linear', w: 1 } },
        Discriminator: { layer_0: { node: 'Linear', w: 1 } },
      }),
    ])
    expect(roles.map((r) => r.role)).toEqual(['Generator', 'Discriminator'])
  })
})

describe('nodeVerdicts (flatten to per-node badges)', () => {
  it('keeps the most severe verdict per node id, skipping unmapped layers', () => {
    const roles = buildHealth([
      {
        epoch: 2,
        epochs: 2,
        metrics: {},
        health: {
          model: {
            layer_0: { node: 'Conv2d', nodeId: 'c1', w: 1, dw: 2 }, // exploding
            layer_1: { node: 'Linear', nodeId: 'fc', w: 1, dw: 1e-3 }, // healthy
            layer_2: { node: 'BatchNorm2d', w: 1, dw: 1e-3 }, // no nodeId → skipped
          },
        },
      },
    ])
    const map = nodeVerdicts(roles)
    expect(map['c1'].level).toBe('error')
    expect(map['fc'].level).toBe('ok')
    expect(Object.keys(map)).toEqual(['c1', 'fc']) // the unmapped layer is absent
  })
})
