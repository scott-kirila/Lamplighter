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

describe('relative lagging (vanishing gradients across layers)', () => {
  const ep2 = (health: RunEpoch['health']): RunEpoch[] => [{ epoch: 2, epochs: 2, metrics: {}, health }]

  it('flags a layer learning far below the model typical (the 6.8e-5 case)', () => {
    const roles = buildHealth(
      ep2({
        model: {
          layer_0: { node: 'Linear', nodeId: 'a', w: 1, dw: 1e-1 },
          layer_1: { node: 'Linear', nodeId: 'b', w: 1, dw: 8e-2 },
          layer_2: { node: 'Linear', nodeId: 'c', w: 1, dw: 6.8e-5 }, // the vanishing layer
        },
      })
    )
    const byNode = Object.fromEntries(roles[0].layers.map((l) => [l.nodeId, l.verdict]))
    expect(byNode['a'].level).toBe('ok')
    expect(byNode['c'].label).toBe('lagging')
  })

  it('does not flag when the spread is small (all layers similar)', () => {
    const roles = buildHealth(
      ep2({
        model: {
          layer_0: { node: 'L', nodeId: 'a', w: 1, dw: 2e-3 },
          layer_1: { node: 'L', nodeId: 'b', w: 1, dw: 1e-3 },
          layer_2: { node: 'L', nodeId: 'c', w: 1, dw: 1.5e-3 },
        },
      })
    )
    expect(roles[0].layers.every((l) => l.verdict.level === 'ok')).toBe(true)
  })

  it('an exploding layer does not drag its healthy peers into "lagging"', () => {
    const roles = buildHealth(
      ep2({
        model: {
          layer_0: { node: 'L', nodeId: 'a', w: 1, dw: 2 }, // exploding → excluded from the reference
          layer_1: { node: 'L', nodeId: 'b', w: 1, dw: 1e-3 }, // stays healthy
        },
      })
    )
    const byNode = Object.fromEntries(roles[0].layers.map((l) => [l.nodeId, l.verdict]))
    expect(byNode['a'].label).toBe('exploding')
    expect(byNode['b'].level).toBe('ok')
  })
})

describe('the smooth-decay case (real vanishing-gradient run)', () => {
  it('flags the layers ~2+ orders below the fastest, keeps the fast ones healthy', () => {
    // The actual per-layer update ratios from the deep-sigmoid MNIST run.
    const dws = [6.3e-5, 6.8e-5, 3.3e-4, 2.5e-3, 1.9e-2, 1.0e-1]
    const roles = buildHealth([
      {
        epoch: 2,
        epochs: 2,
        metrics: {},
        health: {
          model: Object.fromEntries(
            dws.map((dw, i) => [`layer_${i}`, { node: 'Linear', nodeId: `n${i}`, w: 1, dw }])
          ),
        },
      },
    ])
    const levels = roles[0].layers.map((l) => l.verdict.level)
    // max = 1e-1 → threshold 1e-3: the bottom three (6.3e-5, 6.8e-5, 3.3e-4) lag.
    expect(levels).toEqual(['warn', 'warn', 'warn', 'ok', 'ok', 'ok'])
  })
})
