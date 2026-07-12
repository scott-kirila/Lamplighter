import { describe, expect, it } from 'vitest'
import { buildHealth, concernColor, concernScore, nodeHealth } from './useTrainingHealth'
import type { RunEpoch } from '../store/runStore'

const s = (dw: number[], extra: { w?: number[]; g?: number[] } = {}) => ({
  w: extra.w ?? dw.map(() => 1),
  dw,
  g: extra.g ?? [],
})

describe('concernScore (continuous 0→1, no labels)', () => {
  it('~0 for the fastest layer (ratio ≈ 1)', () => {
    expect(concernScore(s([1e-1]), 1e-1).concern).toBeCloseTo(0, 5)
  })

  it('rises to 1 for a layer ~2+ orders below the fastest (vanishing)', () => {
    expect(concernScore(s([6.3e-5]), 1e-1).concern).toBe(1)
  })

  it('lands in the amber middle for a borderline layer', () => {
    const c = concernScore(s([2.5e-3]), 1e-1).concern!
    expect(c).toBeGreaterThan(0.4)
    expect(c).toBeLessThan(0.75)
  })

  it('rises toward 1 as updates approach the weights (exploding)', () => {
    expect(concernScore(s([1.5]), 1).concern).toBe(1)
  })

  it('is 1 on a non-finite weight norm', () => {
    expect(concernScore(s([1e-3], { w: [NaN] })).concern).toBe(1)
  })

  it('is null before there is any update ratio (epoch 1)', () => {
    expect(concernScore(s([])).concern).toBeNull()
  })

  it('notes the factual context (no verdict word)', () => {
    const { note } = concernScore(s([6.3e-5]), 1e-1)
    expect(note).toMatch(/Δw\/w/)
    expect(note).toMatch(/below the fastest layer/)
    expect(note).not.toMatch(/lagging|stalled|healthy/) // colors carry the reading, not words
  })
})

describe('concernColor (green → yellow → red)', () => {
  it('greens at 0, yellows at 0.5, reds at 1', () => {
    expect(concernColor(0)).toBe('hsl(120, 70%, 45%)')
    expect(concernColor(0.5)).toBe('hsl(60, 70%, 45%)')
    expect(concernColor(1)).toBe('hsl(0, 70%, 45%)')
  })
})

describe('buildHealth on the real vanishing-gradient run', () => {
  const dws = [6.3e-5, 6.8e-5, 3.3e-4, 2.5e-3, 1.9e-2, 1.0e-1]
  const roles = buildHealth([
    {
      epoch: 2,
      epochs: 2,
      metrics: {},
      health: {
        model: Object.fromEntries(dws.map((dw, i) => [`layer_${i}`, { node: 'Linear', nodeId: `n${i}`, w: 1, dw }])),
      },
    },
  ])

  it('scores the early (slow) layers red and the output layer green', () => {
    const c = roles[0].layers.map((l) => l.concern!)
    expect(c[0]).toBe(1) // 6.3e-5 → deep red
    expect(c[5]).toBeCloseTo(0, 5) // 1.0e-1 (fastest) → green
    // monotonic: concern never increases as the layers get faster
    for (let i = 1; i < c.length; i++) expect(c[i]).toBeLessThanOrEqual(c[i - 1] + 1e-9)
  })
})

describe('nodeHealth (worst concern per node)', () => {
  const ep = (health: RunEpoch['health']): RunEpoch => ({ epoch: 2, epochs: 2, metrics: {}, health })

  it('keeps the highest concern per node id, skips unmapped/no-data layers', () => {
    const roles = buildHealth([
      ep({
        model: {
          layer_0: { node: 'Linear', nodeId: 'a', w: 1, dw: 6e-5 }, // red
          layer_1: { node: 'Linear', nodeId: 'b', w: 1, dw: 1e-1 }, // green
          layer_2: { node: 'BatchNorm2d', w: 1, dw: 1e-1 }, // no nodeId → skipped
        },
      }),
    ])
    const map = nodeHealth(roles)
    expect(map['a'].concern).toBe(1)
    expect(map['b'].concern).toBeCloseTo(0, 5)
    expect(Object.keys(map)).toEqual(['a', 'b'])
  })
})
