import { describe, expect, it } from 'vitest'
import { diffableData, diffableTraining } from './compareDiff'

describe('diffableTraining (what the compare table may diff)', () => {
  it('drops structural keys and keeps loop knobs', () => {
    const flat = diffableTraining({
      recipe: 'gan', roles: { generator: 'g' }, epochs: 50, lr: 0.001,
    })
    expect(flat).toEqual({ epochs: 50, lr: 0.001 })
  })

  it('surfaces per-role params as "<role> <param>" rows — the GAN diff blindspot', () => {
    // Two GAN runs differing only in generator lr must NOT read as identical.
    const flat = diffableTraining({
      per_role: { generator: { lr: 2e-4 }, discriminator: { lr: 1e-4 } },
      epochs: 50,
    })
    expect(flat).toEqual({ epochs: 50, 'generator lr': 2e-4, 'discriminator lr': 1e-4 })
  })

  it('tolerates a malformed per_role without throwing', () => {
    expect(diffableTraining({ per_role: 'bogus' as unknown as object })).toEqual({})
    expect(diffableTraining({ per_role: { generator: null } })).toEqual({})
  })
})

describe('diffableData (the data axis of the compare table)', () => {
  it('prefixes rows (no training-key collisions) and skips the form-only advanced toggle', () => {
    // Two runs differing only in batch size must NOT read as identical config.
    expect(diffableData({ source: 'memory', batch_size: 64, shuffle: false, advanced: false })).toEqual({
      'data source': 'memory', 'data batch_size': 64, 'data shuffle': false,
    })
  })

  it('flattens per-input picks to one sorted row and drops empty strings', () => {
    // x_vars keys are node ids — meaningless as labels; the picked names matter.
    const flat = diffableData({ x_vars: { n2: 'X_b', n1: 'X_a', n3: '' }, x_var: '', y_var: 'y' })
    expect(flat).toEqual({ 'data picks': 'X_a, X_b', 'data y_var': 'y' })
  })

  it('compares lists order-insensitively and drops empty ones', () => {
    // Config order is click order; [A, B] vs [B, A] is not a difference.
    expect(diffableData({ augmentations: ['RandomCrop', 'Grayscale'] }))
      .toEqual({ 'data augmentations': 'Grayscale, RandomCrop' })
    expect(diffableData({ augmentations: [] })).toEqual({})
  })

  it('tolerates a missing payload (an old sidecar, an env run)', () => {
    expect(diffableData(undefined)).toEqual({})
  })
})
