import { describe, expect, it } from 'vitest'
import { diffableTraining } from './compareDiff'

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
