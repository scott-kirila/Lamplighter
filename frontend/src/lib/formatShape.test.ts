import { describe, expect, it } from 'vitest'
import { formatParamTerms, formatShape } from './formatShape'

describe('formatShape', () => {
  it('renders the leading placeholder batch as N', () => {
    expect(formatShape([1, 128], ' × ')).toBe('N × 128')
    expect(formatShape([1, 3, 28, 28], ', ')).toBe('N, 3, 28, 28')
  })

  it('leaves a non-1 leading dim numeric (e.g. Concat over dim 0)', () => {
    expect(formatShape([2, 784], ' × ')).toBe('2 × 784')
  })

  it('leaves 1-D shapes numeric (no batch to substitute)', () => {
    expect(formatShape([784], ' × ')).toBe('784')
  })

  it('can opt out for pins that are not batch-led (LSTM h_n)', () => {
    expect(formatShape([1, 1, 32], ', ', false)).toBe('1, 1, 32')
  })
})

describe('formatParamTerms', () => {
  it('renders each parameter tensor as a product, summed', () => {
    expect(formatParamTerms([[128, 784], [128]])).toBe('128×784 + 128')
    expect(formatParamTerms([[32, 1, 3, 3], [32]])).toBe('32×1×3×3 + 32')
  })

  it('renders a scalar parameter as 1 and no params as empty', () => {
    expect(formatParamTerms([[]])).toBe('1')
    expect(formatParamTerms([])).toBe('')
  })
})
