import { describe, expect, it } from 'vitest'
import { fmtNum, sampleAt, tensorKind } from './tensor'

describe('tensorKind (render by shape alone)', () => {
  it('scalar for a lone value', () => {
    expect(tensorKind([])).toBe('scalar')
    expect(tensorKind([1])).toBe('scalar')
  })

  it('image for HxW / 1xHxW / 3xHxW', () => {
    expect(tensorKind([28, 28])).toBe('image')
    expect(tensorKind([1, 28, 28])).toBe('image')
    expect(tensorKind([3, 32, 32])).toBe('image')
  })

  it('bars for a vector', () => {
    expect(tensorKind([10])).toBe('bars')
  })

  it('bars for a non-standard channel count (not 1 or 3)', () => {
    expect(tensorKind([5, 8, 8])).toBe('bars')
  })
})

describe('sampleAt (slice one example from a batch)', () => {
  it('strips the batch dim and takes the right window', () => {
    const t = { shape: [2, 3], data: [1, 2, 3, 4, 5, 6] }
    expect(sampleAt(t, 0)).toEqual({ shape: [3], data: [1, 2, 3] })
    expect(sampleAt(t, 1)).toEqual({ shape: [3], data: [4, 5, 6] })
  })

  it('handles a scalar batch [n]', () => {
    expect(sampleAt({ shape: [3], data: [7, 8, 9] }, 1)).toEqual({ shape: [], data: [8] })
  })

  it('handles an image batch', () => {
    const t = { shape: [1, 1, 2, 2], data: [1, 2, 3, 4] }
    expect(sampleAt(t, 0)).toEqual({ shape: [1, 2, 2], data: [1, 2, 3, 4] })
  })
})

describe('fmtNum', () => {
  it('keeps integers, rounds mid values, uses exponent for extremes', () => {
    expect(fmtNum(7)).toBe('7')
    expect(fmtNum(0.1234)).toBe('0.12')
    expect(fmtNum(6.3e-5)).toBe('6.3e-5')
  })
})
