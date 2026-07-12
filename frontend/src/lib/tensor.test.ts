import { describe, expect, it } from 'vitest'
import { fmtNum, sampleAt, squareSide, tensorKind } from './tensor'

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

  it('image-grid for many-channel maps / higher rank (trailing two dims spatial)', () => {
    expect(tensorKind([5, 8, 8])).toBe('image-grid') // 5-channel feature map
    expect(tensorKind([64, 8, 8])).toBe('image-grid')
    expect(tensorKind([4, 3, 28, 28])).toBe('image-grid') // e.g. video frames
  })

  it('stays bars only when a trailing dim is < 2 (a genuine sequence-of-scalars)', () => {
    expect(tensorKind([16, 1])).toBe('bars')
  })

  it('renders any 2-D field as an image/heatmap (matrices included)', () => {
    expect(tensorKind([100, 3])).toBe('image') // a matrix → a (thin) heatmap, honest
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

describe('squareSide (opt-in flattened-image detection)', () => {
  it('returns the side for a perfect-square vector', () => {
    expect(squareSide([784])).toBe(28) // flattened MNIST
    expect(squareSide([4])).toBe(2)
  })

  it('null for a non-square vector or a lone value', () => {
    expect(squareSide([10])).toBeNull()
    expect(squareSide([1])).toBeNull()
  })

  it('null for anything that is not 1-D (already an image / matrix)', () => {
    expect(squareSide([1, 28, 28])).toBeNull()
    expect(squareSide([28, 28])).toBeNull()
  })
})

describe('fmtNum', () => {
  it('keeps integers, rounds mid values, uses exponent for extremes', () => {
    expect(fmtNum(7)).toBe('7')
    expect(fmtNum(0.1234)).toBe('0.12')
    expect(fmtNum(6.3e-5)).toBe('6.3e-5')
  })
})
