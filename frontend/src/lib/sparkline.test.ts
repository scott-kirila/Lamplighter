import { describe, expect, it } from 'vitest'

import { sparkBars } from './sparkline'

describe('sparkBars', () => {
  it('draws every value — no truncation at any epoch count', () => {
    for (const epochs of [3, 48, 250, 1000]) {
      const bars = sparkBars(Array(epochs).fill(1), epochs, 360, 13)
      expect(bars).toHaveLength(epochs)
    }
  })

  it('spans the planned epochs so bars grow rightward mid-run', () => {
    // 5 of 10 planned epochs: bars fill the left half only.
    const bars = sparkBars([1, 1, 1, 1, 1], 10, 100, 13)
    expect(bars).toHaveLength(5)
    expect(bars[4].x + bars[4].w).toBe(50)
  })

  it('fills the full width when the run is complete', () => {
    const bars = sparkBars([1, 2, 3, 4], 4, 100, 13)
    expect(bars[0].x).toBe(0)
    expect(bars[3].x + bars[3].w).toBe(100)
  })

  it('skips non-finite slots but keeps their x position (column alignment)', () => {
    // A leading 2-slot pad: the first bar starts a third of the way in.
    const bars = sparkBars([NaN, NaN, 5, 6, 7, 8], 6, 120, 13)
    expect(bars).toHaveLength(4)
    expect(bars[0].x).toBe(40)
    expect(bars[0].i).toBe(2) // slot index survives the skip, for per-epoch colour lookup
  })

  it('scales heights over the finite range with a 1/8 stub at the minimum', () => {
    const bars = sparkBars([0, 10], 2, 100, 16)
    expect(bars[0].h).toBe(2) // min → 1/8 of 16
    expect(bars[1].h).toBe(16) // max → full height
    expect(bars[1].y).toBe(0) // bars sit on the baseline
  })

  it('renders a flat series as uniform stubs without dividing by zero', () => {
    const bars = sparkBars([5, 5, 5], 3, 90, 16)
    expect(bars.every((b) => b.h === 2)).toBe(true)
  })

  it('returns nothing when no slot is finite', () => {
    expect(sparkBars([NaN, NaN], 2, 100, 13)).toEqual([])
  })
})
