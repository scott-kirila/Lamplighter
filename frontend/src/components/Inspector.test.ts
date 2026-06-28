import { describe, expect, it } from 'vitest'
import { parseDims } from './Inspector'

describe('parseDims', () => {
  it('splits a comma-separated shape into tokens', () => {
    expect(parseDims('1, 3, 28, 28')).toEqual(['1', '3', '28', '28'])
  })

  it('trims whitespace and drops empty tokens (trailing commas)', () => {
    expect(parseDims(' 1 , 2 ,')).toEqual(['1', '2'])
  })

  it('returns an empty array for an empty string', () => {
    expect(parseDims('')).toEqual([])
  })
})
