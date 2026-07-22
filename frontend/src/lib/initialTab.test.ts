import { describe, it, expect } from 'vitest'
import { initialTabFromUrl } from './initialTab'

describe('initialTabFromUrl', () => {
  it('selects the training tab, which is what demo() opens', () => {
    expect(initialTabFromUrl('?tab=training')).toBe('training')
    expect(initialTabFromUrl('?foo=1&tab=training')).toBe('training')
    expect(initialTabFromUrl('?tab=TRAINING')).toBe('training')
  })

  it('leaves the default alone when there is nothing to honor', () => {
    expect(initialTabFromUrl('')).toBeNull()
    expect(initialTabFromUrl('?')).toBeNull()
    expect(initialTabFromUrl('?tab=')).toBeNull()
    expect(initialTabFromUrl('?other=training')).toBeNull()
  })

  // A URL is a hint from outside the app, not a command — a stale or
  // hand-edited one should land normally rather than break the load.
  it('ignores tabs a URL may not select, and junk', () => {
    expect(initialTabFromUrl('?tab=model')).toBeNull()      // needs a model id to mean anything
    expect(initialTabFromUrl('?tab=overview')).toBeNull()   // already the default
    expect(initialTabFromUrl('?tab=../../etc')).toBeNull()
    expect(initialTabFromUrl('?tab=%E0%A4%A')).toBeNull()   // malformed percent-encoding
  })
})
