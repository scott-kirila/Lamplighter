import { beforeEach, describe, expect, it, vi } from 'vitest'
import { applyTheme, initialTheme } from './useTheme'

// jsdom's localStorage/matchMedia are absent or partial here, so stub them.
beforeEach(() => {
  const store = new Map<string, string>()
  vi.stubGlobal('localStorage', {
    getItem: (k: string) => store.get(k) ?? null,
    setItem: (k: string, v: string) => void store.set(k, v),
    removeItem: (k: string) => void store.delete(k),
    clear: () => store.clear(),
  })
  delete (window as unknown as { matchMedia?: unknown }).matchMedia
  delete document.documentElement.dataset.theme
})

describe('initialTheme', () => {
  it('prefers a saved choice over everything', () => {
    localStorage.setItem('lamplighter-theme', 'light')
    expect(initialTheme()).toBe('light')
  })

  it('falls back to the OS preference when nothing is saved', () => {
    window.matchMedia = ((q: string) => ({ matches: q.includes('light') })) as typeof window.matchMedia
    expect(initialTheme()).toBe('light')
  })

  it('defaults to dark with no saved choice and no light preference', () => {
    expect(initialTheme()).toBe('dark')
  })

  it('ignores a garbage saved value', () => {
    localStorage.setItem('lamplighter-theme', 'chartreuse')
    expect(initialTheme()).toBe('dark')
  })
})

describe('applyTheme', () => {
  it('sets data-theme on the document element', () => {
    applyTheme('light')
    expect(document.documentElement.dataset.theme).toBe('light')
  })
})
