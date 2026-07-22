// Mirrors the store's inline `activeTab` union (graphStore.ts). Narrowed to
// what a URL may select, so this file needs no import and the store keeps
// owning the full set.
type UrlTab = 'training'

// Tabs a URL is allowed to select. Deliberately not every tab value: `model`
// would need a model id to mean anything, and `overview` is already the
// default, so honoring them would add ways to fail without adding reach.
const URL_SELECTABLE = ['training'] as const

/**
 * The tab a fresh load should open on, from `?tab=…` — or null to leave the
 * store's default alone.
 *
 * `lamplighter.demo()` opens `?tab=training` so the first thing a new user sees
 * is an armed Run button rather than a canvas they have to find their way
 * around. Anything unrecognized is ignored rather than erroring: a URL is a
 * hint from outside the app, not a command, and a stale or hand-edited one
 * should degrade to the normal landing.
 */
export function initialTabFromUrl(search: string): UrlTab | null {
  let value: string | null
  try {
    value = new URLSearchParams(search).get('tab')
  } catch {
    return null
  }
  if (!value) return null
  const match = URL_SELECTABLE.find((t) => t === value.toLowerCase())
  return match ?? null
}
