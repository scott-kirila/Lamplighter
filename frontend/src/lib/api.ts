// A thrown API failure carrying a human message — the backend's `detail` when it
// sent one, a per-call `fallback` otherwise, and 'backend unreachable' when the
// fetch itself failed. Every mutation surfaces exactly these two shapes, so
// callers can show `err.message` without re-parsing the response.
export class ApiError extends Error {}

// POST (default) or DELETE JSON, returning the parsed body. Throws `ApiError` on
// a non-2xx (message = `detail` ?? `fallback`) and on a network failure. This is
// the fetch + `if (!res.ok) throw` + JSON-parse boilerplate that every write call
// site used to hand-roll.
export async function apiFetch<T = unknown>(
  url: string,
  opts: { method?: string; body?: unknown; fallback?: string } = {}
): Promise<T> {
  let res: Response
  try {
    res = await fetch(url, {
      method: opts.method ?? 'POST',
      ...(opts.body !== undefined
        ? { headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(opts.body) }
        : {}),
    })
  } catch {
    throw new ApiError('backend unreachable')
  }
  const data = await res.json().catch(() => ({}))
  if (!res.ok) {
    throw new ApiError((data as { detail?: string }).detail ?? opts.fallback ?? `request failed (${res.status})`)
  }
  return data as T
}
