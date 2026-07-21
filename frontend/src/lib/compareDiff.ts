// The training config as CompareDiff can diff it: structural keys dropped
// (role→model maps aren't comparable scalars; recipe labels the run
// elsewhere), and per-role params SURFACED as "<role> <param>" rows — a GAN's
// per-role lrs are exactly what its runs differ by, and skipping them
// wholesale made two runs differing only in generator lr read as "identical".
const SKIP = new Set(['roles', 'per_role', 'recipe'])

export function diffableTraining(training: Record<string, unknown>): Record<string, unknown> {
  const out: Record<string, unknown> = {}
  for (const [k, v] of Object.entries(training)) if (!SKIP.has(k)) out[k] = v
  const perRole = training.per_role
  if (perRole && typeof perRole === 'object') {
    for (const [role, params] of Object.entries(perRole as Record<string, unknown>)) {
      if (!params || typeof params !== 'object') continue
      for (const [k, v] of Object.entries(params as Record<string, unknown>)) {
        out[`${role} ${k}`] = v
      }
    }
  }
  return out
}

// The recorded data config as "data <param>" rows (prefixed so batch_size can't
// collide with a training key) — two runs differing only by batch size or
// augmentations must not read as "identical config". `advanced` is a pure
// form-disclosure toggle (no run effect); per-input picks flatten to one sorted
// "data picks" row (their keys are node ids — meaningless as labels); lists
// compare order-insensitively (config order is click order); empty strings and
// empty lists are dropped so an absent key never phantom-diffs an explicit
// empty.
export function diffableData(data: Record<string, unknown> | undefined): Record<string, unknown> {
  const out: Record<string, unknown> = {}
  for (const [k, v] of Object.entries(data ?? {})) {
    if (k === 'advanced') continue
    if (k === 'x_vars') {
      const names = Object.values((v ?? {}) as Record<string, unknown>).map(String).filter(Boolean).sort()
      if (names.length) out['data picks'] = names.join(', ')
      continue
    }
    if (v == null || v === '') continue
    if (Array.isArray(v)) {
      if (v.length) out[`data ${k}`] = v.map(String).sort().join(', ')
      continue
    }
    out[`data ${k}`] = v
  }
  return out
}
