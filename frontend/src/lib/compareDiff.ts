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
