import type { DiagnosticCheck } from '../hooks/useReadiness'

export interface ReadinessSummary {
  /** The worst level present — what the strip's glyph and colour follow. */
  level: 'ok' | 'warn' | 'error'
  /** One line: the worst problem in its own words, or a pass count. */
  text: string
  /** How many checks beyond the one being shown still want attention. */
  more: number
}

/**
 * The one-line verdict for the collapsed readiness strip.
 *
 * Worst-first, and quoting the check's own title rather than a count: "class 7
 * has 12 samples, class 3 has 941" is a reason to look, whereas "2 warnings" is
 * a reason to ignore it. The count only appears as the overflow.
 */
export function readinessSummary(checks: DiagnosticCheck[]): ReadinessSummary {
  const errors = checks.filter((c) => c.level === 'error')
  const warns = checks.filter((c) => c.level === 'warn')
  const worst = errors[0] ?? warns[0]
  const attention = errors.length + warns.length

  if (!worst) {
    return {
      level: 'ok',
      text: `${checks.length} ${checks.length === 1 ? 'check' : 'checks'} passed`,
      more: 0,
    }
  }
  return {
    level: errors.length ? 'error' : 'warn',
    text: worst.title,
    more: attention - 1,
  }
}
