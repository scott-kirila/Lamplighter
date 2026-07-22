import { describe, it, expect } from 'vitest'
import { readinessSummary } from './readinessSummary'
import type { DiagnosticCheck } from '../hooks/useReadiness'

const check = (level: DiagnosticCheck['level'], title: string): DiagnosticCheck =>
  ({ level, title, detail: '' })

describe('readinessSummary', () => {
  it('reports a clean pass with a count', () => {
    const s = readinessSummary([check('ok', 'a'), check('ok', 'b')])
    expect(s).toEqual({ level: 'ok', text: '2 checks passed', more: 0 })
  })

  it('singularises a lone check', () => {
    expect(readinessSummary([check('ok', 'a')]).text).toBe('1 check passed')
  })

  // An error blocks Run, so it outranks any number of warnings.
  it('leads with the error even when warnings came first', () => {
    const s = readinessSummary([
      check('ok', 'shapes match'),
      check('warn', 'class imbalance 78:1'),
      check('error', "'y' is not registered"),
    ])
    expect(s.level).toBe('error')
    expect(s.text).toBe("'y' is not registered")
  })

  it('quotes the check rather than counting it', () => {
    const s = readinessSummary([check('warn', 'BatchNorm meets a final batch of 1')])
    expect(s.text).toBe('BatchNorm meets a final batch of 1')
    expect(s.more).toBe(0)
  })

  // The overflow counts only what still wants attention — passing checks are
  // not "more", they're the background.
  it('counts only the remaining problems as overflow', () => {
    const s = readinessSummary([
      check('ok', 'fine'), check('ok', 'also fine'),
      check('warn', 'first problem'), check('warn', 'second'), check('error', 'third'),
    ])
    expect(s.text).toBe('third')
    expect(s.more).toBe(2)
  })

  it('handles an empty list without inventing a verdict', () => {
    expect(readinessSummary([])).toEqual({ level: 'ok', text: '0 checks passed', more: 0 })
  })
})
