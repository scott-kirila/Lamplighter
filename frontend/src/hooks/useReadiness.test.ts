import { describe, expect, it } from 'vitest'
import { runBlocker, type DiagnosticCheck, type Readiness } from './useReadiness'

const err: DiagnosticCheck = { level: 'error', title: "X sample (8) ≠ Input (784)", detail: '' }
const warn: DiagnosticCheck = { level: 'warn', title: 'batch bigger than the split', detail: '' }
const ok: DiagnosticCheck = { level: 'ok', title: 'shapes fit', detail: '' }
const r = (status: Readiness['status'], checks: DiagnosticCheck[] = []): Readiness => ({ status, checks })

describe('runBlocker (the readiness → Run gate)', () => {
  it('blocks on an error check from a fresh diagnose', () => {
    expect(runBlocker(r('ready', [ok, err]))).toBe(err)
  })

  it('returns the FIRST error (fix one, the next surfaces)', () => {
    const err2: DiagnosticCheck = { level: 'error', title: 'nothing picked', detail: '' }
    expect(runBlocker(r('ready', [err, err2]))).toBe(err)
  })

  it('does not block on warnings or all-clear', () => {
    expect(runBlocker(r('ready', [ok, warn]))).toBeUndefined()
    expect(runBlocker(r('ready', []))).toBeUndefined()
  })

  it('fails OPEN when readiness is unavailable, even with a stale error', () => {
    // The safety-critical case: a diagnose failure must not gate on stale
    // checks — Run stays enabled and the backend's start() is the real gate.
    expect(runBlocker(r('unavailable', [err]))).toBeUndefined()
  })

  it('fails open while pending (before the first result)', () => {
    expect(runBlocker(r('pending', [err]))).toBeUndefined()
  })
})
