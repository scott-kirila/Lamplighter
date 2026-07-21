import { describe, expect, it } from 'vitest'
import { extraMetric } from './evaluation'
import type { EvaluationResult } from '../hooks/useCheckpoints'

const result = (over: Record<string, number | string> = {}): EvaluationResult => ({
  test_loss: 0.4123,
  n: 200,
  split: 'held-out test split',
  evaluated_at: '2026-07-21T12:00:00',
  ...over,
}) as EvaluationResult

describe('extraMetric (the score beside the loss)', () => {
  it('finds whichever metric the run recorded, without naming one', () => {
    expect(extraMetric(result({ test_acc: 0.9134 }))).toBe('acc 0.913')
    expect(extraMetric(result({ test_mae: 0.05 }))).toBe('mae 0.050')
  })

  it('returns null when the metric was gated off — loss only', () => {
    expect(extraMetric(result())).toBeNull()
  })

  it('never mistakes the loss or the bookkeeping fields for a metric', () => {
    // test_loss is shown separately; n/split/evaluated_at aren't metrics.
    expect(extraMetric(result())).toBeNull()
    expect(extraMetric(null)).toBeNull()
    expect(extraMetric(undefined)).toBeNull()
  })
})
