import type { EvaluationResult } from '../hooks/useCheckpoints'

// The metric a test result carries BESIDES the loss ("test_acc" → "acc 0.913"),
// read generically: which key exists depends on the run's own recipe and loss
// (accuracy for a classifier, MAE for a regressor, none when the metric was
// gated off), so nothing here may name a specific one.
export function extraMetric(e: EvaluationResult | null | undefined): string | null {
  if (!e) return null
  const key = Object.keys(e).find(
    (k) => k.startsWith('test_') && k !== 'test_loss' && typeof e[k] === 'number'
  )
  return key ? `${key.slice(5)} ${(e[key] as number).toFixed(3)}` : null
}
