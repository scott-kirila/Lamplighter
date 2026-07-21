import { useState } from 'react'
import { useQueryClient } from '@tanstack/react-query'
import { useCheckpoints } from '../hooks/useCheckpoints'
import { extraMetric } from '../lib/evaluation'
import { useRunStore } from '../store/runStore'

// Score the SHOWN run on its held-out test split — data it never trained or
// tuned on. Lives beside the run's config label (this is a fact about that run,
// not about the next one). Uses the kernel's live weights when the shown run IS
// the kernel's, so a run can be scored right after finishing without saving it;
// otherwise it rebuilds the stored run from its own saved weights.
export function EvaluateControl() {
  const runName = useRunStore((s) => s.runName)
  const kernelRunName = useRunStore((s) => s.kernelRunName)
  const { data: metas } = useCheckpoints()
  const queryClient = useQueryClient()
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const meta = (metas ?? []).find((c) => c.name === runName)
  const evaluation = meta?.evaluation ?? null
  const isKernelRun = Boolean(runName) && runName === kernelRunName
  // A stored run needs its weights to be rebuilt; the kernel's run doesn't.
  const canEvaluate = isKernelRun || Boolean(meta?.has_weights)

  const evaluate = async () => {
    if (!runName) return
    setBusy(true)
    setError(null)
    try {
      const url = isKernelRun
        ? '/api/run/evaluate'
        : `/api/checkpoints/${encodeURIComponent(runName)}/evaluate`
      const res = await fetch(url, { method: 'POST' })
      if (!res.ok) {
        const body = await res.json().catch(() => ({}))
        setError(body.detail ?? `couldn't evaluate (HTTP ${res.status})`)
        return
      }
      queryClient.invalidateQueries({ queryKey: ['checkpoints'] })
    } catch {
      setError('backend unreachable')
    } finally {
      setBusy(false)
    }
  }

  if (!runName) return null
  const extra = evaluation ? extraMetric(evaluation) : null
  return (
    <span style={{ display: 'flex', alignItems: 'center', gap: 8, minWidth: 0 }}>
      <button
        onClick={evaluate}
        disabled={busy || !canEvaluate}
        data-tip={
          !canEvaluate
            ? "Evaluating rebuilds the run from its weights — this run saved none, and it isn't the kernel's"
            : 'Score this run on its held-out test split — data it never trained or tuned on'
        }
        style={{
          background: 'none', border: '1px solid var(--border)', borderRadius: 4,
          color: 'var(--text-5)', padding: '2px 9px', fontSize: 11, lineHeight: 1.4,
          cursor: busy || !canEvaluate ? 'default' : 'pointer',
          opacity: canEvaluate ? 1 : 0.45,
          whiteSpace: 'nowrap',
        }}
      >
        {busy ? 'scoring…' : evaluation ? 're-evaluate' : 'evaluate'}
      </button>
      {/* The score is a FACT about the run, so it reads as one rather than
          living inside the button that produced it. */}
      {evaluation && !busy && (
        <span
          data-tip={`Scored on ${evaluation.n} samples of the ${evaluation.split} at ${evaluation.evaluated_at}`}
          style={{ color: 'var(--text-3)', fontSize: 11, whiteSpace: 'nowrap' }}
        >
          test {evaluation.test_loss.toFixed(4)}{extra ? ` · ${extra}` : ''}
        </span>
      )}
      {error && (
        <span style={{ color: 'var(--text-6)', fontSize: 11, overflow: 'hidden', textOverflow: 'ellipsis' }}>
          {error}
        </span>
      )}
    </span>
  )
}
