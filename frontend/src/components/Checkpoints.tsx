import { useState } from 'react'
import { useCheckpoints } from '../hooks/useCheckpoints'
import { epochsFromHistory, useGraphStore } from '../store/graphStore'

const actionButton: React.CSSProperties = {
  background: 'none',
  border: '1px solid var(--border)',
  borderRadius: 4,
  color: 'var(--text-3)',
  cursor: 'pointer',
  fontFamily: 'monospace',
  fontSize: 11,
  padding: '2px 9px',
}

// The session's checkpoint store, as a strip under the run dashboard: name a
// finished run's weights to keep them (in kernel memory), then restore,
// download, or delete entries. Saves from the notebook (sess.checkpoint) show
// up live via the WS push.
export function Checkpoints() {
  const { data: checkpoints } = useCheckpoints()
  const runState = useGraphStore((s) => s.runState)
  const replaceRun = useGraphStore((s) => s.replaceRun)
  const [name, setName] = useState('')
  const [error, setError] = useState<string | null>(null)

  // A trained model exists after a completed (or stopped) run — including a
  // restored one, whose state is "done".
  const canSave = (runState === 'done' || runState === 'stopped') && name.trim().length > 0
  const running = runState === 'running'

  const save = async () => {
    setError(null)
    try {
      const res = await fetch('/api/checkpoints', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name: name.trim() }),
      })
      if (!res.ok) {
        const body = await res.json().catch(() => ({}))
        setError(body.detail ?? 'could not save the checkpoint')
        return
      }
      setName('') // the list itself updates via the WS push
    } catch {
      setError('backend unreachable')
    }
  }

  const remove = (ckpt: string) =>
    fetch(`/api/checkpoints/${encodeURIComponent(ckpt)}`, { method: 'DELETE' }).catch(() => {})

  // Restore/resume both return a run status whose history seeds this tab's
  // charts wholesale (restore: the stored run as-is; resume: the stored curve
  // preloaded, with the warm-started run's epochs streaming in after it).
  const runStatusPost = async (url: string, body: unknown, failMsg: string) => {
    setError(null)
    try {
      const res = await fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      })
      const status = await res.json().catch(() => ({}))
      if (!res.ok) {
        setError(status.detail ?? failMsg)
        return
      }
      replaceRun(
        status.state,
        status.error ?? null,
        epochsFromHistory(status.history, status.epochs ?? 0),
        status.seed ?? null,
        status.best_epoch ?? null
      )
    } catch {
      setError('backend unreachable')
    }
  }

  const restore = (ckpt: string) =>
    runStatusPost(
      `/api/checkpoints/${encodeURIComponent(ckpt)}/restore`,
      {},
      'could not restore the checkpoint'
    )

  const resume = (ckpt: string) =>
    runStatusPost('/api/run/resume', { name: ckpt }, 'could not resume from the checkpoint')

  return (
    <div
      style={{
        borderTop: '1px solid var(--border)',
        background: 'var(--panel)',
        padding: '10px 16px',
        fontFamily: 'monospace',
        fontSize: 11,
        flexShrink: 0,
        maxHeight: 180,
        overflowY: 'auto',
      }}
    >
      <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
        <span style={{ textTransform: 'uppercase', letterSpacing: 1, color: 'var(--text-4)' }}>
          Checkpoints
        </span>
        <input
          value={name}
          onChange={(e) => setName(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && canSave && save()}
          placeholder="name"
          style={{
            background: 'var(--field)',
            border: '1px solid var(--border)',
            borderRadius: 4,
            color: 'var(--text)',
            fontFamily: 'monospace',
            fontSize: 11,
            padding: '3px 8px',
            width: 140,
          }}
        />
        <button
          onClick={save}
          disabled={!canSave}
          title="Keep the last run's weights under this name (in kernel memory)"
          style={{ ...actionButton, opacity: canSave ? 1 : 0.4, cursor: canSave ? 'pointer' : 'default' }}
        >
          ＋ Save
        </button>
        {error && <span style={{ color: 'var(--error)' }}>✗ {error}</span>}
        {(checkpoints ?? []).length === 0 && !error && (
          <span style={{ color: 'var(--text-6)' }}>
            no checkpoints yet — save a finished run to restore it later
          </span>
        )}
      </div>

      {(checkpoints ?? []).map((c) => (
        <div
          key={c.name}
          style={{ display: 'flex', alignItems: 'center', gap: 12, paddingTop: 8 }}
        >
          <span style={{ color: 'var(--text)', fontWeight: 600 }}>{c.name}</span>
          <span style={{ color: 'var(--text-6)' }}>{c.created.replace('T', ' ')}</span>
          <span style={{ color: 'var(--text-5)' }}>
            epoch {c.epoch ?? '—'}
            {c.best_epoch != null && ` · best @${c.best_epoch}`}
            {c.val_loss != null && ` · val ${c.val_loss.toFixed(4)}`}
          </span>
          <span style={{ marginLeft: 'auto', display: 'flex', gap: 6, alignItems: 'center' }}>
            <button
              onClick={() => resume(c.name)}
              disabled={running}
              title="Train further from this checkpoint (warm start: fresh optimizer, new seed; epoch numbering continues)"
              style={{ ...actionButton, opacity: running ? 0.4 : 1, cursor: running ? 'default' : 'pointer' }}
            >
              ▶ Resume
            </button>
            <button
              onClick={() => restore(c.name)}
              disabled={running}
              title="Load this checkpoint as the current run (weights, history, snapshot)"
              style={{ ...actionButton, opacity: running ? 0.4 : 1, cursor: running ? 'default' : 'pointer' }}
            >
              Restore
            </button>
            <a
              href={`/api/checkpoints/${encodeURIComponent(c.name)}/weights`}
              title="Download as a .pt file (load with lamplighter.load_checkpoint)"
              style={{ ...actionButton, textDecoration: 'none' }}
            >
              ⬇
            </a>
            <button onClick={() => remove(c.name)} title="Delete this checkpoint" style={actionButton}>
              ✕
            </button>
          </span>
        </div>
      ))}
    </div>
  )
}
