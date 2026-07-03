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

  // Restore repopulates the kernel-side run artifacts; the returned status
  // replaces this tab's run state so the dashboard shows the restored run.
  const restore = async (ckpt: string) => {
    setError(null)
    try {
      const res = await fetch(`/api/checkpoints/${encodeURIComponent(ckpt)}/restore`, {
        method: 'POST',
      })
      const body = await res.json().catch(() => ({}))
      if (!res.ok) {
        setError(body.detail ?? 'could not restore the checkpoint')
        return
      }
      replaceRun(
        body.state,
        body.error ?? null,
        epochsFromHistory(body.history, body.epochs ?? 0),
        body.seed ?? null,
        body.best_epoch ?? null
      )
    } catch {
      setError('backend unreachable')
    }
  }

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
