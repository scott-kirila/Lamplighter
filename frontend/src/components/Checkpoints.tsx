import { useState } from 'react'
import { useCheckpoints, type CheckpointMeta } from '../hooks/useCheckpoints'
import { epochsFromHistory, useRunStore } from '../store/runStore'

const actionButton: React.CSSProperties = {
  background: 'none',
  border: '1px solid var(--border)',
  borderRadius: 4,
  color: 'var(--text-3)',
  cursor: 'pointer',
  fontFamily: 'monospace',
  fontSize: 11,
  padding: '2px 9px',
  // Keep their size in the narrower side panel — otherwise flex compresses/clips
  // them and the icons look mis-sized. The row wraps instead (see the row style).
  flexShrink: 0,
  lineHeight: 1.4,
}

// Resume continues toward the checkpoint's planned epoch target. Interrupted
// entries (epoch < epochs) finish their plan with one click; finished ones
// need a new, higher target — a small pre-filled input next to the button.
function ResumeControl({
  meta,
  running,
  resume,
}: {
  meta: CheckpointMeta
  running: boolean
  resume: (name: string, epochs?: number) => void
}) {
  const finished = meta.epoch != null && meta.epochs != null && meta.epoch >= meta.epochs
  // Default extension: another full plan on top of what's trained.
  const [target, setTarget] = useState((meta.epoch ?? 0) + (meta.epochs ?? 0))
  const disabled = running || (finished && !(target > (meta.epoch ?? 0)))
  const style = {
    ...actionButton,
    opacity: disabled ? 0.4 : 1,
    cursor: disabled ? 'default' : 'pointer',
  }
  return (
    <>
      {finished && (
        <input
          type="number"
          value={target}
          min={(meta.epoch ?? 0) + 1}
          onChange={(e) => setTarget(Number(e.target.value))}
          title="New total epoch target for the resumed run"
          style={{
            background: 'var(--field)', border: '1px solid var(--border)', borderRadius: 4,
            color: 'var(--text)', fontFamily: 'monospace', fontSize: 11,
            padding: '2px 5px', width: 52,
          }}
        />
      )}
      <button
        onClick={() => resume(meta.name, finished ? target : undefined)}
        disabled={disabled}
        title={
          finished
            ? `Train on toward epoch ${target} (warm start: fresh optimizer, new seed; numbering continues)`
            : `Finish the plan: train the remaining ${(meta.epochs ?? 0) - (meta.epoch ?? 0)} epochs (warm start)`
        }
        style={style}
      >
        ▶ Resume
      </button>
    </>
  )
}

// The session's checkpoint store, as a strip under the run dashboard: name a
// finished run's weights to keep them (persisted with autosave), then restore,
// download, or delete entries. Saves from the notebook (sess.checkpoint) show
// up live via the WS push.
export function Checkpoints({
  compared = [],
  onToggleCompare,
}: {
  // Names currently overlaid on the run charts + the toggle (owned by the
  // Training tab, which renders the charts the overlay lands on).
  compared?: string[]
  onToggleCompare?: (name: string) => void
} = {}) {
  const { data: checkpoints } = useCheckpoints()
  const runState = useRunStore((s) => s.runState)
  const replaceRun = useRunStore((s) => s.replaceRun)
  const [name, setName] = useState('')
  const [error, setError] = useState<string | null>(null)
  // The checkpoint awaiting delete confirmation. An inline confirm (not a
  // blocking window.confirm, which freezes the event loop — and the live charts
  // — until dismissed) since a run may be streaming while you tidy checkpoints.
  const [pendingDelete, setPendingDelete] = useState<string | null>(null)

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

  // Confirmed: a checkpoint's trained weights are deleted from memory AND disk,
  // and undo doesn't cover it — the ⬇ download is the only recovery, so point
  // at it. (The model rebuilds from the canvas; these weights don't.)
  const remove = (ckpt: string) => {
    setPendingDelete(null)
    fetch(`/api/checkpoints/${encodeURIComponent(ckpt)}`, { method: 'DELETE' })
      .then(async (res) => {
        if (!res.ok) {
          const body = await res.json().catch(() => ({}))
          setError(body.detail ?? 'could not delete the checkpoint')
        }
      })
      .catch(() => setError('backend unreachable'))
  }

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
        // Pass the checkpoint's health curve too, so restore/resume seeds the
        // health panel (not just the loss curves) instead of it resetting.
        epochsFromHistory(status.history, status.epochs ?? 0, status.health_history),
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

  // `epochs` is the run's TOTAL target: omitted, an interrupted checkpoint
  // finishes its plan; a finished one needs a higher target to extend.
  const resume = (ckpt: string, epochs?: number) =>
    runStatusPost(
      '/api/run/resume',
      epochs != null ? { name: ckpt, epochs } : { name: ckpt },
      'could not resume from the checkpoint'
    )

  return (
    <div
      style={{
        // A side panel beside the epoch table: fills its resizable panel and
        // scrolls on its own (the divider is the PanelResizeHandle, so no border).
        background: 'var(--panel)',
        padding: '10px 16px',
        fontFamily: 'monospace',
        fontSize: 11,
        height: '100%',
        minWidth: 0,
        overflowY: 'auto',
        boxSizing: 'border-box',
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
          title="Keep the last run's weights under this name (persists across kernel restarts when autosave is on)"
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
          style={{ display: 'flex', alignItems: 'center', gap: 12, paddingTop: 8, flexWrap: 'wrap' }}
        >
          <span style={{ color: 'var(--text)', fontWeight: 600 }}>{c.name}</span>
          <span style={{ color: 'var(--text-6)' }}>{c.created.replace('T', ' ')}</span>
          <span style={{ color: 'var(--text-5)' }}>
            epoch {c.epoch ?? '—'}
            {c.epochs != null && c.epoch != null && c.epoch < c.epochs && ` of ${c.epochs}`}
            {c.best_epoch != null && ` · best @${c.best_epoch}`}
            {c.val_loss != null && ` · val ${c.val_loss.toFixed(4)}`}
          </span>
          <span style={{ marginLeft: 'auto', display: 'flex', gap: 6, alignItems: 'center', flexShrink: 0 }}>
            {onToggleCompare && (
              <button
                onClick={() => onToggleCompare(c.name)}
                title={
                  compared.includes(c.name)
                    ? 'Remove from the comparison overlay'
                    : 'Overlay this run’s curves on the charts'
                }
                style={{
                  ...actionButton,
                  ...(compared.includes(c.name)
                    ? { color: 'var(--accent)', borderColor: 'var(--accent)' }
                    : {}),
                }}
              >
                ⊕ compare
              </button>
            )}
            <ResumeControl
              key={`${c.name}:${c.epoch}:${c.epochs}`}
              meta={c}
              running={running}
              resume={resume}
            />
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
            {pendingDelete === c.name ? (
              <span style={{ display: 'inline-flex', alignItems: 'center', gap: 6 }}>
                <span style={{ color: 'var(--text-6)' }}>delete? weights gone —</span>
                <button
                  onClick={() => remove(c.name)}
                  title="Delete for good (download ⬇ first to keep the weights)"
                  style={{ ...actionButton, color: 'var(--error)' }}
                >
                  yes
                </button>
                <button onClick={() => setPendingDelete(null)} title="Keep it" style={actionButton}>
                  no
                </button>
              </span>
            ) : (
              <button onClick={() => setPendingDelete(c.name)} title="Delete this checkpoint" style={actionButton}>
                ✕
              </button>
            )}
          </span>
        </div>
      ))}
    </div>
  )
}
