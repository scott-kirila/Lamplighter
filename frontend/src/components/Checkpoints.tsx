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

// Resume continues toward a TOTAL epoch target. The target input is ALWAYS
// rendered — for an interrupted run it defaults to finishing its own plan,
// for a finished one to another full plan on top — so the control never
// changes shape by run state (reserve the slot, vary the value), and an
// interrupted run can be extended in the same step as finishing it.
function ResumeControl({
  meta,
  running,
  resume,
  enabled = true,
}: {
  meta: CheckpointMeta
  running: boolean
  resume: (name: string, epochs?: number) => void
  // False for a weightless run: the slot renders (layout never changes by run
  // state) but both controls are inert — resume needs weights.
  enabled?: boolean
}) {
  const finished = meta.epoch != null && meta.epochs != null && meta.epoch >= meta.epochs
  const [target, setTarget] = useState(
    finished ? (meta.epoch ?? 0) + (meta.epochs ?? 0) : meta.epochs ?? 0
  )
  const disabled = running || !enabled || !(target > (meta.epoch ?? 0))
  const style = {
    ...actionButton,
    // Fill the rest of the fixed-width action column so Resume sits flush with
    // the full-width buttons above/below it (the input keeps its 52px).
    flex: 1,
    minWidth: 0,
    opacity: disabled ? 0.4 : 1,
    cursor: disabled ? 'default' : 'pointer',
  }
  return (
    <>
      <input
        type="number"
        value={target}
        min={(meta.epoch ?? 0) + 1}
        disabled={!enabled}
        onChange={(e) => setTarget(Number(e.target.value))}
        title="Total epoch target for the resumed run — its own plan by default; raise it to train further"
        style={{
          background: 'var(--field)', border: '1px solid var(--border)', borderRadius: 4,
          color: 'var(--text)', fontFamily: 'monospace', fontSize: 11,
          padding: '2px 5px', width: 52, opacity: enabled ? 1 : 0.4, boxSizing: 'border-box',
        }}
      />
      <button
        onClick={() => resume(meta.name, target)}
        disabled={disabled}
        title={`Train toward epoch ${target} (warm start: fresh optimizer, new seed; numbering continues)`}
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
  embedded = false,
}: {
  // Names currently overlaid on the run charts + the toggle (owned by the
  // Training tab, which renders the charts the overlay lands on).
  compared?: string[]
  onToggleCompare?: (name: string) => void
  // Inside the side pane's Runs accordion: the section header owns the title,
  // so the strip renders bodies only.
  embedded?: boolean
} = {}) {
  const { data: checkpoints } = useCheckpoints()
  const runState = useRunStore((s) => s.runState)
  const shownRun = useRunStore((s) => s.runName)
  const kernelRun = useRunStore((s) => s.kernelRunName)
  const replaceRun = useRunStore((s) => s.replaceRun)
  const [renaming, setRenaming] = useState<{ name: string; value: string } | null>(null)
  const [error, setError] = useState<string | null>(null)
  // The checkpoint awaiting delete confirmation. An inline confirm (not a
  // blocking window.confirm, which freezes the event loop — and the live charts
  // — until dismissed) since a run may be streaming while you tidy checkpoints.
  const [pendingDelete, setPendingDelete] = useState<string | null>(null)

  const running = runState === 'running'

  // "Keep weights" upgrades a run's auto record with the kernel's weights —
  // only offered on the row of the run the kernel actually holds.
  const keepWeights = async (runName: string) => {
    setError(null)
    try {
      const res = await fetch('/api/checkpoints', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name: runName }),
      })
      if (!res.ok) {
        const body = await res.json().catch(() => ({}))
        setError(body.detail ?? 'could not keep the weights')
      }
    } catch {
      setError('backend unreachable')
    }
  }

  // Show a stored run on the dashboard — read-only; the kernel's model and
  // current run are untouched (restore stays the explicit weights action).
  const view = async (runName: string) => {
    setError(null)
    try {
      const res = await fetch(`/api/checkpoints/${encodeURIComponent(runName)}/view`)
      const status = await res.json().catch(() => ({}))
      if (!res.ok) {
        setError(status.detail ?? 'could not load the run')
        return
      }
      replaceRun(
        status.state,
        status.error ?? null,
        epochsFromHistory(status.history, status.epochs ?? 0, status.health_history),
        status.seed ?? null,
        status.best_epoch ?? null,
        status.steps ?? [],
        status.step_total ?? 0,
        status.config ?? null,
        runName
      )
    } catch {
      setError('backend unreachable')
    }
  }

  const submitRename = async () => {
    if (!renaming) return
    const { name: oldName, value } = renaming
    setRenaming(null)
    if (!value.trim() || value.trim() === oldName) return
    setError(null)
    try {
      const res = await fetch(`/api/checkpoints/${encodeURIComponent(oldName)}/rename`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name: value.trim() }),
      })
      if (!res.ok) {
        const body = await res.json().catch(() => ({}))
        setError(body.detail ?? 'could not rename the run')
      }
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
        status.best_epoch ?? null,
        // Its step curve and recorded config too — the restored run wears its
        // own chips and step-resolution loss, not the previous run's leftovers.
        status.steps ?? [],
        status.step_total ?? 0,
        status.config ?? null
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
        background: 'var(--panel)',
        padding: embedded ? '0 16px 12px' : '10px 16px',
        fontFamily: 'monospace',
        fontSize: 11,
        minWidth: 0,
        boxSizing: 'border-box',
        ...(embedded ? {} : { height: '100%', overflowY: 'auto' }),
      }}
    >
      <div style={{ display: 'flex', alignItems: 'center', gap: 10, flexWrap: 'wrap' }}>
        {!embedded && (
          <span style={{ textTransform: 'uppercase', letterSpacing: 1, color: 'var(--text-4)' }}>
            Runs
          </span>
        )}
        {error && <span style={{ color: 'var(--error)' }}>✗ {error}</span>}
        {(checkpoints ?? []).length === 0 && !running && !error && (
          <span style={{ color: 'var(--text-6)' }}>
            every run records here — keep weights on the ones worth resuming
          </span>
        )}
      </div>

      {/* The live run, before its record lands at run end. */}
      {running && shownRun && !(checkpoints ?? []).some((c) => c.name === shownRun) && (
        <div style={{ padding: '8px 0', borderBottom: '1px solid var(--border)', display: 'flex', flexDirection: 'column', gap: 3 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <span style={{ width: 10, flexShrink: 0, textAlign: 'center', color: 'var(--warn)' }}>▶</span>
            <span style={{ color: 'var(--text)', fontWeight: 600 }}>{shownRun}</span>
          </div>
          <span style={{ color: 'var(--warn)', paddingLeft: 18 }}>running…</span>
        </div>
      )}

      {[...(checkpoints ?? [])].reverse().map((c) => {
        const hasWeights = c.has_weights ?? true
        const state = c.state ?? 'done'
        return (
        <div
          key={c.name}
          style={{
            // A left accent marks the run shown on the dashboard. The border is
            // reserved (transparent) on every row and the negative margin is
            // uniform, so the marker appears/clears without shifting the layout.
            padding: '8px 0 8px 13px',
            marginLeft: -16,
            borderLeft: `3px solid ${shownRun === c.name ? 'var(--accent)' : 'transparent'}`,
            borderBottom: '1px solid var(--border)',
            display: 'flex',
            gap: 12,
            alignItems: 'flex-start',
          }}
        >
          {/* The run's facts, one per line — the pane is tall, not wide. */}
          <div style={{ flex: 1, minWidth: 0, display: 'flex', flexDirection: 'column', gap: 3 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          {/* A fixed-width state slot, ALWAYS rendered, so no state ever
              shifts the name/date — the layout holds still and only the
              glyph changes. Green = the health scale's "fine" green. */}
          <span
            title={state}
            style={{
              width: 10, flexShrink: 0, textAlign: 'center',
              color:
                state === 'failed'
                  ? 'var(--error)'
                  : state === 'stopped'
                    ? 'var(--text-5)'
                    : 'hsl(120, 70%, 45%)',
            }}
          >
            {state === 'failed' ? '✕' : state === 'stopped' ? '■' : '✓'}
          </span>
          {renaming?.name === c.name ? (
            <input
              autoFocus
              value={renaming.value}
              onChange={(e) => setRenaming({ name: c.name, value: e.target.value })}
              onKeyDown={(e) => {
                if (e.key === 'Enter') submitRename()
                if (e.key === 'Escape') setRenaming(null)
              }}
              onBlur={submitRename}
              style={{
                background: 'var(--field)', border: '1px solid var(--border)', borderRadius: 4,
                color: 'var(--text)', fontFamily: 'monospace', fontSize: 11, padding: '2px 6px', width: 120,
              }}
            />
          ) : (
            <button
              onClick={() => !running && view(c.name)}
              onDoubleClick={() => setRenaming({ name: c.name, value: c.name })}
              title={
                running
                  ? 'A run is streaming — it owns the dashboard until it finishes (double-click to rename)'
                  : 'Show this run on the dashboard (double-click to rename)'
              }
              style={{
                background: 'none', border: 'none', padding: 0, cursor: running ? 'default' : 'pointer',
                color: shownRun === c.name ? 'var(--accent)' : 'var(--text)',
                fontFamily: 'monospace', fontSize: 11, fontWeight: 600,
              }}
            >
              {c.name}
            </button>
          )}
          </div>
          <span style={{ color: 'var(--text-6)', paddingLeft: 18 }}>{c.created.replace('T', ' ')}</span>
          <span style={{ color: 'var(--text-5)', paddingLeft: 18 }}>
            epoch {c.epoch ?? '—'}
            {c.epochs != null && c.epoch != null && c.epoch < c.epochs && ` of ${c.epochs}`}
          </span>
          {(c.best_epoch != null || c.val_loss != null) && (
            <span style={{ color: 'var(--text-5)', paddingLeft: 18 }}>
              {c.best_epoch != null && `best @${c.best_epoch}`}
              {c.best_epoch != null && c.val_loss != null && ' · '}
              {c.val_loss != null && `val ${c.val_loss.toFixed(4)}`}
            </span>
          )}
          </div>
          {/* Actions: every row renders the SAME five slots at a fixed width —
              no button changes size or neighbours by run state. Weights-needing
              actions render disabled (with the reason) when the run kept none,
              and the top slot tells the weights story either way. */}
          <span style={{ display: 'flex', flexDirection: 'column', gap: 4, width: 132, flexShrink: 0 }}>
            {/* Slot 1: keep — an affordance only while the kernel still holds
                THIS run's weights; a status label otherwise. */}
            {!hasWeights && kernelRun === c.name && !running ? (
              <button
                onClick={() => keepWeights(c.name)}
                title="Keep this run's weights (the kernel still holds them) — enables restore/resume/download"
                style={{ ...actionButton, width: '100%', color: 'var(--accent)', borderColor: 'var(--accent)' }}
              >
                ＋ keep weights
              </button>
            ) : (
              <span
                title={
                  hasWeights
                    ? 'The weights are stored — restore, resume, and download are available'
                    : 'Only the curves were recorded; the kernel no longer holds these weights'
                }
                style={{
                  ...actionButton,
                  width: '100%', border: '1px solid transparent', cursor: 'default',
                  color: 'var(--text-6)', textAlign: 'center', boxSizing: 'border-box',
                }}
              >
                {hasWeights ? 'weights kept ✓' : 'weights not kept'}
              </span>
            )}
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
                  width: '100%',
                  ...(compared.includes(c.name)
                    ? { color: 'var(--accent)', borderColor: 'var(--accent)' }
                    : {}),
                }}
              >
                ⊕ compare
              </button>
            )}
            <span
              style={{ display: 'flex', gap: 4, alignItems: 'center' }}
              title={hasWeights ? undefined : 'Resume needs weights — this run kept none'}
            >
              <ResumeControl
                key={`${c.name}:${c.epoch}:${c.epochs}`}
                meta={c}
                running={running}
                enabled={hasWeights}
                resume={resume}
              />
            </span>
            <button
              onClick={() => hasWeights && restore(c.name)}
              disabled={running || !hasWeights}
              title={
                hasWeights
                  ? 'Load this run as the current one (weights into the kernel, history, snapshot)'
                  : 'Restore needs weights — this run kept none'
              }
              style={{
                ...actionButton, width: '100%',
                opacity: running || !hasWeights ? 0.4 : 1,
                cursor: running || !hasWeights ? 'default' : 'pointer',
              }}
            >
              Restore
            </button>
            {pendingDelete === c.name ? (
              <span
                title={hasWeights ? 'The weights are gone for good — download ⬇ first to keep them' : 'This run record is gone for good'}
                style={{ display: 'flex', gap: 4, alignItems: 'center' }}
              >
                <span style={{ color: 'var(--text-6)', flex: 1 }}>sure?</span>
                <button onClick={() => remove(c.name)} style={{ ...actionButton, color: 'var(--error)' }}>
                  yes
                </button>
                <button onClick={() => setPendingDelete(null)} title="Keep it" style={actionButton}>
                  no
                </button>
              </span>
            ) : (
              <span style={{ display: 'flex', gap: 4 }}>
                <a
                  href={hasWeights ? `/api/checkpoints/${encodeURIComponent(c.name)}/weights` : undefined}
                  title={
                    hasWeights
                      ? 'Download as a .pt file (load with lamplighter.load_checkpoint)'
                      : 'Download needs weights — this run kept none'
                  }
                  style={{
                    ...actionButton, textDecoration: 'none', flex: 1, textAlign: 'center',
                    ...(hasWeights ? {} : { opacity: 0.4, cursor: 'default', pointerEvents: 'none' }),
                  }}
                >
                  ⬇
                </a>
                <button
                  onClick={() => setPendingDelete(c.name)}
                  title="Delete this run"
                  style={{ ...actionButton, flex: 1 }}
                >
                  ✕
                </button>
              </span>
            )}
          </span>
        </div>
        )
      })}
    </div>
  )
}
