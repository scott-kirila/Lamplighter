import { useState } from 'react'
import { createPortal } from 'react-dom'
import { useCheckpoints, type CheckpointMeta } from '../hooks/useCheckpoints'
import { useCheckpointActions } from '../hooks/useCheckpointActions'
import { epochsFromHistory, useRunStore } from '../store/runStore'
import { button, chip, eyebrow, field } from '../styles/ui'

// The row's action buttons — the shared ghost button, but flex-pinned so the
// narrower side panel can't compress/clip them (the row wraps instead).
const actionButton: React.CSSProperties = { ...button, flexShrink: 0 }

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
        style={{ ...field, padding: '2px 5px', width: 52, opacity: enabled ? 1 : 0.4 }}
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
  const { save, rename, remove: removeCkpt } = useCheckpointActions()
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
  // The run the pointer is over — a whole-row hover cue, since the row is
  // click-to-view.
  const [hovered, setHovered] = useState<string | null>(null)
  // A pending restore/resume held back because it would discard the kernel's
  // unsaved live model. `run` fires it once the user decides; `kernelName` is
  // the at-risk run to optionally save first.
  const [pendingSwap, setPendingSwap] = useState<
    { run: () => void; kernelName: string; target: string } | null
  >(null)

  const running = runState === 'running'

  // "Keep weights" upgrades a run's auto record with the kernel's weights —
  // only offered on the row of the run the kernel actually holds.
  const keepWeights = async (runName: string): Promise<boolean> => {
    setError(null)
    try {
      await save.mutateAsync(runName)
      return true
    } catch (e) {
      setError(e instanceof Error ? e.message : 'could not save the weights')
      return false
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
      await rename.mutateAsync({ name: oldName, to: value.trim() })
    } catch (e) {
      setError(e instanceof Error ? e.message : 'could not rename the run')
    }
  }

  // Confirmed: a checkpoint's trained weights are deleted from memory AND disk,
  // and undo doesn't cover it — the ⬇ download is the only recovery, so point
  // at it. (The model rebuilds from the canvas; these weights don't.)
  const remove = (ckpt: string) => {
    setPendingDelete(null)
    setError(null)
    removeCkpt.mutate(ckpt, { onError: (e) => setError(e.message) })
  }

  // Restore/resume both return a run status whose history seeds this tab's
  // charts wholesale (restore: the stored run as-is; resume: the stored curve
  // preloaded, with the warm-started run's epochs streaming in after it).
  // `shownName` is the run the dashboard now shows, so its row gets the accent:
  // the restored run for restore; the new warm-started run for resume (its
  // reserved name rides the returned status).
  const runStatusPost = async (url: string, body: unknown, failMsg: string, shownName?: string) => {
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
        status.config ?? null,
        shownName ?? status.run_name ?? null,
        // Restore/resume both change the run the kernel holds — hand its name to
        // the store so "keep weights" tracks the live model to the right row.
        status.run_name ?? null
      )
    } catch {
      setError('backend unreachable')
    }
  }

  // The kernel's live model, when it's a run whose weights AREN'T saved — its
  // in-memory weights vanish the moment a restore/resume replaces it. (A saved
  // live model loses nothing; no live model, nothing to lose.)
  const liveUnsaved =
    kernelRun && (checkpoints ?? []).some((c) => c.name === kernelRun && c.has_weights === false)
      ? kernelRun
      : null

  // Restore/resume both swap the live model. If that would drop an unsaved one,
  // hold the action and ask; otherwise run it straight away.
  const guardSwap = (run: () => void, target: string) => {
    if (liveUnsaved && liveUnsaved !== target) {
      setPendingSwap({ run, kernelName: liveUnsaved, target })
    } else {
      run()
    }
  }

  const confirmSwap = async (save: boolean) => {
    const p = pendingSwap
    if (!p) return
    setPendingSwap(null)
    // Save while the kernel still holds it (before the swap). If the save
    // fails, keep the model — don't discard on a false promise of safety.
    if (save && !(await keepWeights(p.kernelName))) return
    p.run()
  }

  const restore = (ckpt: string) =>
    guardSwap(
      () =>
        runStatusPost(
          `/api/checkpoints/${encodeURIComponent(ckpt)}/restore`,
          {},
          'could not restore the checkpoint',
          ckpt
        ),
      ckpt
    )

  // `epochs` is the run's TOTAL target: omitted, an interrupted checkpoint
  // finishes its plan; a finished one needs a higher target to extend.
  const resume = (ckpt: string, epochs?: number) =>
    guardSwap(
      () =>
        runStatusPost(
          '/api/run/resume',
          epochs != null ? { name: ckpt, epochs } : { name: ckpt },
          'could not resume from the checkpoint'
        ),
      ckpt
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
          <span style={{ ...eyebrow, color: 'var(--text-4)' }}>Runs</span>
        )}
        {error && <span style={{ color: 'var(--error)' }}>✗ {error}</span>}
        {(checkpoints ?? []).length === 0 && !running && !error && (
          <span style={{ color: 'var(--text-6)' }}>
            every run records here — save weights on the ones worth resuming
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
        const hasWeights = c.has_weights
        const state = c.state ?? 'done'
        return (
        <div
          key={c.name}
          // The whole row shows the run on the dashboard — a lightweight,
          // read-only view (works for weightless runs too). The action column
          // stops propagation so its buttons stay their own targets. A live run
          // owns the dashboard, so rows are inert then.
          onClick={() => !running && view(c.name)}
          onMouseEnter={() => setHovered(c.name)}
          onMouseLeave={() => setHovered((h) => (h === c.name ? null : h))}
          title={running ? undefined : 'Show this run on the dashboard'}
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
            cursor: running ? 'default' : 'pointer',
            background: hovered === c.name && !running ? 'var(--field)' : 'transparent',
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
              onClick={(e) => e.stopPropagation()}
              onChange={(e) => setRenaming({ name: c.name, value: e.target.value })}
              onKeyDown={(e) => {
                if (e.key === 'Enter') submitRename()
                if (e.key === 'Escape') setRenaming(null)
              }}
              onBlur={submitRename}
              style={{ ...field, width: 120 }}
            />
          ) : (
            // The row handles view on click; the name keeps the double-click to
            // rename (single clicks bubble up and view, harmlessly).
            <span
              onDoubleClick={() => setRenaming({ name: c.name, value: c.name })}
              title="Double-click to rename"
              style={{
                color: shownRun === c.name ? 'var(--accent)' : 'var(--text)',
                fontFamily: 'monospace', fontSize: 11, fontWeight: 600,
              }}
            >
              {c.name}
            </span>
          )}
          {/* The kernel's live model — distinct from the dashboard accent, which
              marks whichever run is merely being shown. */}
          {kernelRun === c.name && (
            <span
              title="Loaded in the kernel — this is the live model"
              style={{
                fontSize: 9, textTransform: 'uppercase', letterSpacing: 0.5,
                color: 'var(--warn)', border: '1px solid var(--warn)',
                borderRadius: 4, padding: '0 4px', flexShrink: 0,
              }}
            >
              live
            </span>
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
          <span
            onClick={(e) => e.stopPropagation()}
            style={{ display: 'flex', flexDirection: 'column', gap: 4, width: 132, flexShrink: 0 }}
          >
            {/* Slot 1: keep — an affordance only while the kernel still holds
                THIS run's weights; a status label otherwise. */}
            {!hasWeights && kernelRun === c.name && !running ? (
              <button
                onClick={() => keepWeights(c.name)}
                title="Save this run's weights (the kernel still holds them) — enables restore/resume/download"
                style={{ ...actionButton, width: '100%', color: 'var(--accent)', borderColor: 'var(--accent)' }}
              >
                ＋ save weights
              </button>
            ) : (
              <span
                title={
                  hasWeights
                    ? 'The weights are saved — restore, resume, and download are available'
                    : 'Only the curves were recorded; the kernel no longer holds these weights'
                }
                style={{
                  ...actionButton,
                  width: '100%', border: '1px solid transparent', cursor: 'default',
                  color: 'var(--text-6)', textAlign: 'center', boxSizing: 'border-box',
                }}
              >
                {hasWeights ? 'saved ✓' : 'not saved'}
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
                // Always set color AND borderColor (never the shorthand alone):
                // toggling off must revert the value, not remove the longhand —
                // a removed border-color falls back to currentColor (black).
                style={{
                  ...actionButton,
                  width: '100%',
                  color: compared.includes(c.name) ? 'var(--accent)' : 'var(--text-3)',
                  borderColor: compared.includes(c.name) ? 'var(--accent)' : 'var(--border)',
                }}
              >
                ⊕ compare
              </button>
            )}
            <span
              style={{ display: 'flex', gap: 4, alignItems: 'center' }}
              title={hasWeights ? undefined : 'Resume needs weights — this run saved none'}
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
                  : 'Restore needs weights — this run saved none'
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
                      : 'Download needs weights — this run saved none'
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

      {/* Swapping the live model would drop an unsaved run. A modal (portaled to
          the body) rather than an inline banner: the decision is modal, and
          nothing in the pane shifts to make room for it. */}
      {pendingSwap &&
        createPortal(
          <div
            onClick={() => setPendingSwap(null)}
            style={{
              position: 'fixed', inset: 0, zIndex: 1000,
              background: 'rgba(0, 0, 0, 0.45)',
              display: 'flex', alignItems: 'center', justifyContent: 'center',
            }}
          >
            <div
              onClick={(e) => e.stopPropagation()}
              style={{
                background: 'var(--panel)', border: '1px solid var(--border)',
                borderRadius: 8, padding: 16, width: 'min(360px, calc(100vw - 32px))',
                boxShadow: '0 8px 30px rgba(0, 0, 0, 0.35)',
                fontFamily: 'monospace', fontSize: 12,
                display: 'flex', flexDirection: 'column', gap: 14,
              }}
            >
              <span style={{ color: 'var(--text)', lineHeight: 1.5 }}>
                <code style={chip}>{pendingSwap.kernelName}</code> is the live model, and its
                weights aren't saved. Loading <code style={chip}>{pendingSwap.target}</code> will
                discard them.
              </span>
              <div style={{ display: 'flex', gap: 8, justifyContent: 'flex-end', flexWrap: 'wrap' }}>
                <button onClick={() => setPendingSwap(null)} style={actionButton}>
                  cancel
                </button>
                <button onClick={() => confirmSwap(false)} style={actionButton}>
                  discard &amp; continue
                </button>
                <button
                  onClick={() => confirmSwap(true)}
                  style={{ ...actionButton, color: 'var(--accent)', borderColor: 'var(--accent)' }}
                >
                  save &amp; continue
                </button>
              </div>
            </div>
          </div>,
          document.body
        )}
    </div>
  )
}
