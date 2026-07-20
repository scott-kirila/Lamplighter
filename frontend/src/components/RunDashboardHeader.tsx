import type { RunConfig } from '../store/runStore'
import type { DiagnosticCheck } from '../hooks/useReadiness'
import { unitWord, useRecipes } from '../hooks/useRecipes'
import { useSweepControls } from '../hooks/useSweepControls'
import { useSweepStore } from '../store/sweepStore'
import { eyebrow } from '../styles/ui'

// The lr as people write it: 0.0002 → "2e-4"; 0.01 and up rounded to ~3
// significant digits (a sweep-suggested 0.012214485996651061 reads "0.0122").
const fmtLr = (v: number) =>
  v >= 0.01 ? String(parseFloat(v.toPrecision(3))) : v.toExponential(0)

// "cgan · 80 ep · lr g 2e-4 / d 2e-4 · cpu": the config the SHOWN run actually
// used (its snapshot), labelling the dashboard — the form edits the NEXT run,
// so the two can drift and the results must carry their own record. An RL
// run's plan reads in iters (its form says Iterations, so its chip must too).
function runConfigLabel(c: RunConfig, units: 'epoch' | 'iter'): string {
  const parts: string[] = []
  if (c.recipe) parts.push(c.recipe)
  if (c.epochs != null) parts.push(`${c.epochs} ${units === 'iter' ? 'iters' : 'ep'}`)
  if (c.lrs && Object.keys(c.lrs).length > 0) {
    parts.push('lr ' + Object.entries(c.lrs).map(([role, v]) => `${role[0]} ${fmtLr(v)}`).join(' / '))
  } else if (c.lr != null) {
    parts.push(`lr ${fmtLr(c.lr)}`)
  }
  if (c.device) parts.push(c.device)
  return parts.join(' · ')
}

const RUN_STATE_COLOR: Record<string, string> = {
  running: 'var(--warn)',
  done: 'var(--accent)',
  stopped: 'var(--text-4)',
  failed: 'var(--error)',
}

// The run dashboard's titlebar: the "Training run" label, the stats-column
// toggle, the shown run's recorded-config label, and the right-hand cluster
// (seed · state · Run/Stop button · blocker reason). Each right-cluster slot is a
// fixed-width, always-rendered box so starting a run fills the slots in place
// rather than inserting them and shoving the button sideways.
export function RunDashboardHeader({
  runState,
  runConfig,
  runSeed,
  blocker,
  readinessUnavailable,
  showRun,
  isDashboard,
  resultsOpen,
  onToggleResults,
  onRun,
  onStop,
}: {
  runState: string
  runConfig: RunConfig | null
  runSeed: number | null
  blocker: DiagnosticCheck | undefined
  readinessUnavailable: boolean
  showRun: boolean
  isDashboard: boolean
  resultsOpen: boolean
  onToggleResults: () => void
  onRun: () => void
  onStop: () => void
}) {
  const { data: recipes } = useRecipes()
  const units = unitWord(recipes, runConfig?.recipe)
  const sweepRunning = useSweepStore((s) => s.state === 'running')
  const { stop: stopSweep } = useSweepControls()
  const onStopSweep = () => stopSweep.mutate()
  return (
    <div
      style={{
        height: 36,
        background: 'var(--panel)',
        borderBottom: '1px solid var(--border)',
        display: 'flex',
        alignItems: 'center',
        gap: 10,
        padding: '0 16px',
        fontSize: 11,
        color: 'var(--text-4)',
        flexShrink: 0,
      }}
    >
      <span style={eyebrow}>Training run</span>
      {/* Hide the epoch results to give the graphs the whole dashboard — a
          labeled pill (the full label swaps, so the state reads at a glance) that
          names the thing it toggles. (Dashboard view only — the Preview sub-tab
          has no stats column.) */}
      {showRun && isDashboard && (
        <button
          onClick={onToggleResults}
          title={resultsOpen ? 'Hide the stats — graphs take the full width' : 'Show the stats column'}
          style={{
            background: resultsOpen ? 'var(--surface)' : 'none',
            color: resultsOpen ? 'var(--text-3)' : 'var(--text-6)',
            border: '1px solid var(--border)', borderRadius: 4, padding: '2px 9px',
            fontSize: 11, cursor: 'pointer', lineHeight: 1.4,
            textTransform: 'none', letterSpacing: 0,
          }}
        >
          {resultsOpen ? 'hide stats' : 'show stats'}
        </button>
      )}
      {/* The shown run's own recorded config — the form edits the next run. */}
      {runState !== 'idle' && runConfig && (
        <span
          title="What this run actually used — the form on the left configures the next run"
          style={{ color: 'var(--text-6)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', minWidth: 0 }}
        >
          {runConfigLabel(runConfig, units)}
        </span>
      )}
      {/* Right cluster: seed · state · button. Each is a fixed-width slot ALWAYS
          rendered (empty when idle), so STARTING a run fills the slots in place
          instead of inserting them and shoving the button and its neighbours
          sideways. Seed is reserved wide enough for the longest value, so a new
          run's differing digit count can't jostle the cluster either. */}
      <span style={{ marginLeft: 'auto', minWidth: 108, textAlign: 'right', color: 'var(--text-6)' }}>
        {runState !== 'idle' && runSeed !== null ? `seed ${runSeed}` : ''}
      </span>
      <span style={{ minWidth: 52, color: RUN_STATE_COLOR[runState] ?? 'var(--text-6)' }}>
        {runState === 'idle' ? '' : runState}
      </span>
      {/* Downloading weights lives on each saved run in the runs list now — save
          the run, then download its .pt from its row. */}
      {sweepRunning ? (
        // A live sweep owns the run slot: the auto-jump lands you HERE, so this
        // is where ending the sweep must live. Stopping just the current trial
        // stays available as the manual prune it is; ▶ Run disappears entirely
        // (between trials it would hijack the slot and abort the sweep — the
        // API refuses it too).
        <>
          {runState === 'running' && (
            <button
              onClick={onStop}
              title="Stop this trial only — it records as stopped and the sweep moves on (a manual prune)"
              style={{
                background: 'none', border: '1px solid var(--border)', borderRadius: 5,
                color: 'var(--text-4)', cursor: 'pointer', fontSize: 11, padding: '3px 10px',
              }}
            >
              skip trial
            </button>
          )}
          <button
            onClick={onStopSweep}
            title="End the sweep — the current trial stops and records; the best so far is kept"
            style={{
              background: 'none', border: '1px solid var(--border)', borderRadius: 5,
              color: 'var(--error)', cursor: 'pointer',
              fontSize: 12, fontWeight: 600, padding: '3px 14px',
              minWidth: 76, textAlign: 'center',
            }}
          >
            ■ Stop sweep
          </button>
        </>
      ) : runState === 'running' ? (
        <button
          onClick={onStop}
          style={{
            background: 'none', border: '1px solid var(--border)', borderRadius: 5,
            color: 'var(--error)', cursor: 'pointer',
            fontSize: 12, fontWeight: 600, padding: '3px 14px',
            minWidth: 76, textAlign: 'center',
          }}
        >
          ■ Stop
        </button>
      ) : (
        <button
          onClick={onRun}
          disabled={!!blocker}
          title={
            blocker
              ? `Can't run: ${blocker.title}${blocker.detail ? ` — ${blocker.detail}` : ''}`
              : 'Train in the notebook kernel using the wired data node(s) — runs exactly this code'
          }
          style={{
            background: 'var(--accent)', border: 'none', borderRadius: 5,
            color: 'var(--text-on-accent)',
            fontSize: 12, fontWeight: 600, padding: '4px 16px',
            cursor: blocker ? 'default' : 'pointer',
            opacity: blocker ? 0.5 : 1,
            minWidth: 76, textAlign: 'center',
          }}
        >
          ▶ Run
        </button>
      )}
      {/* The blocker that disabled Run, spelled out inline (the tooltip is easy
          to miss on a greyed button). */}
      {blocker && runState !== 'running' && !sweepRunning && (
        <span style={{ color: 'var(--error)', fontSize: 11 }}>
          ✗ {blocker.title}
        </span>
      )}
      {/* Readiness couldn't be checked — say so (don't imply "all clear"). Run
          still works; the backend validates on start. */}
      {readinessUnavailable && runState !== 'running' && !sweepRunning && (
        <span
          title="The readiness check didn't respond, so pre-run blockers can't be shown. Run still validates on the backend."
          style={{ color: 'var(--text-6)', fontSize: 11 }}
        >
          ⚠ readiness unavailable
        </span>
      )}
    </div>
  )
}
