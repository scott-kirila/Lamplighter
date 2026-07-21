import { useEffect, useRef } from 'react'
import { fmtDuration, fmtMetric, metricColumns } from '../lib/epochMetrics'
import { unitWord, useRecipes } from '../hooks/useRecipes'
import { useRunStore, type RunEpoch } from '../store/runStore'
import { ReadinessPanel } from './ReadinessPanel'
import type { Readiness } from '../hooks/useReadiness'

// The dashboard's numbers half: the per-epoch metrics table (newest on top, with
// a total/ETA header), or — before a run streams — the "starting…" state or the
// pre-flight readiness checklist. Shared verbatim by the two-column and
// full-width dashboard layouts. Extracted from TrainingTab to keep that file to
// layout/orchestration.
export function RunEpochsPanel({
  epochs,
  bestEpoch,
  runState,
  runError,
  etaSecs,
  readiness,
}: {
  epochs: RunEpoch[]
  bestEpoch: number | null
  runState: string
  runError: string | null
  // Mean epoch time × epochs remaining, or null when not computable (no timing).
  etaSecs: number | null
  readiness: Readiness
}) {
  // Keep the newest epoch line visible as they stream in.
  const epochsEndRef = useRef<HTMLDivElement>(null)
  useEffect(() => {
    epochsEndRef.current?.scrollIntoView({ block: 'nearest' })
  }, [epochs.length])

  // The shown run's progress column reads in its own unit (RL = iterations).
  const { data: recipes } = useRecipes()
  const shownRecipe = useRunStore((s) => s.runConfig?.recipe)
  const units = unitWord(recipes, shownRecipe)

  const showRun = epochs.length > 0 || !!runError

  if (!showRun) {
    return runState === 'running' ? (
      <div
        style={{
          height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center',
          fontSize: 12, color: 'var(--text-6)',
        }}
      >
        starting…
      </div>
    ) : (
      <ReadinessPanel readiness={readiness} />
    )
  }

  return (
    <div style={{ padding: '10px 20px 8px', fontSize: 12, lineHeight: 1.6 }}>
      {/* Newest epoch on top: scroll the top into view as they stream. */}
      <div ref={epochsEndRef} />
      {(() => {
        const cols = metricColumns(epochs)
        // Per-epoch wall time — live runs carry it; epochs rebuilt from
        // history on a reconnect don't, so only show the column when present.
        const hasTiming = epochs.some((e) => e.secs !== undefined)
        const totalSecs = epochs.reduce((a, e) => a + (e.secs ?? 0), 0)
        // Left-pad the epoch number to the total's width so the "/N" lines
        // up (2/25 under 12/25). The ★ lives in its own leading column, so
        // its (non-space) glyph width can't shift the epoch text.
        const width = Math.max(1, ...epochs.map((e) => String(e.epochs).length))
        const th: React.CSSProperties = {
          textAlign: 'right', padding: '0 0 5px 16px', color: 'var(--text-5)', fontWeight: 400,
          fontSize: 10.5, textTransform: 'uppercase', letterSpacing: 0.5, whiteSpace: 'nowrap',
          borderBottom: '1px solid var(--border)', position: 'sticky', top: 0, background: 'var(--bg)',
        }
        const td: React.CSSProperties = { textAlign: 'right', padding: '2px 0 2px 16px', whiteSpace: 'nowrap' }
        const table = (
          <table style={{ borderCollapse: 'collapse', fontSize: 12 }}>
            <thead>
              <tr>
                <th style={{ ...th, padding: '0 0 5px 0', width: 14 }} aria-label="best" />
                <th style={{ ...th, textAlign: 'left', padding: '0 0 5px 6px' }}>{units}</th>
                {cols.map((c) => (
                  <th key={c} style={th}>{c}</th>
                ))}
                {hasTiming && <th style={th}>time</th>}
              </tr>
            </thead>
            <tbody>
              {[...epochs].reverse().map((e) => {
                const best = bestEpoch != null && e.epoch === bestEpoch
                return (
                  <tr key={e.epoch}>
                    <td
                      style={{ ...td, textAlign: 'center', padding: '2px 0', color: 'var(--accent)' }}
                      data-tip={best ? 'best epoch (lowest val loss)' : undefined}
                    >
                      {best ? '★' : ''}
                    </td>
                    <td
                      style={{
                        ...td, textAlign: 'left', padding: '2px 0 2px 6px', whiteSpace: 'pre',
                        color: best ? 'var(--accent)' : 'var(--text-5)',
                      }}
                    >
                      {`${String(e.epoch).padStart(width)}/${e.epochs}`}
                    </td>
                    {cols.map((c) => (
                      <td key={c} style={{ ...td, color: 'var(--text-3)' }}>
                        {fmtMetric(e.metrics[c])}
                      </td>
                    ))}
                    {hasTiming && (
                      <td style={{ ...td, color: 'var(--text-5)' }}>{fmtDuration(e.secs)}</td>
                    )}
                  </tr>
                )
              })}
            </tbody>
          </table>
        )
        return (
          <>
            {/* Total time above the table — a fixed spot, so it doesn't ride the
                newest row as epochs stream in (which prepend at the top). */}
            {hasTiming && (
              <div style={{ fontSize: 11, color: 'var(--text-6)', padding: '0 0 6px 6px' }}>
                <span data-tip="elapsed wall-time so far">total {fmtDuration(totalSecs)}</span>
                {runState === 'running' && etaSecs !== null && (
                  <>
                    <span style={{ color: 'var(--text-8)', margin: '0 10px' }}>·</span>
                    <span data-tip="mean epoch time × epochs remaining">~{fmtDuration(etaSecs)} left</span>
                  </>
                )}
              </div>
            )}
            {table}
          </>
        )
      })()}
      {runError && <div style={{ color: 'var(--error)', marginTop: 4 }}>✗ {runError}</div>}
    </div>
  )
}
