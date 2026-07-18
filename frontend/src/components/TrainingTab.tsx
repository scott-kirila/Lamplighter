import { useEffect, useMemo, useRef, useState } from 'react'
import { useGraphStore } from '../store/graphStore'
import { useRunStore, type RunConfig } from '../store/runStore'
import { runBlocker, useReadiness } from '../hooks/useReadiness'
import { useCheckpoints } from '../hooks/useCheckpoints'
import { useRecipes } from '../hooks/useRecipes'
import { fmtDuration, fmtMetric, metricColumns } from '../lib/epochMetrics'
import { formatShape } from '../lib/formatShape'
import { paramVisible } from '../lib/paramVisible'
import type { CompareRun } from '../lib/runChart'
import type { ParamDef } from '../types/graph'
import { Checkpoints } from './Checkpoints'
import { OptionalControl, ParamControl } from './ParamControl'
import { ReadinessPanel } from './ReadinessPanel'
import { TrainingHealthPanel } from './TrainingHealthPanel'
import { PreviewView } from './PreviewView'
import { Group, Panel, Separator, useDefaultLayout } from 'react-resizable-panels'
import { RunCharts } from './RunCharts'

// A compared checkpoint: its curves (overlaid on the charts) + the training
// config that produced it (fed to the diff table).
type ComparedRun = CompareRun & { training: Record<string, unknown> }

// The lr as people write it: 0.0002 → "2e-4"; 0.01 and up as-is.
const fmtLr = (v: number) => (v >= 0.01 ? String(v) : v.toExponential(0))

// "cgan · 80 ep · lr g 2e-4 / d 2e-4 · cpu": the config the SHOWN run actually
// used (its snapshot), labelling the dashboard — the form edits the NEXT run,
// so the two can drift and the results must carry their own record.
function runConfigLabel(c: RunConfig): string {
  const parts: string[] = []
  if (c.recipe) parts.push(c.recipe)
  if (c.epochs != null) parts.push(`${c.epochs} ep`)
  if (c.lrs && Object.keys(c.lrs).length > 0) {
    parts.push('lr ' + Object.entries(c.lrs).map(([role, v]) => `${role[0]} ${fmtLr(v)}`).join(' / '))
  } else if (c.lr != null) {
    parts.push(`lr ${fmtLr(c.lr)}`)
  }
  if (c.device) parts.push(c.device)
  return parts.join(' · ')
}

// The "what changed between these runs?" table: one row per training param
// whose value differs across the compared runs. Structural keys (role
// assignments) aren't comparable scalars, so they're skipped.
function CompareDiff({ runs }: { runs: ComparedRun[] }) {
  if (runs.length < 2) return null
  const skip = new Set(['roles', 'per_role', 'recipe'])
  const keys = [...new Set(runs.flatMap((r) => Object.keys(r.training)))]
    .filter((k) => !skip.has(k))
    .filter((k) => new Set(runs.map((r) => JSON.stringify(r.training[k] ?? null))).size > 1)
  if (keys.length === 0) {
    return (
      <div style={{ color: 'var(--text-6)', fontSize: 11, marginBottom: 10 }}>
        compared runs share an identical training config
      </div>
    )
  }
  const cell: React.CSSProperties = { padding: '2px 14px 2px 0', textAlign: 'left', fontWeight: 400 }
  return (
    <table style={{ borderCollapse: 'collapse', fontSize: 11, marginBottom: 10 }}>
      <thead>
        <tr>
          <th style={{ ...cell, color: 'var(--text-6)' }}>differs</th>
          {runs.map((r) => (
            <th key={r.name} style={{ ...cell, color: 'var(--text)', fontWeight: 600 }}>{r.name}</th>
          ))}
        </tr>
      </thead>
      <tbody>
        {keys.map((k) => (
          <tr key={k}>
            <td style={{ ...cell, color: 'var(--text-5)' }}>{k}</td>
            {runs.map((r) => (
              <td key={r.name} style={{ ...cell, color: 'var(--text-3)' }}>
                {r.training[k] == null ? '—' : String(r.training[k])}
              </td>
            ))}
          </tr>
        ))}
      </tbody>
    </table>
  )
}

const RUN_STATE_COLOR: Record<string, string> = {
  running: 'var(--warn)',
  done: 'var(--accent)',
  stopped: 'var(--text-4)',
  failed: 'var(--error)',
}

const selectStyle: React.CSSProperties = {
  width: '100%',
  background: 'var(--field)',
  color: 'var(--text)',
  border: '1px solid var(--border)',
  borderRadius: 5,
  padding: '5px 8px',
  fontFamily: 'monospace',
  fontSize: 13,
}

// The side pane's accordion headers (matches the diagnostics panels' toggles).
const sectionLabel: React.CSSProperties = {
  color: 'var(--text-6)',
  fontSize: 10,
  letterSpacing: 1,
  textTransform: 'uppercase',
  marginBottom: 8,
}

// The left pane's Settings ↔ Runs tab strip (an active underline, like the
// primary tabs) — pinned above the scrolling content.
const sideTabStrip: React.CSSProperties = {
  display: 'flex',
  gap: 4,
  padding: '0 12px',
  borderBottom: '1px solid var(--border)',
  flexShrink: 0,
  background: 'var(--panel)',
}

const sideTabBtn = (active: boolean): React.CSSProperties => ({
  background: 'none',
  border: 'none',
  borderBottom: `2px solid ${active ? 'var(--accent)' : 'transparent'}`,
  color: active ? 'var(--text)' : 'var(--text-5)',
  cursor: 'pointer',
  fontFamily: 'monospace',
  fontSize: 11,
  textTransform: 'uppercase',
  letterSpacing: 1,
  padding: '9px 10px',
  marginBottom: -1,
})

export function TrainingTab() {
  const { data: recipes } = useRecipes()
  const training = useGraphStore((s) => s.training)
  const setTrainingParam = useGraphStore((s) => s.setTrainingParam)
  const setTrainingRoleParam = useGraphStore((s) => s.setTrainingRoleParam)
  const models = useGraphStore((s) => s.models)
  const nodes = useGraphStore((s) => s.nodes)
  const shapes = useGraphStore((s) => s.shapes)
  const paramCounts = useGraphStore((s) => s.paramCounts)
  const toProject = useGraphStore((s) => s.toProject)
  const trainingView = useGraphStore((s) => s.trainingView)
  const runState = useRunStore((s) => s.runState)
  const runEpochs = useRunStore((s) => s.runEpochs)
  const runError = useRunStore((s) => s.runError)
  const runSeed = useRunStore((s) => s.runSeed)
  const runBestEpoch = useRunStore((s) => s.runBestEpoch)
  const runConfig = useRunStore((s) => s.runConfig)
  const setRunStatus = useRunStore((s) => s.setRunStatus)

  // A hard readiness failure (data↔model mismatch, no data picked, a
  // loss/target incompatibility) disables ▶ Run with the reason, rather than
  // letting the click fail. Warn-level checks don't block. Only the first is
  // shown — fix it and the next surfaces. We gate ONLY on a fresh, successful
  // diagnose (status 'ready'); if diagnose is unavailable, Run stays enabled
  // (fail-open) and the backend's own start() validation is the backstop.
  const readiness = useReadiness()
  const blocker = runBlocker(readiness)

  // Run comparison: checkpoints toggled onto the charts (full history fetched
  // per toggle — metas stay light). Deleted checkpoints drop out automatically.
  // A failed fetch must say so: the classic cause is a running backend that
  // predates the endpoint (backend edits need a kernel restart).
  const [compare, setCompare] = useState<Record<string, ComparedRun>>({})
  const [compareError, setCompareError] = useState<string | null>(null)
  const [resultsOpen, setResultsOpen] = useState(true)
  // The left pane's two views — the settings form and the runs list — as tabs
  // rather than accordions (one is always fully open, no dropdowns to fiddle).
  const [sideTab, setSideTab] = useState<'settings' | 'runs'>('settings')
  const { data: checkpointMetas } = useCheckpoints()
  const toggleCompare = (ckptName: string) => {
    setCompareError(null)
    if (compare[ckptName]) {
      const { [ckptName]: _dropped, ...rest } = compare
      setCompare(rest)
      return
    }
    fetch(`/api/checkpoints/${encodeURIComponent(ckptName)}/history`)
      .then(async (r) => {
        if (!r.ok) {
          setCompareError(
            `couldn't load "${ckptName}" (HTTP ${r.status})` +
              (r.status === 404
                ? ' — the running backend may predate run comparison; restart the kernel'
                : '')
          )
          return
        }
        const run = await r.json()
        setCompare((prev) => ({ ...prev, [ckptName]: run }))
      })
      .catch(() => setCompareError('backend unreachable'))
  }
  useEffect(() => {
    const alive = new Set((checkpointMetas ?? []).map((c) => c.name))
    setCompare((prev) => {
      const kept = Object.fromEntries(Object.entries(prev).filter(([n]) => alive.has(n)))
      return Object.keys(kept).length === Object.keys(prev).length ? prev : kept
    })
  }, [checkpointMetas])
  const compareRuns = Object.values(compare)
  // Whether there's a run to show (streamed epochs, an error, or a compared run).
  // Otherwise the left cell shows the pre-flight readiness checklist instead.
  const showRun = runEpochs.length > 0 || !!runError || compareRuns.length > 0
  // Mean epoch wall-time × epochs left (live epochs carry secs; hydrated may not).
  const timedEpochs = runEpochs.filter((e) => e.secs !== undefined)
  const lastEpoch = runEpochs[runEpochs.length - 1]
  const etaSecs =
    timedEpochs.length > 0 && lastEpoch && lastEpoch.epochs > lastEpoch.epoch
      ? (timedEpochs.reduce((a, e) => a + (e.secs ?? 0), 0) / timedEpochs.length) *
        (lastEpoch.epochs - lastEpoch.epoch)
      : null
  // The draggable table ↔ checkpoints split, persisted to localStorage.
  const { defaultLayout, onLayoutChanged } = useDefaultLayout({
    // The resizable side pane (settings + run history) ↔ the dashboard.
    id: 'lamplighter-training-pane',
    panelIds: ['train-side', 'train-main'],
    storage: localStorage,
  })
  // The dashboard's own split: stacked graphs (left) ↔ the epoch results (right).
  const dash = useDefaultLayout({
    id: 'lamplighter-dashboard-split',
    panelIds: ['dash-graphs', 'dash-results'],
    storage: localStorage,
  })

  const recipeName = (training.recipe as string) ?? 'supervised'
  const recipe = recipes?.find((r) => r.name === recipeName) ?? recipes?.[0]
  const loopParams = recipe?.params ?? []
  // Memoized so the reference is stable across renders — otherwise the role
  // sync effect below (which lists `roles` as a dep) would re-run every render.
  const roles = useMemo(() => (training.roles as Record<string, string>) ?? {}, [training.roles])
  const perRole = (training.per_role as Record<string, Record<string, unknown>>) ?? {}
  // Assign roles explicitly whenever it's ambiguous — a multi-role recipe, or a
  // single role with more than one model to choose from. A lone model stays
  // auto-assigned (the classic zero-click path).
  const assignsRoles = !!recipe && (recipe.roles.length > 1 || models.length > 1)

  // Keep training.roles a valid role→model map: default each role to a model
  // positionally, prune roles the current recipe doesn't have.
  useEffect(() => {
    if (!recipe) return
    const next: Record<string, string> = {}
    if (recipe.roles.length > 1 || models.length > 1) {
      recipe.roles.forEach((role, i) => {
        const existing = roles[role.role]
        next[role.role] =
          existing && models.some((m) => m.id === existing)
            ? existing
            : models[Math.min(i, models.length - 1)]?.id ?? ''
      })
    }
    if (JSON.stringify(next) !== JSON.stringify(roles)) setTrainingParam('roles', next)
  }, [recipe, models, roles, setTrainingParam])

  // A GAN's generator needs a latent source: provision a noise node wired into
  // it (recipe-provisioned but explicit) once the generator role is assigned.
  const ensureGanNoise = useGraphStore((s) => s.ensureGanNoise)
  const ensureDatasetFor = useGraphStore((s) => s.ensureDatasetFor)
  const ensureCganWiring = useGraphStore((s) => s.ensureCganWiring)
  // The data-fed model (the recipe's data_role, positional fallback) gets a
  // dataset node; a GAN's generator additionally gets a noise node.
  const dataRoleIndex = Math.max(0, recipe ? recipe.roles.findIndex((r) => r.role === recipe.data_role) : 0)
  const dataModelId = (recipe && roles[recipe.data_role]) || models[dataRoleIndex]?.id || models[0]?.id
  useEffect(() => {
    // A cGAN provisions its whole conditional wiring at once (noise + a labeled
    // dataset feeding both models) — and only once BOTH roles are known. Never a
    // plain dataset in the meantime: a port-less dataset link into the
    // discriminator would otherwise pre-empt the fan-out. Every other recipe gets
    // a plain dataset (and a GAN also a noise node).
    if (recipe?.name === 'cgan') {
      if (roles.generator && roles.discriminator) ensureCganWiring(roles.generator, roles.discriminator)
      return
    }
    if (dataModelId) ensureDatasetFor(dataModelId)
    if (recipe?.name === 'gan' && roles.generator) ensureGanNoise(roles.generator)
  }, [recipe?.name, roles.generator, roles.discriminator, dataModelId, ensureDatasetFor, ensureGanNoise, ensureCganWiring])

  // Output-shape + size readout for the active model — context for choosing a
  // loss without the canvas.
  const outputNode = nodes.find((n) => n.data.nodeType === 'Output')
  const outShape = outputNode ? shapes[outputNode.id] : undefined
  const totalParams = Object.values(paramCounts).reduce((a, b) => a + b.count, 0)

  const defaults = Object.fromEntries(loopParams.map((p) => [p.name, p.default]))

  const renderParam = (param: ParamDef, value: unknown, onChange: (v: unknown) => void) => {
    const props = { param, value, nodeColor: 'var(--accent)', onChange }
    return (
      <div key={param.name} style={{ marginBottom: 14 }}>
        <label style={{ display: 'block', color: 'var(--text-5)', fontSize: 11, marginBottom: 4 }}>
          {param.label}
        </label>
        {param.optional ? <OptionalControl {...props} /> : <ParamControl {...props} />}
      </div>
    )
  }

  // Start/stop the in-kernel run. The whole project is posted, so multi-model
  // recipes (GAN) send every model; progress/state stream back over the WS.
  const startRun = async () => {
    try {
      const res = await fetch('/api/run/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(toProject()),
      })
      if (!res.ok) {
        const body = await res.json().catch(() => ({}))
        setRunStatus('failed', body.detail ?? 'could not start the run')
      }
    } catch {
      setRunStatus('failed', 'backend unreachable')
    }
  }
  const stopRun = () => fetch('/api/run/stop', { method: 'POST' }).catch(() => {})

  // Keep the newest epoch line visible as they stream in.
  const epochsEndRef = useRef<HTMLDivElement>(null)
  useEffect(() => {
    epochsEndRef.current?.scrollIntoView({ block: 'nearest' })
  }, [runEpochs.length])

  // The epoch results (or the pre-run readiness checklist / "starting…") — the
  // dashboard's numbers half, shared by the two-column and full-width layouts.
  const epochResults = showRun ? (
    <div style={{ padding: '10px 20px 8px', fontFamily: 'monospace', fontSize: 12, lineHeight: 1.6 }}>
      {/* Newest epoch on top: scroll the top into view as they stream. */}
      <div ref={epochsEndRef} />
      {(() => {
        const cols = metricColumns(runEpochs)
        // Per-epoch wall time — live runs carry it; epochs rebuilt from
        // history on a reconnect don't, so only show the column when present.
        const hasTiming = runEpochs.some((e) => e.secs !== undefined)
        const totalSecs = runEpochs.reduce((a, e) => a + (e.secs ?? 0), 0)
        // Left-pad the epoch number to the total's width so the "/N" lines
        // up (2/25 under 12/25). The ★ lives in its own leading column, so
        // its (non-space) glyph width can't shift the epoch text.
        const width = Math.max(1, ...runEpochs.map((e) => String(e.epochs).length))
        const th: React.CSSProperties = {
          textAlign: 'right', padding: '0 0 5px 16px', color: 'var(--text-5)', fontWeight: 400,
          fontSize: 10.5, textTransform: 'uppercase', letterSpacing: 0.5, whiteSpace: 'nowrap',
          borderBottom: '1px solid var(--border)', position: 'sticky', top: 0, background: 'var(--bg)',
        }
        const td: React.CSSProperties = { textAlign: 'right', padding: '2px 0 2px 16px', whiteSpace: 'nowrap' }
        const table = (
          <table style={{ borderCollapse: 'collapse', fontFamily: 'monospace', fontSize: 12 }}>
            <thead>
              <tr>
                <th style={{ ...th, padding: '0 0 5px 0', width: 14 }} aria-label="best" />
                <th style={{ ...th, textAlign: 'left', padding: '0 0 5px 6px' }}>epoch</th>
                {cols.map((c) => (
                  <th key={c} style={th}>{c}</th>
                ))}
                {hasTiming && <th style={th}>time</th>}
              </tr>
            </thead>
            <tbody>
              {[...runEpochs].reverse().map((e) => {
                const best = runBestEpoch != null && e.epoch === runBestEpoch
                return (
                  <tr key={e.epoch}>
                    <td
                      style={{ ...td, textAlign: 'center', padding: '2px 0', color: 'var(--accent)' }}
                      title={best ? 'best epoch (lowest val loss)' : undefined}
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
              <div style={{ fontFamily: 'monospace', fontSize: 11, color: 'var(--text-6)', padding: '0 0 6px 6px' }}>
                <span title="elapsed wall-time so far">total {fmtDuration(totalSecs)}</span>
                {runState === 'running' && etaSecs !== null && (
                  <>
                    <span style={{ color: 'var(--text-8)', margin: '0 10px' }}>·</span>
                    <span title="mean epoch time × epochs remaining">~{fmtDuration(etaSecs)} left</span>
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
  ) : runState === 'running' ? (
    <div
      style={{
        height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center',
        fontFamily: 'monospace', fontSize: 12, color: 'var(--text-6)',
      }}
    >
      starting…
    </div>
  ) : (
    <ReadinessPanel readiness={readiness} />
  )

  // The stacked graphs — the dashboard's visual half. (The input→output preview
  // is its own Training sub-tab now, see PreviewView.)
  const graphsPane = (
    <div style={{ flex: 1, minHeight: 0, overflowY: 'auto', padding: '14px 20px 0', fontFamily: 'monospace', fontSize: 12, lineHeight: 1.6 }}>
      <RunCharts epochs={runEpochs} height={200} bestEpoch={runBestEpoch} compare={compareRuns} stacked />
      <CompareDiff runs={compareRuns} />
    </div>
  )

  // The dashboard body has three shapes: a run with results shown → two columns
  // (graphs | results); a run with results hidden → graphs take the whole
  // stage; no run yet → the readiness checklist full-width.
  const dashboardBody = !showRun ? (
    <div style={{ flex: 1, minHeight: 0, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
      <div style={{ flex: 1, minHeight: 0, overflow: 'auto' }}>{epochResults}</div>
    </div>
  ) : resultsOpen ? (
    <Group
      orientation="horizontal"
      defaultLayout={dash.defaultLayout}
      onLayoutChanged={dash.onLayoutChanged}
      style={{ flex: 1, minHeight: 0 }}
    >
      {/* Graphs (+ the run's preview) — the widest-hungry content — left. */}
      <Panel id="dash-graphs" defaultSize={58} minSize={30} style={{ minWidth: 0, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
        {graphsPane}
      </Panel>
      <Separator style={{ width: 7, display: 'flex', alignItems: 'stretch', justifyContent: 'center', cursor: 'col-resize' }}>
        <div style={{ width: 1, background: 'var(--border)' }} />
      </Separator>
      {/* Epoch results — the narrow numbers column — right. */}
      <Panel id="dash-results" defaultSize={42} minSize={20} style={{ minWidth: 0, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
        <div style={{ flex: 1, minHeight: 0, overflow: 'auto' }}>{epochResults}</div>
      </Panel>
    </Group>
  ) : (
    // Results hidden: the graphs get the whole dashboard.
    <div style={{ flex: 1, minHeight: 0, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
      {graphsPane}
    </div>
  )

  return (
    <Group
      orientation="horizontal"
      defaultLayout={defaultLayout}
      onLayoutChanged={onLayoutChanged}
      style={{ flex: 1, minHeight: 0 }}
    >
      {/* The side pane: training settings and the run history as two TABS (no
          accordion dropdowns — one is always fully open). Lives beside the form,
          out of the dashboard's vertical budget (charts growing can't push it). */}
      <Panel
        id="train-side"
        defaultSize={22}
        minSize={15}
        style={{ minWidth: 0, display: 'flex', flexDirection: 'column', overflow: 'hidden', background: 'var(--panel)' }}
      >
        <div style={sideTabStrip}>
          <button onClick={() => setSideTab('settings')} style={sideTabBtn(sideTab === 'settings')}>
            Settings
          </button>
          <button onClick={() => setSideTab('runs')} style={sideTabBtn(sideTab === 'runs')}>
            Runs
          </button>
        </div>
        <div style={{ flex: 1, minHeight: 0, overflowY: 'auto', fontFamily: 'monospace' }}>
        {sideTab === 'settings' ? (
        // Capped: widening the pane gives the RUNS list room — form controls
        // at 300px stay comfortably scannable instead of stretching with it.
        <div style={{ padding: '14px 16px 16px', maxWidth: 300 }}>
        <div style={{ color: 'var(--text-6)', fontSize: 11, marginBottom: 4 }}>
          model output:{' '}
          <span style={{ color: 'var(--accent)' }}>
            {outShape ? `[${formatShape(outShape, ', ')}]` : '—'}
          </span>
        </div>
        <div style={{ color: 'var(--text-6)', fontSize: 11, marginBottom: 16 }}>
          parameters:{' '}
          <span style={{ color: 'var(--accent)' }}>
            {totalParams > 0 ? totalParams.toLocaleString('en-US') : '—'}
          </span>
        </div>

        {/* Recipe picker — only when there's a choice. */}
        {recipes && recipes.length > 1 && (
          <div style={{ marginBottom: 18 }}>
            <label style={{ display: 'block', color: 'var(--text-5)', fontSize: 11, marginBottom: 4 }}>
              Recipe
            </label>
            <select
              value={recipeName}
              onChange={(e) => setTrainingParam('recipe', e.target.value)}
              style={selectStyle}
            >
              {recipes.map((r) => (
                <option key={r.name} value={r.name}>
                  {r.label}
                </option>
              ))}
            </select>
          </div>
        )}

        {/* Role assignment + per-role params (when ambiguous). */}
        {assignsRoles && recipe && (
          <div style={{ marginBottom: 18 }}>
            <div style={sectionLabel}>Roles</div>
            {recipe.roles.map((role) => (
              <div key={role.role} style={{ marginBottom: 14 }}>
                <label style={{ display: 'block', color: 'var(--text-5)', fontSize: 11, marginBottom: 4 }}>
                  {role.label}
                </label>
                <select
                  value={roles[role.role] ?? ''}
                  onChange={(e) => setTrainingParam('roles', { ...roles, [role.role]: e.target.value })}
                  style={selectStyle}
                >
                  {models.map((m) => (
                    <option key={m.id} value={m.id}>
                      {m.name}
                    </option>
                  ))}
                </select>
                {(recipe.role_params[role.role] ?? []).map((param) =>
                  renderParam(
                    param,
                    perRole[role.role]?.[param.name] ?? param.default,
                    (v) => setTrainingRoleParam(role.role, param.name, v)
                  )
                )}
              </div>
            ))}
          </div>
        )}

        {/* Loop params for the selected recipe. */}
        {loopParams
          .filter((param) => paramVisible(param, { ...defaults, ...training }))
          .map((param) => renderParam(param, training[param.name], (v) => setTrainingParam(param.name, v)))}
        </div>
        ) : (
          <Checkpoints embedded compared={Object.keys(compare)} onToggleCompare={toggleCompare} />
        )}
        </div>
      </Panel>
      {/* Draggable divider — the pane split persists via useDefaultLayout. */}
      <Separator
        style={{ width: 7, display: 'flex', alignItems: 'stretch', justifyContent: 'center', cursor: 'col-resize' }}
      >
        <div style={{ width: 1, background: 'var(--border)' }} />
      </Separator>

      {/* Run dashboard — live charts + epoch log. The generated train() opens
          via the titlebar's Show code button (a CodePanel, like the Model tab). */}
      {/* minHeight: 0 lets this column bound its content instead of growing to
          fit it, so the epoch table's own overflow scrolls (a long run doesn't
          push the charts/table past the window). */}
      <Panel id="train-main" defaultSize={78} minSize={50} style={{ minWidth: 0, display: 'flex', overflow: 'hidden' }}>
      <div style={{ flex: 1, minWidth: 0, minHeight: 0, display: 'flex', flexDirection: 'column', background: 'var(--bg)' }}>
        <div
          style={{
            height: 36,
            background: 'var(--panel)',
            borderBottom: '1px solid var(--border)',
            display: 'flex',
            alignItems: 'center',
            gap: 10,
            padding: '0 16px',
            fontFamily: 'monospace',
            fontSize: 11,
            color: 'var(--text-4)',
            flexShrink: 0,
          }}
        >
          <span style={{ textTransform: 'uppercase', letterSpacing: 1 }}>Training run</span>
          {/* Hide the epoch results to give the graphs the whole dashboard —
              a labeled pill (the full label swaps, so the state reads at a
              glance) that names the thing it toggles. (Dashboard view only —
              the Preview sub-tab has no stats column.) */}
          {showRun && trainingView === 'dashboard' && (
            <button
              onClick={() => setResultsOpen((o) => !o)}
              title={resultsOpen ? 'Hide the stats — graphs take the full width' : 'Show the stats column'}
              style={{
                background: resultsOpen ? 'var(--surface)' : 'none',
                color: resultsOpen ? 'var(--text-3)' : 'var(--text-6)',
                border: '1px solid var(--border)', borderRadius: 4, padding: '2px 9px',
                fontFamily: 'monospace', fontSize: 11, cursor: 'pointer', lineHeight: 1.4,
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
              {runConfigLabel(runConfig)}
            </span>
          )}
          {/* Right cluster: seed · state · eta-or-weights · button. Each is a
              fixed-width slot ALWAYS rendered (empty when idle), so STARTING a run
              fills the slots in place instead of inserting them and shoving the
              button and its neighbours sideways. Seed is reserved wide enough for
              the longest value, so a new run's differing digit count can't jostle
              the cluster either. */}
          <span
            style={{ marginLeft: 'auto', minWidth: 108, textAlign: 'right', color: 'var(--text-6)' }}
          >
            {runState !== 'idle' && runSeed !== null ? `seed ${runSeed}` : ''}
          </span>
          <span style={{ minWidth: 52, color: RUN_STATE_COLOR[runState] ?? 'var(--text-6)' }}>
            {runState === 'idle' ? '' : runState}
          </span>
          {/* Downloading weights lives on each saved run in the runs list now —
              save the run, then download its .pt from its row. */}
          {runState === 'running' ? (
            <button
              onClick={stopRun}
              style={{
                background: 'none', border: '1px solid var(--border)', borderRadius: 5,
                color: 'var(--error)', cursor: 'pointer', fontFamily: 'monospace',
                fontSize: 12, fontWeight: 600, padding: '3px 14px',
                minWidth: 76, textAlign: 'center',
              }}
            >
              ■ Stop
            </button>
          ) : (
            <button
              onClick={startRun}
              disabled={!!blocker}
              title={
                blocker
                  ? `Can't run: ${blocker.title}${blocker.detail ? ` — ${blocker.detail}` : ''}`
                  : 'Train in the notebook kernel using the wired data node(s) — runs exactly this code'
              }
              style={{
                background: 'var(--accent)', border: 'none', borderRadius: 5,
                color: 'var(--text-on-accent)', fontFamily: 'monospace',
                fontSize: 12, fontWeight: 600, padding: '4px 16px',
                cursor: blocker ? 'default' : 'pointer',
                opacity: blocker ? 0.5 : 1,
                minWidth: 76, textAlign: 'center',
              }}
            >
              ▶ Run
            </button>
          )}
          {/* The blocker that disabled Run, spelled out inline (the tooltip is
              easy to miss on a greyed button). */}
          {blocker && runState !== 'running' && (
            <span style={{ color: 'var(--error)', fontFamily: 'monospace', fontSize: 11 }}>
              ✗ {blocker.title}
            </span>
          )}
          {/* Readiness couldn't be checked — say so (don't imply "all clear").
              Run still works; the backend validates on start. */}
          {readiness.status === 'unavailable' && runState !== 'running' && (
            <span
              title="The readiness check didn't respond, so pre-run blockers can't be shown. Run still validates on the backend."
              style={{ color: 'var(--text-6)', fontFamily: 'monospace', fontSize: 11 }}
            >
              ⚠ readiness unavailable
            </span>
          )}
        </div>
        {/* The main area swaps with the Training sub-tab: the run dashboard
            (graphs + stats) or the input→output model preview. The side pane
            (settings + runs list) stays put across both. */}
        {trainingView === 'preview' ? <PreviewView /> : dashboardBody}
        {/* Collapsible diagnostics pinned at the bottom — each self-hides until it
            has data. Dashboard view only. */}
        {trainingView === 'dashboard' && <TrainingHealthPanel />}
        {compareError && (
          <div
            style={{
              borderTop: '1px solid var(--border)', background: 'var(--panel)',
              padding: '6px 16px', fontFamily: 'monospace', fontSize: 11, color: 'var(--error)', flexShrink: 0,
            }}
          >
            ✗ {compareError}
          </div>
        )}
      </div>
      </Panel>
    </Group>
  )
}
