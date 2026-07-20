import { useEffect, useMemo, useRef, useState } from 'react'
import { useGraphStore } from '../store/graphStore'
import { useRunStore } from '../store/runStore'
import { runBlocker, useReadiness } from '../hooks/useReadiness'
import { useRunView } from '../hooks/useRunView'
import { belongsToModel, isSweepTrial, useCheckpoints } from '../hooks/useCheckpoints'
import { useCheckpointActions } from '../hooks/useCheckpointActions'
import { useRunControls } from '../hooks/useRunControls'
import { useRecipes } from '../hooks/useRecipes'
import { formatShape } from '../lib/formatShape'
import { paramVisible } from '../lib/paramVisible'
import type { CompareRun } from '../lib/runChart'
import type { ParamDef } from '../types/graph'
import { Checkpoints } from './Checkpoints'
import { OptionalControl, ParamControl } from './ParamControl'
import { TrainingHealthPanel } from './TrainingHealthPanel'
import { PreviewView } from './PreviewView'
import { OptimizeView } from './OptimizeView'
import { Group, Panel, Separator, useDefaultLayout, type Layout } from 'react-resizable-panels'
import { useTrainingHealth } from '../hooks/useTrainingHealth'
import { RunCharts } from './RunCharts'
import { RunEpochsPanel } from './RunEpochsPanel'
import { RunDashboardHeader } from './RunDashboardHeader'
import { DiscardWeightsModal } from './DiscardWeightsModal'
import { eyebrow } from '../styles/ui'

// A compared checkpoint: its curves (overlaid on the charts) + the training
// config that produced it (fed to the diff table) + which model(s) it trained
// (so a cross-model comparison can name each run's architecture).
type ComparedRun = CompareRun & {
  training: Record<string, unknown>
  models?: { id: string; name: string; role: string }[]
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
  // Which model each run trained — the point of a cross-model comparison. Shown
  // as its own row when the runs trained different models.
  const modelLabel = (r: ComparedRun) => r.models?.map((m) => m.name).join(', ') || '—'
  const modelsDiffer = new Set(runs.map(modelLabel)).size > 1
  if (keys.length === 0 && !modelsDiffer) {
    return (
      <div style={{ color: 'var(--text-6)', fontSize: 11, marginBottom: 10 }}>
        compared runs share an identical model and training config
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
        {modelsDiffer && (
          <tr>
            <td style={{ ...cell, color: 'var(--text-5)' }}>model</td>
            {runs.map((r) => (
              <td key={r.name} style={{ ...cell, color: 'var(--text-3)' }}>{modelLabel(r)}</td>
            ))}
          </tr>
        )}
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
  ...eyebrow,
  color: 'var(--text-6)',
  fontSize: 10,
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
  ...eyebrow,
  background: 'none',
  border: 'none',
  borderBottom: `2px solid ${active ? 'var(--accent)' : 'transparent'}`,
  color: active ? 'var(--text)' : 'var(--text-5)',
  cursor: 'pointer',
  fontFamily: 'monospace',
  fontSize: 11,
  padding: '9px 10px',
  marginBottom: -1,
})

export function TrainingTab() {
  const { data: recipes } = useRecipes()
  const training = useGraphStore((s) => s.training)
  const setTrainingParam = useGraphStore((s) => s.setTrainingParam)
  const setTrainingRoleParam = useGraphStore((s) => s.setTrainingRoleParam)
  const models = useGraphStore((s) => s.models)
  const activeModelId = useGraphStore((s) => s.activeModelId)
  const openModel = useGraphStore((s) => s.openModel)
  const nodes = useGraphStore((s) => s.nodes)
  const shapes = useGraphStore((s) => s.shapes)
  const paramCounts = useGraphStore((s) => s.paramCounts)
  const toProject = useGraphStore((s) => s.toProject)
  const trainingView = useGraphStore((s) => s.trainingView)
  const setTrainingView = useGraphStore((s) => s.setTrainingView)
  const runState = useRunStore((s) => s.runState)
  const runEpochs = useRunStore((s) => s.runEpochs)
  const runError = useRunStore((s) => s.runError)
  const runSeed = useRunStore((s) => s.runSeed)
  const runBestEpoch = useRunStore((s) => s.runBestEpoch)
  const runConfig = useRunStore((s) => s.runConfig)
  const setRunStatus = useRunStore((s) => s.setRunStatus)
  const kernelRunName = useRunStore((s) => s.kernelRunName)
  const runName = useRunStore((s) => s.runName)
  const clearShownRun = useRunStore((s) => s.clearShownRun)
  const viewRun = useRunView()

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
  // A run start held back (from the Preview tab) because it would discard the
  // current model's unsaved weights.
  const [pendingRun, setPendingRun] = useState(false)
  const { data: checkpointMetas } = useCheckpoints()
  const { save } = useCheckpointActions()
  const runControls = useRunControls()
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
        // The listing row carries the run's model attribution; the /history
        // payload doesn't, so pull it from the meta we already have.
        const runModels = (checkpointMetas ?? []).find((c) => c.name === ckptName)?.models
        setCompare((prev) => ({ ...prev, [ckptName]: { ...run, models: runModels } }))
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
  // The live graphs↔results ratio, so the health band below can mirror it. Seeded
  // from the persisted layout (or the panels' 58/42 defaults) and updated on every
  // drag move via the Group's onLayoutChange (below).
  const [dashSplit, setDashSplit] = useState<Layout>(
    dash.defaultLayout ?? { 'dash-graphs': 58, 'dash-results': 42 },
  )
  // Training health streams once a run starts. Lifted here (rather than inside
  // the panel) so the layout can gate on whether there's any health data — it's
  // a full-width strip anchored below the graphs|results columns.
  const healthRoles = useTrainingHealth()
  const hasHealth = healthRoles.length > 0
  const plannedEpochs = runEpochs[runEpochs.length - 1]?.epochs ?? 0

  const recipeName = (training.recipe as string) ?? 'supervised'
  const recipe = recipes?.find((r) => r.name === recipeName) ?? recipes?.[0]
  const loopParams = recipe?.params ?? []
  // Memoized so the reference is stable across renders — otherwise the role
  // sync effect below (which lists `roles` as a dep) would re-run every render.
  const roles = useMemo(() => (training.roles as Record<string, string>) ?? {}, [training.roles])
  const perRole = (training.per_role as Record<string, Record<string, unknown>>) ?? {}
  // The explicit Roles dropdown shows ONLY for multi-role recipes (GAN/VAE),
  // which span several models. A single-role recipe (Supervised) instead trains
  // the ACTIVE model — the one you're editing — so there's no role to pick.
  const assignsRoles = !!recipe && recipe.roles.length > 1
  // The "Training: <model>" switcher: a single-role recipe with more than one
  // model to choose from. (A lone model needs no switcher — it's the only one.)
  const singleRoleMultiModel = !!recipe && recipe.roles.length === 1 && models.length > 1

  // Keep training.roles a valid role→model map. Multi-role recipes default each
  // role positionally (keeping valid explicit picks); a single-role recipe with
  // several models targets the ACTIVE model; a lone model stays auto-assigned
  // (empty roles → the backend picks the sole model — the classic path).
  useEffect(() => {
    if (!recipe) return
    let next: Record<string, string> = {}
    if (recipe.roles.length > 1) {
      recipe.roles.forEach((role, i) => {
        const existing = roles[role.role]
        next[role.role] =
          existing && models.some((m) => m.id === existing)
            ? existing
            : models[Math.min(i, models.length - 1)]?.id ?? ''
      })
    } else if (models.length > 1) {
      const target = models.some((m) => m.id === activeModelId) ? activeModelId : models[0]?.id ?? ''
      next = { [recipe.roles[0].role]: target }
    }
    if (JSON.stringify(next) !== JSON.stringify(roles)) setTrainingParam('roles', next)
  }, [recipe, models, roles, activeModelId, setTrainingParam])

  // The dashboard follows the active model: switching models shows that model's
  // latest recorded run (read-only), or the readiness checklist when it has
  // none. Fires ONLY on a real model switch (the ref skips mount and any
  // metas/run-state change), so it never clobbers a just-finished run or fights
  // an explicit row click; a live run always owns the dashboard.
  const prevActive = useRef(activeModelId)
  useEffect(() => {
    if (prevActive.current === activeModelId) return
    prevActive.current = activeModelId
    if (runState === 'running') return
    // Skip tucked-away sweep trials: following the model switch must land on
    // a run the list actually shows (a crowned best qualifies).
    const mine = (checkpointMetas ?? []).filter(
      (c) => belongsToModel(c, activeModelId) && !isSweepTrial(c)
    )
    if (runName && mine.some((c) => c.name === runName)) return
    const latest = mine[mine.length - 1] // metas are insertion order → newest last
    if (latest) viewRun(latest.name)
    else clearShownRun()
  }, [activeModelId, checkpointMetas, runState, runName, viewRun, clearShownRun])

  // A GAN's generator needs a latent source: provision a noise node wired into
  // it (recipe-provisioned but explicit) once the generator role is assigned.
  const ensureGanNoise = useGraphStore((s) => s.ensureGanNoise)
  const ensureDatasetFor = useGraphStore((s) => s.ensureDatasetFor)
  const ensureEnvFor = useGraphStore((s) => s.ensureEnvFor)
  const ensureCganWiring = useGraphStore((s) => s.ensureCganWiring)
  // The data-fed model (the recipe's data_role, positional fallback) gets a
  // dataset node — or a Gymnasium env for an RL recipe; a GAN's generator
  // additionally gets a noise node.
  const isEnvRecipe = recipe?.data === 'env'
  const dataRoleIndex = Math.max(0, recipe ? recipe.roles.findIndex((r) => r.role === recipe.data_role) : 0)
  const dataModelId = (recipe && roles[recipe.data_role]) || models[dataRoleIndex]?.id || models[0]?.id
  useEffect(() => {
    // Wait for the recipe to load before provisioning — otherwise the
    // undefined-recipe first render would fall through to a spurious dataset
    // (isEnvRecipe reads recipe.data, which isn't there yet).
    if (!recipe) return
    // A cGAN provisions its whole conditional wiring at once (noise + a labeled
    // dataset feeding both models) — and only once BOTH roles are known. Never a
    // plain dataset in the meantime: a port-less dataset link into the
    // discriminator would otherwise pre-empt the fan-out. Every other recipe gets
    // a plain dataset (and a GAN also a noise node); an RL recipe gets an env.
    if (recipe.name === 'cgan') {
      if (roles.generator && roles.discriminator) ensureCganWiring(roles.generator, roles.discriminator)
      return
    }
    if (dataModelId && isEnvRecipe) ensureEnvFor(dataModelId)
    else if (dataModelId) ensureDatasetFor(dataModelId)
    if (recipe.name === 'gan' && roles.generator) ensureGanNoise(roles.generator)
  }, [recipe, isEnvRecipe, roles.generator, roles.discriminator, dataModelId, ensureDatasetFor, ensureEnvFor, ensureGanNoise, ensureCganWiring])

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
  // The current model's weights are at risk when it's a run whose weights aren't
  // saved — starting a new run replaces it and those weights are lost.
  const liveUnsaved =
    !!kernelRunName &&
    (checkpointMetas ?? []).some((c) => c.name === kernelRunName && c.has_weights === false)

  const saveKernelWeights = async (): Promise<boolean> => {
    if (!kernelRunName) return true
    try {
      await save.mutateAsync(kernelRunName)
      return true
    } catch {
      return false
    }
  }

  const doStartRun = async () => {
    setTrainingView('dashboard') // watch the run on the charts, not the preview
    setSideTab('runs') // a run started — show it landing in the runs list
    try {
      await runControls.start.mutateAsync(toProject())
    } catch (e) {
      setRunStatus('failed', e instanceof Error ? e.message : 'could not start the run')
    }
  }

  // From the Preview tab, warn before a new run discards the model you're
  // inspecting (when its weights aren't saved). The dashboard doesn't warn —
  // iterating runs there is the normal loop.
  const startRun = () => {
    if (trainingView === 'preview' && liveUnsaved) setPendingRun(true)
    else doStartRun()
  }

  const confirmRun = async (save: boolean) => {
    setPendingRun(false)
    if (save && !(await saveKernelWeights())) return // save failed — keep the model, don't run
    doStartRun()
  }

  const stopRun = () => runControls.stop.mutate()

  // The dashboard's numbers half (epoch table / "starting…" / readiness
  // checklist), shared by the two-column and full-width layouts.
  const epochResults = (
    <RunEpochsPanel
      epochs={runEpochs}
      bestEpoch={runBestEpoch}
      runState={runState}
      runError={runError}
      etaSecs={etaSecs}
      readiness={readiness}
    />
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
      onLayoutChange={setDashSplit}
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
        {/* Single-role recipe, several models: ▶ Run trains the ACTIVE model.
            The switcher makes that explicit and lets you retarget without
            leaving the tab (it re-scopes the runs list + dashboard too). */}
        {singleRoleMultiModel && (
          <div style={{ marginBottom: 16 }}>
            <label style={{ display: 'block', color: 'var(--text-5)', fontSize: 11, marginBottom: 4 }}>
              Training
            </label>
            <select
              value={activeModelId}
              onChange={(e) => openModel(e.target.value, { navigate: false })}
              title="▶ Run trains the model you're editing — switch it here or on the Models tab"
              style={selectStyle}
            >
              {models.map((m) => (
                <option key={m.id} value={m.id}>
                  {m.name}
                </option>
              ))}
            </select>
          </div>
        )}
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
        <RunDashboardHeader
          runState={runState}
          runConfig={runConfig}
          runSeed={runSeed}
          blocker={blocker}
          readinessUnavailable={readiness.status === 'unavailable'}
          showRun={showRun}
          isDashboard={trainingView === 'dashboard'}
          resultsOpen={resultsOpen}
          onToggleResults={() => setResultsOpen((o) => !o)}
          onRun={startRun}
          onStop={stopRun}
        />
        {/* The main area swaps with the Training sub-tab: the run dashboard
            (graphs + stats) or the input→output model preview. The side pane
            (settings + runs list) stays put across both. On the dashboard, once
            a run streams health, a full-width strip is anchored below the
            graphs|results columns — its own header can collapse it to reclaim
            the columns' vertical space. */}
        {trainingView === 'preview' ? (
          <PreviewView />
        ) : trainingView === 'optimize' ? (
          // Starting a sweep jumps to the Dashboard (watch trials stream) and
          // the Runs side pane — the same move ▶ Run makes.
          <OptimizeView
            onStarted={() => {
              setTrainingView('dashboard')
              setSideTab('runs')
            }}
          />
        ) : hasHealth ? (
          <div style={{ flex: 1, minHeight: 0, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
            <div style={{ flex: 1, minHeight: 0, display: 'flex' }}>{dashboardBody}</div>
            <TrainingHealthPanel
              roles={healthRoles}
              planned={plannedEpochs}
              // Mirror the graphs↔results split only when both columns are up;
              // otherwise the band falls back to its single full-width layout.
              split={
                showRun && resultsOpen
                  ? { graphs: dashSplit['dash-graphs'], results: dashSplit['dash-results'] }
                  : null
              }
            />
          </div>
        ) : (
          dashboardBody
        )}
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

      {/* Starting a run replaces the current model — warn (from Preview) when
          that would drop unsaved weights. */}
      {pendingRun && (
        <DiscardWeightsModal
          kernelRunName={kernelRunName}
          onCancel={() => setPendingRun(false)}
          onConfirm={confirmRun}
        />
      )}
    </Group>
  )
}
