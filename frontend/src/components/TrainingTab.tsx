import { useEffect, useRef } from 'react'
import { useGraphStore } from '../store/graphStore'
import { useRecipes } from '../hooks/useRecipes'
import { formatEpochLine } from '../lib/formatEpochLine'
import { formatShape } from '../lib/formatShape'
import { paramVisible } from '../lib/paramVisible'
import type { ParamDef } from '../types/graph'
import { Checkpoints } from './Checkpoints'
import { OptionalControl, ParamControl } from './Inspector'
import { RunCharts } from './RunCharts'

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

const sectionLabel: React.CSSProperties = {
  color: 'var(--text-6)',
  fontSize: 10,
  letterSpacing: 1,
  textTransform: 'uppercase',
  marginBottom: 8,
}

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
  const runState = useGraphStore((s) => s.runState)
  const runEpochs = useGraphStore((s) => s.runEpochs)
  const runError = useGraphStore((s) => s.runError)
  const runSeed = useGraphStore((s) => s.runSeed)
  const runBestEpoch = useGraphStore((s) => s.runBestEpoch)
  const setRunStatus = useGraphStore((s) => s.setRunStatus)

  const recipeName = (training.recipe as string) ?? 'supervised'
  const recipe = recipes?.find((r) => r.name === recipeName) ?? recipes?.[0]
  const multiRole = (recipe?.roles.length ?? 1) > 1
  const loopParams = recipe?.params ?? []
  const roles = (training.roles as Record<string, string>) ?? {}
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

  return (
    <div style={{ display: 'flex', flex: 1, overflow: 'hidden' }}>
      {/* Config form */}
      <div
        style={{
          width: 320,
          background: 'var(--panel)',
          borderRight: '1px solid var(--border)',
          padding: 20,
          overflowY: 'auto',
          fontFamily: 'monospace',
          flexShrink: 0,
        }}
      >
        <div style={{ color: 'var(--text)', fontWeight: 700, fontSize: 14, marginBottom: 4 }}>
          Training
        </div>
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

      {/* Run dashboard — live charts + epoch log. The generated train() opens
          via the titlebar's Show code button (a CodePanel, like the Model tab). */}
      <div style={{ flex: 1, display: 'flex', flexDirection: 'column', background: 'var(--bg)' }}>
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
          {/* The run's seed — reproducibility at a glance (sess.snapshot has the rest). */}
          {runState !== 'idle' && runSeed !== null && (
            <span style={{ marginLeft: 'auto', color: 'var(--text-6)' }}>seed {runSeed}</span>
          )}
          <span
            style={{
              marginLeft: runState !== 'idle' && runSeed !== null ? 0 : 'auto',
              color: RUN_STATE_COLOR[runState] ?? 'var(--text-6)',
            }}
          >
            {runState === 'idle' ? '' : runState}
          </span>
          {/* Trained weights exist after a completed (or stopped) single-model run.
              (Multi-model — e.g. GAN — checkpoints are a later slice.) */}
          {!multiRole && (runState === 'done' || runState === 'stopped') && (
            <a
              href="/api/run/weights"
              title="Download the trained weights + run snapshot (load with lamplighter.load_checkpoint)"
              style={{
                background: 'none', border: '1px solid var(--border)', borderRadius: 5,
                color: 'var(--text-3)', textDecoration: 'none', fontFamily: 'monospace',
                fontSize: 12, fontWeight: 600, padding: '3px 12px',
              }}
            >
              ⬇ Weights
            </a>
          )}
          {runState === 'running' ? (
            <button
              onClick={stopRun}
              style={{
                background: 'none', border: '1px solid var(--border)', borderRadius: 5,
                color: 'var(--error)', cursor: 'pointer', fontFamily: 'monospace',
                fontSize: 12, fontWeight: 600, padding: '3px 14px',
              }}
            >
              ■ Stop
            </button>
          ) : (
            <button
              onClick={startRun}
              title="Train in the notebook kernel using the picked data — runs exactly this code"
              style={{
                background: 'var(--accent)', border: 'none', borderRadius: 5,
                color: 'var(--text-on-accent)', cursor: 'pointer', fontFamily: 'monospace',
                fontSize: 12, fontWeight: 600, padding: '4px 16px',
              }}
            >
              ▶ Run
            </button>
          )}
        </div>
        {runEpochs.length === 0 && !runError ? (
          // Nothing streamed yet — point at the workflow instead of blank space.
          <div
            style={{
              flex: 1,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              fontFamily: 'monospace',
              fontSize: 12,
              color: 'var(--text-6)',
              padding: 24,
              textAlign: 'center',
              lineHeight: 1.8,
            }}
          >
            {runState === 'running'
              ? 'starting…'
              : 'Pick data in the Data tab, set the loop here, then press ▶ Run — live metrics stream in. (Show code previews the exact train() that runs.)'}
          </div>
        ) : (
          <div
            style={{
              flex: 1,
              display: 'flex',
              flexDirection: 'column',
              minHeight: 0,
              padding: '14px 20px',
              fontFamily: 'monospace',
              fontSize: 12,
              lineHeight: 1.6,
            }}
          >
            <RunCharts epochs={runEpochs} height={200} bestEpoch={runBestEpoch} />
            <div style={{ flex: 1, minHeight: 0, overflowY: 'auto' }}>
              {runEpochs.map((e) => (
                <div key={e.epoch} style={{ color: 'var(--text-3)', whiteSpace: 'pre' }}>
                  {formatEpochLine(e)}
                </div>
              ))}
              {runError && <div style={{ color: 'var(--error)' }}>✗ {runError}</div>}
              <div ref={epochsEndRef} />
            </div>
          </div>
        )}
        {/* Named checkpoints — single-model runs only for now. */}
        {!multiRole && <Checkpoints />}
      </div>
    </div>
  )
}
