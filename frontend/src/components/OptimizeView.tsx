import { useState } from 'react'
import { useGraphStore } from '../store/graphStore'
import { useRunStore } from '../store/runStore'
import { useSweepStore } from '../store/sweepStore'
import { useCheckpoints } from '../hooks/useCheckpoints'
import { useRecipes } from '../hooks/useRecipes'
import { useRegistry } from '../hooks/useRegistry'
import { useRunView } from '../hooks/useRunView'
import { useSweepControls } from '../hooks/useSweepControls'
import { sweepScript, type SweepParamSpec } from '../lib/sweepScript'
import type { ParamDef } from '../types/graph'
import { button, chip, eyebrow, field } from '../styles/ui'

// Knobs where a sweep makes no sense (identity/bookkeeping, not capacity).
const UNSWEEPABLE = new Set(['device', 'seed', 'autosave_every', 'metric', 'early_stop_patience'])

const label: React.CSSProperties = { color: 'var(--text-5)', fontSize: 11 }
const section: React.CSSProperties = { ...eyebrow, color: 'var(--text-6)', fontSize: 10, margin: '18px 0 8px' }
const num: React.CSSProperties = { ...field, width: 76, padding: '3px 6px' }

// Sensible starting range around the knob's current value: an order of
// magnitude each way for floats (log-scaled when it lives below 1, like lr),
// halved/doubled for ints. All editable — this is a prefill, not a policy.
function specFor(p: ParamDef, current: unknown): SweepParamSpec {
  if (p.type === 'enum') return { name: p.name, type: 'categorical', choices: [...(p.choices ?? [])] }
  const v = Number(current ?? p.default)
  if (p.type === 'int') {
    const base = Number.isFinite(v) && v > 0 ? v : 10
    return { name: p.name, type: 'int', low: Math.max(1, Math.round(base / 2)), high: base * 2 }
  }
  const pos = Number.isFinite(v) && v > 0
  return {
    name: p.name, type: 'float',
    low: pos ? v / 10 : 0, high: pos ? v * 10 : 1, log: pos && v < 1,
  }
}

// The Optimize view: configure a hyperparameter sweep and run it in-kernel —
// N SEQUENTIAL trials, each a real recorded run. Starting jumps to the
// Dashboard (via onStarted) so you watch trials stream; the config draft lives
// in the sweep store, so it's intact when you come back. Trials are tucked out
// of the Runs list (their table is HERE); the best trial's weights are kept as
// they happen and crowned <study>-best. The notebook script pane is the eject
// path — the same sweep as copyable code.
export function OptimizeView({ onStarted }: { onStarted?: () => void } = {}) {
  const training = useGraphStore((s) => s.training)
  const toProject = useGraphStore((s) => s.toProject)
  const nodes = useGraphStore((s) => s.nodes)
  const activeModelId = useGraphStore((s) => s.activeModelId)
  const models = useGraphStore((s) => s.models)
  const { data: recipes } = useRecipes()
  const { data: registry } = useRegistry()
  const sweep = useSweepStore()
  const runState = useRunStore((s) => s.runState)
  const { data: checkpoints } = useCheckpoints()
  const viewRun = useRunView()
  const { start, stop } = useSweepControls()

  // The draft rides the sweep store (starting unmounts this view — see above);
  // error/script-visibility are honestly transient, so they stay local.
  const { params, nTrials, prune, metric } = useSweepStore((s) => s.draft)
  const setDraft = useSweepStore((s) => s.setDraft)
  const setParams = (upd: (ps: SweepParamSpec[]) => SweepParamSpec[]) =>
    setDraft({ params: upd(useSweepStore.getState().draft.params) })
  const [error, setError] = useState<string | null>(null)
  const [showScript, setShowScript] = useState(false)

  const running = sweep.state === 'running'
  const recipeName = (training.recipe as string) ?? 'supervised'
  const recipe = recipes?.find((r) => r.name === recipeName) ?? recipes?.[0]
  // The objective is recipe-shaped: an RL recipe MAXIMIZES the mean return; a
  // supervised recipe MINIMIZES a loss. The sweep engine is direction-generic;
  // this is the only place that knew otherwise.
  const isRL = recipe?.data === 'env'
  const metricOptions = isRL ? ['mean_return'] : ['val_loss', 'train_loss']
  const direction = isRL ? 'maximize' : 'minimize'
  const effectiveMetric = metricOptions.includes(metric) ? metric : metricOptions[0]
  const addable = (recipe?.params ?? []).filter(
    (p) =>
      !UNSWEEPABLE.has(p.name) &&
      ['float', 'int', 'enum'].includes(p.type) &&
      !params.some((sp) => sp.name === p.name)
  )

  // The ACTIVE model's numeric node params — architecture sweeps (hidden dims,
  // dropout p). Node labels use the node's name param, else its type; repeats
  // get #k so two Linears read apart.
  const activeModelName = models.find((m) => m.id === activeModelId)?.name ?? 'Model'
  const seen: Record<string, number> = {}
  const nodeOptions = nodes.flatMap((n) => {
    const def = registry?.[n.data.nodeType]
    if (!def || n.data.nodeType === 'Input' || n.data.nodeType === 'Output') return []
    const base = String(n.data.params.name ?? '').trim() || n.data.nodeType
    seen[base] = (seen[base] ?? 0) + 1
    const nodeLabel = seen[base] > 1 ? `${base} #${seen[base]}` : base
    return def.params
      .filter((p) => ['float', 'int'].includes(p.type))
      .map((p) => ({
        value: `node:${n.id}:${p.name}`,
        label: `${nodeLabel} · ${p.label}`,
        nodeId: n.id,
        param: p,
        current: n.data.params[p.name],
      }))
      .filter((o) => !params.some((sp) => sp.name === `${o.nodeId}.${o.param.name}`))
  })

  const addSelection = (value: string) => {
    if (value.startsWith('node:')) {
      const opt = nodeOptions.find((o) => o.value === value)
      if (!opt) return
      setParams((ps) => [
        ...ps,
        {
          ...specFor(opt.param, opt.current),
          name: `${opt.nodeId}.${opt.param.name}`,
          label: opt.label,
          node: { model: activeModelId, node: opt.nodeId, param: opt.param.name },
        },
      ])
      return
    }
    const def = addable.find((p) => p.name === value)
    if (def) setParams((ps) => [...ps, { ...specFor(def, training[def.name]), label: def.label }])
  }

  const patch = (name: string, upd: Partial<SweepParamSpec>) =>
    setParams((ps) => ps.map((p) => (p.name === name ? { ...p, ...upd } : p)))

  const toggleChoice = (p: SweepParamSpec, choice: string) => {
    const has = p.choices?.includes(choice)
    if (has && (p.choices?.length ?? 0) <= 1) return // categorical needs ≥1
    patch(p.name, { choices: has ? p.choices!.filter((c) => c !== choice) : [...(p.choices ?? []), choice] })
  }

  const config = { n_trials: nTrials, prune, metric: effectiveMetric, direction, params }

  const startSweep = async () => {
    setError(null)
    try {
      await start.mutateAsync({ project: toProject(), config })
      onStarted?.() // watch the trials stream on the Dashboard
    } catch (e) {
      setError(e instanceof Error ? e.message : 'could not start the sweep')
    }
  }

  const trials = (checkpoints ?? []).filter((c) => sweep.study != null && c.study === sweep.study)
  const bestName = sweep.best?.run_name ?? null
  // Each trial's objective (a run's meta doesn't carry the sweep metric — the
  // engine records it in the status). Formatted by the RUNNING sweep's metric,
  // which may differ from the current recipe's.
  const trialValue = new Map(sweep.trials.map((t) => [t.name, t.value]))
  const fmtObjective = (v: number) => (sweep.metric === 'mean_return' ? v.toFixed(1) : v.toFixed(4))

  return (
    <div style={{ flex: 1, minHeight: 0, overflowY: 'auto', background: 'var(--bg)' }}>
      <div style={{ maxWidth: 780, padding: '16px 20px 32px', fontFamily: 'monospace', fontSize: 12 }}>
        {/* -- config ------------------------------------------------------- */}
        <div style={section}>Sweep</div>
        <div style={{ opacity: running ? 0.55 : 1, pointerEvents: running ? 'none' : 'auto' }}>
          {params.length === 0 && (
            <div style={{ color: 'var(--text-6)', marginBottom: 10 }}>
              pick a hyperparameter to sweep — each trial trains the current project with the
              suggested values merged into its training config
            </div>
          )}
          {params.map((p) => (
            <div key={p.name} style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 8, flexWrap: 'wrap' }}>
              <code style={{ ...chip, minWidth: 92, textAlign: 'center' }}>{p.label ?? p.name}</code>
              {p.type === 'categorical' ? (
                (recipe?.params.find((rp) => rp.name === p.name)?.choices ?? p.choices ?? []).map((c) => (
                  <button
                    key={c}
                    onClick={() => toggleChoice(p, c)}
                    title={p.choices?.includes(c) ? 'Exclude from the sweep' : 'Include in the sweep'}
                    style={{
                      ...button,
                      color: p.choices?.includes(c) ? 'var(--accent)' : 'var(--text-6)',
                      borderColor: p.choices?.includes(c) ? 'var(--accent)' : 'var(--border)',
                    }}
                  >
                    {c}
                  </button>
                ))
              ) : (
                <>
                  <span style={label}>low</span>
                  <input type="number" value={p.low} style={num}
                    onChange={(e) => patch(p.name, { low: Number(e.target.value) })} />
                  <span style={label}>high</span>
                  <input type="number" value={p.high} style={num}
                    onChange={(e) => patch(p.name, { high: Number(e.target.value) })} />
                  {p.type === 'float' && (
                    <label style={{ ...label, display: 'flex', alignItems: 'center', gap: 4, cursor: 'pointer' }}>
                      <input type="checkbox" checked={!!p.log}
                        onChange={(e) => patch(p.name, { log: e.target.checked })} />
                      log
                    </label>
                  )}
                </>
              )}
              <button
                onClick={() => setParams((ps) => ps.filter((x) => x.name !== p.name))}
                title="Remove from the sweep"
                style={{ ...button, marginLeft: 'auto' }}
              >
                ✕
              </button>
            </div>
          ))}
          {(addable.length > 0 || nodeOptions.length > 0) && (
            <select
              value=""
              onChange={(e) => addSelection(e.target.value)}
              style={{ ...field, padding: '4px 8px', fontSize: 12, marginBottom: 12 }}
            >
              <option value="" disabled>
                ＋ sweep a parameter…
              </option>
              {addable.length > 0 && (
                <optgroup label="Training">
                  {addable.map((p) => (
                    <option key={p.name} value={p.name}>
                      {p.label}
                    </option>
                  ))}
                </optgroup>
              )}
              {nodeOptions.length > 0 && (
                <optgroup label={`Model — ${activeModelName}`}>
                  {nodeOptions.map((o) => (
                    <option key={o.value} value={o.value}>
                      {o.label}
                    </option>
                  ))}
                </optgroup>
              )}
            </select>
          )}

          <div style={{ display: 'flex', alignItems: 'center', gap: 16, flexWrap: 'wrap', marginTop: 4 }}>
            <label style={{ ...label, display: 'flex', alignItems: 'center', gap: 6 }}>
              trials
              <input type="number" min={1} value={nTrials} style={num}
                onChange={(e) => setDraft({ nTrials: Math.max(1, Number(e.target.value) || 1) })} />
            </label>
            <label style={{ ...label, display: 'flex', alignItems: 'center', gap: 6 }}>
              {direction === 'maximize' ? 'maximize' : 'minimize'}
              <select value={effectiveMetric} onChange={(e) => setDraft({ metric: e.target.value })}
                disabled={metricOptions.length === 1}
                style={{ ...field, padding: '3px 6px' }}>
                {metricOptions.map((m) => (
                  <option key={m} value={m}>{m}</option>
                ))}
              </select>
            </label>
            <label style={{ ...label, display: 'flex', alignItems: 'center', gap: 4, cursor: 'pointer' }}
              title="Stop unpromising trials early (median rule) — the pruned trial records as a stopped run">
              <input type="checkbox" checked={prune} onChange={(e) => setDraft({ prune: e.target.checked })} />
              prune bad trials
            </label>
          </div>
          {!isRL && effectiveMetric === 'val_loss' && (
            <div style={{ color: 'var(--text-6)', fontSize: 11, marginTop: 6 }}>
              val_loss needs a validation split on the data node
            </div>
          )}
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginTop: 14 }}>
          {running ? (
            <button onClick={() => stop.mutate()} style={{ ...button, color: 'var(--error)', borderColor: 'var(--error)' }}>
              ■ Stop sweep
            </button>
          ) : (
            <button
              onClick={startSweep}
              disabled={params.length === 0 || runState === 'running'}
              title={runState === 'running' ? 'A run is in progress — stop it first' : 'Run the sweep (trials are sequential — one kernel, one run at a time)'}
              style={{
                ...button,
                color: 'var(--accent)', borderColor: 'var(--accent)', padding: '4px 14px',
                opacity: params.length === 0 || runState === 'running' ? 0.45 : 1,
              }}
            >
              ▶ Start sweep
            </button>
          )}
          {error && <span style={{ color: 'var(--error)' }}>✗ {error}</span>}
        </div>

        {/* -- progress + trials -------------------------------------------- */}
        {sweep.study != null && (
          <>
            <div style={section}>
              {sweep.study} — {running ? `running · trial ${sweep.trial ?? '…'} of ${sweep.n_trials}` : sweep.state}
            </div>
            {sweep.error && <div style={{ color: 'var(--error)', marginBottom: 8 }}>✗ {sweep.error}</div>}
            <div style={{ color: 'var(--text-5)', marginBottom: 10 }}>
              {sweep.completed} complete · {sweep.pruned} pruned · {sweep.failed} failed
              {sweep.best && (
                <>
                  {' · best '}
                  <span style={{ color: 'var(--accent)' }}>
                    {sweep.metric} {fmtObjective(sweep.best.value)}
                  </span>
                </>
              )}
            </div>

            {/* Which params moved the metric — PedAnova over the completed
                trials, computed at sweep end. Bars scale to the top param. */}
            {sweep.importance && Object.keys(sweep.importance).length > 0 && (
              <div style={{ marginBottom: 14 }}>
                {Object.entries(sweep.importance)
                  .sort(([, a], [, b]) => b - a)
                  .map(([name, value]) => {
                    const max = Math.max(...Object.values(sweep.importance!))
                    const display = params.find((p) => p.name === name)?.label ?? name
                    return (
                      <div key={name} style={{ display: 'flex', alignItems: 'center', gap: 10, padding: '2px 0' }}>
                        <span style={{ color: 'var(--text-5)', width: 190, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                          {display}
                        </span>
                        <div style={{ width: 180, height: 8, background: 'var(--field)', border: '1px solid var(--border)', borderRadius: 3 }}>
                          <div style={{ width: `${max > 0 ? (value / max) * 100 : 0}%`, height: '100%', background: 'var(--accent)', borderRadius: 2 }} />
                        </div>
                        <span style={{ color: 'var(--text-5)' }}>{value.toFixed(2)}</span>
                      </div>
                    )
                  })}
              </div>
            )}

            {trials.map((c, i) => (
              <div
                key={c.name}
                onClick={() => runState !== 'running' && viewRun(c.name)}
                title={runState === 'running' ? undefined : 'Show this trial on the Dashboard'}
                style={{
                  display: 'flex', alignItems: 'center', gap: 10, padding: '5px 8px 5px 5px',
                  borderLeft: `3px solid ${c.name === bestName ? 'var(--accent)' : 'transparent'}`,
                  borderBottom: '1px solid var(--border)',
                  cursor: runState === 'running' ? 'default' : 'pointer',
                }}
              >
                <span style={{ color: 'var(--text-6)', width: 26, textAlign: 'right' }}>{i + 1}</span>
                <span
                  title={c.state ?? 'done'}
                  style={{
                    width: 12, textAlign: 'center',
                    color: c.state === 'failed' ? 'var(--error)' : c.state === 'stopped' ? 'var(--text-5)' : 'hsl(120, 70%, 45%)',
                  }}
                >
                  {c.state === 'failed' ? '✕' : c.state === 'stopped' ? '■' : '✓'}
                </span>
                <span style={{ color: c.name === bestName ? 'var(--accent)' : 'var(--text)', fontWeight: 600 }}>
                  {c.name}
                </span>
                {c.name === bestName && (
                  <span style={{ ...eyebrow, fontSize: 9, color: 'var(--accent)', border: '1px solid var(--accent)', borderRadius: 4, padding: '0 4px' }}>
                    best
                  </span>
                )}
                <span style={{ marginLeft: 'auto', color: 'var(--text-5)' }}>
                  {trialValue.get(c.name) != null ? `${sweep.metric} ${fmtObjective(trialValue.get(c.name)!)}` : '—'}
                </span>
              </div>
            ))}
            {trials.length === 0 && running && (
              <div style={{ color: 'var(--text-6)' }}>first trial running — it lands here when it records…</div>
            )}
          </>
        )}

        {/* -- eject -------------------------------------------------------- */}
        <div style={section}>Own the loop</div>
        <button onClick={() => setShowScript((s) => !s)} style={button}>
          {showScript ? 'Hide' : 'Show'} notebook script
        </button>
        {showScript && (
          <div style={{ position: 'relative', marginTop: 8 }}>
            <button
              onClick={() => navigator.clipboard?.writeText(sweepScript({ ...config, study: sweep.study ?? undefined }))}
              style={{ ...button, position: 'absolute', top: 6, right: 6 }}
            >
              Copy
            </button>
            <pre
              style={{
                background: 'var(--panel)', border: '1px solid var(--border)', borderRadius: 6,
                padding: 12, overflowX: 'auto', fontSize: 11, lineHeight: 1.5, color: 'var(--text-2)',
              }}
            >
              {sweepScript({ ...config, study: sweep.study ?? undefined })}
            </pre>
          </div>
        )}
      </div>
    </div>
  )
}
