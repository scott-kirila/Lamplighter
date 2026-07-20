// The Optimize view's config shape (mirrors the backend SweepManager's) and
// the eject path: the equivalent notebook script for the sweep the view runs —
// same params, same mechanism (trials as real managed runs) — so owning the
// loop is one copy-paste away, the same escape hatch as every code panel.

export interface SweepParamSpec {
  name: string // the Optuna key — unique (node params use "<nodeId>.<param>")
  label?: string // display-only (the backend ignores it)
  type: 'float' | 'int' | 'categorical'
  low?: number
  high?: number
  log?: boolean
  choices?: string[]
  // When set, the suggested value patches this node's param in the trial's
  // graph (an architecture sweep) instead of merging into project.training.
  node?: { model: string; node: string; param: string }
}

export interface SweepConfig {
  study?: string
  n_trials: number
  prune: boolean
  metric: string
  direction?: string // "minimize" (loss) | "maximize" (RL return)
  seed?: number
  params: SweepParamSpec[]
}

function suggestExpr(p: SweepParamSpec): string {
  if (p.type === 'float') {
    const log = p.log ? ', log=True' : ''
    return `trial.suggest_float("${p.name}", ${p.low}, ${p.high}${log})`
  }
  if (p.type === 'int') return `trial.suggest_int("${p.name}", ${p.low}, ${p.high})`
  const choices = (p.choices ?? []).map((c) => `"${c}"`).join(', ')
  return `trial.suggest_categorical("${p.name}", [${choices}])`
}

// The notebook-equivalent sweep. Runs in the SAME kernel as the app, so trials
// go through the same run manager and land in the same runs list (tagged
// source "notebook" so they read as yours, not the Optimize view's).
export function sweepScript(config: SweepConfig): string {
  const pruner = config.prune
    ? 'optuna.pruners.MedianPruner(n_startup_trials=1, n_warmup_steps=0)'
    : 'optuna.pruners.NopPruner()'
  const study = config.study ?? 'my-sweep'
  const loop = config.params.filter((p) => !p.node)
  const nodeSpecs = config.params.filter((p) => p.node)
  const nodeBlock = nodeSpecs.length
    ? `    _nodes = {n.id: n for m in p.models for n in m.graph.nodes}\n` +
      nodeSpecs
        .map((p) => `    _nodes["${p.node!.node}"].params["${p.node!.param}"] = ${suggestExpr(p)}`)
        .join('\n') + '\n'
    : ''
  return `# The sweep the Optimize view runs — paste into a notebook cell to own it.
# pip install "lamplighter[sweep]"
import optuna

from lamplighter.backend import state
from lamplighter.backend.runner import run_manager

project = state.get_project()  # the live canvas (same kernel as the app)


def objective(trial):
    p = project.model_copy(deep=True)
    p.training = {
        **(project.training or {}),
        ${loop.map((p) => `"${p.name}": ${suggestExpr(p)},`).join('\n        ')}
    }
${nodeBlock}    err = run_manager.start(p, source="notebook", study="${study}")
    assert err is None, err
    run_manager.join()
    assert run_manager.state == "done", run_manager.error
    return run_manager.history["${config.metric}"][-1]


study = optuna.create_study(
    direction="${config.direction ?? 'minimize'}",
    sampler=optuna.samplers.TPESampler(${config.seed != null ? `seed=${config.seed}` : ''}),
    pruner=${pruner},
)
study.optimize(objective, n_trials=${config.n_trials})
print("best:", study.best_params, study.best_value)
`
}
