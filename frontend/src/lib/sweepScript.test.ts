import { describe, expect, it } from 'vitest'
import { sweepScript } from './sweepScript'

describe('sweepScript (the Optimize view eject path)', () => {
  it('renders each param type as its optuna suggest call', () => {
    const src = sweepScript({
      study: 's1', n_trials: 12, prune: true, metric: 'val_loss', seed: 7,
      params: [
        { name: 'lr', type: 'float', low: 0.0001, high: 0.1, log: true },
        { name: 'epochs', type: 'int', low: 5, high: 20 },
        { name: 'optimizer', type: 'categorical', choices: ['Adam', 'SGD'] },
      ],
    })
    expect(src).toContain('"lr": trial.suggest_float("lr", 0.0001, 0.1, log=True),')
    expect(src).toContain('"epochs": trial.suggest_int("epochs", 5, 20),')
    expect(src).toContain('"optimizer": trial.suggest_categorical("optimizer", ["Adam", "SGD"]),')
    expect(src).toContain('study.optimize(objective, n_trials=12, catch=(AssertionError,))')
    expect(src).toContain('optuna.pruners.MedianPruner(n_startup_trials=1, n_warmup_steps=0)')
    expect(src).toContain('optuna.samplers.TPESampler(seed=7)')
    expect(src).toContain('run_manager.start(p, emit=on_event, source="notebook", study="s1")')
    expect(src).toContain('run_manager.history["val_loss"][-1]')
  })

  it('wires REAL pruning through the emit hook (and keeps tabs streaming)', () => {
    // A MedianPruner without trial.report would never prune — the script must
    // carry the same cooperative report → should_prune → stop loop the engine
    // runs, and forward events so open tabs keep streaming the trial.
    const src = sweepScript({ n_trials: 4, prune: true, metric: 'val_loss', params: [] })
    expect(src).toContain('from lamplighter.backend.ws import manager as _tabs')
    expect(src).toContain('_tabs.broadcast_threadsafe(msg)')
    expect(src).toContain('trial.report(float(value), step=int(msg["epoch"]))')
    expect(src).toContain('if trial.should_prune():')
    expect(src).toContain('run_manager.stop()')
    expect(src).toContain('raise optuna.TrialPruned()')
  })

  it('node-targeted params patch the trial graph instead of training', () => {
    const src = sweepScript({
      n_trials: 5, prune: true, metric: 'val_loss',
      params: [
        { name: 'lr', type: 'float', low: 0.001, high: 0.1, log: true },
        { name: 'l1.out_features', label: 'Linear · Out Features', type: 'int', low: 32, high: 256,
          node: { model: 'model', node: 'l1', param: 'out_features' } },
      ],
    })
    // The loop knob stays a training merge; the node param becomes graph surgery.
    expect(src).toContain('"lr": trial.suggest_float("lr", 0.001, 0.1, log=True),')
    expect(src).toContain('_nodes = {n.id: n for m in p.models for n in m.graph.nodes}')
    expect(src).toContain('_nodes["l1"].params["out_features"] = trial.suggest_int("l1.out_features", 32, 256)')
    expect(src).not.toContain('"l1.out_features":') // never merged into training
  })

  it('targets a data node config for a swept batch_size', () => {
    const src = sweepScript({
      n_trials: 4, prune: false, metric: 'val_loss',
      params: [
        { name: 'lr', type: 'float', low: 0.001, high: 0.1 },
        { name: 'd1.batch_size', label: 'Data · Batch Size (N)', type: 'int', low: 16, high: 128,
          data: { node: 'd1', param: 'batch_size' } },
      ],
    })
    expect(src).toContain('_data = {d.id: d for d in p.data_nodes}')
    expect(src).toContain('_data["d1"].config["batch_size"] = trial.suggest_int("d1.batch_size", 16, 128)')
    expect(src).not.toContain('"d1.batch_size":') // never merged into training
    expect(src).toContain('"lr": trial.suggest_float("lr", 0.001, 0.1),') // loop knobs unaffected
  })

  it('prune off means NopPruner and no emit hook; no seed means an unseeded sampler', () => {
    const src = sweepScript({ n_trials: 3, prune: false, metric: 'train_loss', params: [] })
    expect(src).toContain('optuna.pruners.NopPruner()')
    expect(src).toContain('optuna.samplers.TPESampler()')
    expect(src).toContain('study="my-sweep"') // the placeholder before a study exists
    // No hook: the run's default WS emit streams to tabs; nothing reports.
    expect(src).toContain('run_manager.start(p, source="notebook", study="my-sweep")')
    expect(src).not.toContain('on_event')
    expect(src).not.toContain('_tabs')
    expect(src).toContain('raise optuna.TrialPruned()') // a hand-stopped trial still isn't a crash
  })

  it('carries the direction — an RL return sweep maximizes', () => {
    const rl = sweepScript({ n_trials: 8, prune: true, metric: 'mean_return', direction: 'maximize', params: [] })
    expect(rl).toContain('direction="maximize"')
    expect(rl).toContain('run_manager.history["mean_return"][-1]')
    // Default stays minimize when unspecified.
    expect(sweepScript({ n_trials: 3, prune: false, metric: 'val_loss', params: [] })).toContain('direction="minimize"')
  })
})
