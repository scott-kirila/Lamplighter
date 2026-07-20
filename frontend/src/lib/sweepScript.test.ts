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
    expect(src).toContain('study.optimize(objective, n_trials=12)')
    expect(src).toContain('optuna.pruners.MedianPruner(n_startup_trials=1, n_warmup_steps=0)')
    expect(src).toContain('optuna.samplers.TPESampler(seed=7)')
    expect(src).toContain('run_manager.start(p, source="notebook", study="s1")')
    expect(src).toContain('run_manager.history["val_loss"][-1]')
  })

  it('prune off means NopPruner; no seed means an unseeded sampler', () => {
    const src = sweepScript({ n_trials: 3, prune: false, metric: 'train_loss', params: [] })
    expect(src).toContain('optuna.pruners.NopPruner()')
    expect(src).toContain('optuna.samplers.TPESampler()')
    expect(src).toContain('study="my-sweep"') // the placeholder before a study exists
  })
})
