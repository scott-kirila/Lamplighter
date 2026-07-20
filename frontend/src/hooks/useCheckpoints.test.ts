import { describe, expect, it } from 'vitest'
import { belongsToModel, isSweepTrial, type CheckpointMeta } from './useCheckpoints'

const meta = (models?: CheckpointMeta['models']): CheckpointMeta => ({
  name: 'run-1', created: '2026-07-19T00:00:00', epoch: 1, epochs: 1,
  best_epoch: null, seed: 1, val_loss: null, state: 'done', source: 'app',
  has_weights: false, auto: true, models,
})

describe('belongsToModel (the Runs list scoping predicate)', () => {
  it('matches a run attributed to the model', () => {
    expect(belongsToModel(meta([{ id: 'm1', name: 'Alpha', role: 'model' }]), 'm1')).toBe(true)
  })

  it('rejects a run attributed to a different model', () => {
    expect(belongsToModel(meta([{ id: 'm2', name: 'Beta', role: 'model' }]), 'm1')).toBe(false)
  })

  it('treats pre-attribution runs (no models key) as belonging everywhere', () => {
    expect(belongsToModel(meta(undefined), 'm1')).toBe(true)
  })

  it('treats an EMPTY models list as unattributed, not belonging-to-nobody', () => {
    // A zero-length attribution would otherwise hide the run from every scope
    // except show-all — hiding history is worse than over-showing it.
    expect(belongsToModel(meta([]), 'm1')).toBe(true)
  })
})

describe('isSweepTrial (what the Runs list tucks into the Optimize view)', () => {
  it('hides the Optimize view’s own auto trials, keeps everything kept', () => {
    const trial = { ...meta(), source: 'sweep', study: 's1', auto: true }
    expect(isSweepTrial(trial)).toBe(true)
    // The crowned best: study-tagged but auto=false (saved + renamed) → shows.
    expect(isSweepTrial({ ...meta(), source: 'sweep', study: 's1', auto: false })).toBe(false)
    // Regular runs always show, auto or not.
    expect(isSweepTrial(meta())).toBe(false)
    expect(isSweepTrial({ ...meta(), study: null })).toBe(false)
  })

  it('an ejected notebook-script sweep stays visible — its runs are the user’s own', () => {
    // The script tags a study too (for grouping), but its source is
    // "notebook": tucking keys on the SOURCE, never the study tag alone.
    expect(isSweepTrial({ ...meta(), source: 'notebook', study: 'my-sweep', auto: true })).toBe(false)
  })
})
