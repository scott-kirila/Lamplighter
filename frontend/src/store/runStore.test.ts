import { beforeEach, describe, expect, it } from 'vitest'
import { epochsFromHistory, useRunStore } from './runStore'

const store = useRunStore.getState
beforeEach(() => store().reset())

describe('epochsFromHistory', () => {
  it('rebuilds the per-epoch stream from metric series', () => {
    const epochs = epochsFromHistory({ train_loss: [1, 0.5], val_loss: [0.9, 0.6] }, 10)
    expect(epochs).toEqual([
      { epoch: 1, epochs: 10, metrics: { train_loss: 1, val_loss: 0.9 } },
      { epoch: 2, epochs: 10, metrics: { train_loss: 0.5, val_loss: 0.6 } },
    ])
  })

  it('omits metrics whose series never ran (empty val without a val_loader)', () => {
    const epochs = epochsFromHistory({ train_loss: [1], val_loss: [] }, 5)
    expect(epochs).toEqual([{ epoch: 1, epochs: 5, metrics: { train_loss: 1 } }])
  })

  it('handles a null history (idle)', () => {
    expect(epochsFromHistory(null, 0)).toEqual([])
  })
})

describe('run hydration + event merging', () => {
  const e = (n: number) => ({ epoch: n, epochs: 5, metrics: { train_loss: 1 / n } })

  it('appendRunEpoch ignores an epoch at/behind the newest (hydration race)', () => {
    store().appendRunEpoch(e(1))
    store().appendRunEpoch(e(2))
    store().appendRunEpoch(e(2)) // duplicate delivery
    store().appendRunEpoch(e(1)) // stale
    expect(store().runEpochs.map((x) => x.epoch)).toEqual([1, 2])
  })

  it('hydrateRun seeds a late-joining tab', () => {
    store().hydrateRun('running', null, [e(1), e(2), e(3)])
    expect(store().runState).toBe('running')
    expect(store().runEpochs).toHaveLength(3)
  })

  it('hydrateRun never downgrades live state or a longer epoch list', () => {
    store().setRunStatus('done', null)
    store().appendRunEpoch(e(1))
    store().appendRunEpoch(e(2))
    // A stale fetch resolving late must not overwrite what the WS delivered.
    store().hydrateRun('running', null, [e(1)])
    expect(store().runState).toBe('done')
    expect(store().runEpochs).toHaveLength(2)
  })

  it('replaceRun overwrites the shown run wholesale (checkpoint restore)', () => {
    store().setRunStatus('failed', 'boom', 99, null)
    store().appendRunEpoch(e(1))
    store().appendRunEpoch(e(2))
    // Restoring a checkpoint must replace everything — even a shorter history.
    store().replaceRun('done', null, [e(1)], 3, 1)
    expect(store().runState).toBe('done')
    expect(store().runError).toBeNull()
    expect(store().runEpochs).toHaveLength(1)
    expect(store().runSeed).toBe(3)
    expect(store().runBestEpoch).toBe(1)
  })

  it('replaceRun moves the kernel run when given a kernelName (restore/resume)', () => {
    // A run finished, its name streamed in — the kernel holds run-2.
    store().setRunStatus('done', null, 1, null, null, 'run-2')
    expect(store().kernelRunName).toBe('run-2')
    // Restoring run-1 hands the kernel to run-1: "keep weights" must follow it.
    store().replaceRun('done', null, [e(1)], 3, 1, [], 0, null, 'run-1', 'run-1')
    expect(store().runName).toBe('run-1')
    expect(store().kernelRunName).toBe('run-1')
  })

  it('replaceRun without a kernelName leaves the kernel run untouched (read-only view)', () => {
    store().setRunStatus('done', null, 1, null, null, 'run-2')
    // Viewing run-1 shows it, but the kernel still holds run-2.
    store().replaceRun('done', null, [e(1)], 3, 1, [], 0, null, 'run-1')
    expect(store().runName).toBe('run-1')
    expect(store().kernelRunName).toBe('run-2')
  })

  it('setRunStatus clears the previous run lines on entering "running"', () => {
    store().replaceRun('done', null, [e(1), e(2)])
    store().setRunStatus('running', null)
    expect(store().runEpochs).toEqual([]) // a fresh run starts with a blank panel
  })

  it('reset returns to idle with no curves (a new project)', () => {
    store().replaceRun('done', null, [e(1)], 7, 1)
    store().reset()
    expect(store()).toMatchObject({
      runState: 'idle', runEpochs: [], runError: null, runSeed: null, runBestEpoch: null,
    })
  })
})

describe('per-step metrics buffer', () => {
  it('accumulates points (metrics per step) in order and records the run total', () => {
    store().appendRunStep(1, { g_loss: 1.4, d_loss: 0.7 }, 200)
    store().appendRunStep(4, { g_loss: 1.2, d_loss: 0.9 }, 200) // throttled — not contiguous
    expect(store().stepMetrics).toEqual([
      { step: 1, metrics: { g_loss: 1.4, d_loss: 0.7 } },
      { step: 4, metrics: { g_loss: 1.2, d_loss: 0.9 } },
    ])
    expect(store().stepTotal).toBe(200) // the fixed x-axis extent
  })

  it('halves resolution at the cap instead of sliding — the run start survives', () => {
    // The chart's x-axis spans the whole run: a sliding window shrank long runs
    // to a sliver at the right edge (the "step chart disappears" bug). Thinning
    // keeps points from step 1 to the newest, at coarser density.
    for (let i = 1; i <= 12000; i++) store().appendRunStep(i, { train_loss: 1 / i }, 12000)
    const buf = store().stepMetrics
    expect(buf.length).toBeLessThanOrEqual(4000)
    expect(buf.length).toBeGreaterThan(1000) // still dense, not decimated to nothing
    expect(buf[0].step).toBe(1) // the start is never forgotten
    expect(buf[buf.length - 1].step).toBe(12000) // the newest point is kept
    // Strictly increasing — thinning never reorders.
    for (let i = 1; i < buf.length; i++) expect(buf[i].step).toBeGreaterThan(buf[i - 1].step)
  })

  it('carries the run-recorded config on status events and hydration', () => {
    store().setRunStatus('running', null, 7, null, { recipe: 'cgan', epochs: 80, device: 'cpu' })
    expect(store().runConfig).toEqual({ recipe: 'cgan', epochs: 80, device: 'cpu' })
    // A later event without config keeps the recorded one.
    store().setRunStatus('done', null)
    expect(store().runConfig?.recipe).toBe('cgan')
    // A fresh run without config clears it (a new run owns the label).
    store().setRunStatus('running', null)
    expect(store().runConfig).toBeNull()
    // Hydration seeds it when empty, e.g. after a refresh.
    store().hydrateRun('done', null, [], null, null, [], 0, { recipe: 'supervised', epochs: 10 })
    expect(store().runConfig?.epochs).toBe(10)
  })

  it('hydrates step points from the backend buffer (a refreshed tab keeps the chart)', () => {
    store().hydrateRun('running', null, [], null, null, [{ step: 3, metrics: { train_loss: 0.4 } }], 200)
    expect(store().stepMetrics).toEqual([{ step: 3, metrics: { train_loss: 0.4 } }])
    expect(store().stepTotal).toBe(200)
  })

  it('hydration never clobbers points this tab already streamed (live wins)', () => {
    store().setRunStatus('running', null)
    store().appendRunStep(9, { train_loss: 0.2 }, 200)
    store().hydrateRun('running', null, [], null, null, [{ step: 1, metrics: { train_loss: 0.9 } }], 200)
    expect(store().stepMetrics).toEqual([{ step: 9, metrics: { train_loss: 0.2 } }])
  })

  it('clears (points and total) on a fresh run and on reset', () => {
    store().appendRunStep(1, { train_loss: 0.5 }, 200)
    store().setRunStatus('running', null) // idle -> running is a fresh run
    expect(store().stepMetrics).toEqual([])
    expect(store().stepTotal).toBe(0)

    store().appendRunStep(1, { train_loss: 0.5 }, 200)
    store().reset()
    expect(store().stepMetrics).toEqual([])
    expect(store().stepTotal).toBe(0)
  })
})
