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

  it('keeps a bounded rolling window, dropping the oldest', () => {
    for (let i = 1; i <= 1005; i++) store().appendRunStep(i, { train_loss: 1 / i }, 1005)
    const buf = store().stepMetrics
    expect(buf).toHaveLength(1000)
    expect(buf[0].step).toBe(6) // steps 1..5 dropped
    expect(buf[buf.length - 1].step).toBe(1005)
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
