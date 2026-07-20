import { beforeEach, describe, expect, it } from 'vitest'
import { useSweepStore, type SweepStatus } from './sweepStore'

const store = useSweepStore.getState
const status = (over: Partial<SweepStatus> = {}): SweepStatus => ({
  state: 'running', error: null, study: 's1', n_trials: 5, trial: 2,
  completed: 1, pruned: 0, failed: 0, metric: 'val_loss', direction: 'minimize',
  best: { run_name: 'run-1', value: 0.4, params: { lr: 0.01 } },
  ...over,
})

beforeEach(() =>
  useSweepStore.setState({
    state: 'idle', error: null, study: null, n_trials: 0, trial: null,
    completed: 0, pruned: 0, failed: 0, metric: 'val_loss', direction: 'minimize', best: null,
  })
)

describe('sweepStore', () => {
  it('applies WS sweep_status events wholesale', () => {
    store().setSweepStatus(status())
    expect(store().state).toBe('running')
    expect(store().trial).toBe(2)
    expect(store().best?.value).toBe(0.4)
    store().setSweepStatus(status({ state: 'done', trial: null, completed: 5 }))
    expect(store().state).toBe('done')
    expect(store().completed).toBe(5)
  })

  it('hydrates only an idle store — live events win over the fetch', () => {
    store().hydrateSweep(status({ state: 'done', completed: 5 }))
    expect(store().state).toBe('done') // idle → seeded

    store().setSweepStatus(status({ state: 'running', completed: 2 }))
    store().hydrateSweep(status({ state: 'done', completed: 5 })) // stale fetch resolves late
    expect(store().state).toBe('running') // not clobbered
    expect(store().completed).toBe(2)
  })
})
