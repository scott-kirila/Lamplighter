import { describe, expect, it } from 'vitest'
import { paramVisible, sweepOfferable, sweepableChoices } from './paramVisible'
import type { ParamDef } from '../types/graph'

const p = (name: string, show_if?: Record<string, unknown>): ParamDef => ({
  name,
  label: name,
  type: 'string',
  default: '',
  show_if,
})

describe('paramVisible', () => {
  it('always shows a param with no show_if', () => {
    expect(paramVisible(p('batch_size'), { source: 'tensors' })).toBe(true)
  })

  it('hides a param whose show_if does not match the effective config', () => {
    expect(paramVisible(p('dataset', { source: 'torchvision' }), { source: 'tensors' })).toBe(false)
  })

  it('shows a param whose show_if matches', () => {
    expect(paramVisible(p('dataset', { source: 'torchvision' }), { source: 'torchvision' })).toBe(true)
  })

  it('requires every show_if key to match', () => {
    const param = p('x', { source: 'torchvision', dataset: 'MNIST' })
    expect(paramVisible(param, { source: 'torchvision', dataset: 'MNIST' })).toBe(true)
    expect(paramVisible(param, { source: 'torchvision', dataset: 'CIFAR10' })).toBe(false)
  })

  it('treats an array rule value as membership', () => {
    const param = p('root', { source: ['torchvision', 'imagefolder'] })
    expect(paramVisible(param, { source: 'imagefolder' })).toBe(true)
    expect(paramVisible(param, { source: 'torchvision' })).toBe(true)
    expect(paramVisible(param, { source: 'tensors' })).toBe(false)
  })
})

describe('sweepOfferable (the Optimize picker gate for conditional knobs)', () => {
  const momentum = p('momentum', { optimizer: ['SGD', 'RMSprop'] })
  const stepSize = p('step_size', { scheduler: 'StepLR' })

  it('ungated knobs are always offerable', () => {
    expect(sweepOfferable(p('lr'), { optimizer: 'Adam' }, [])).toBe(true)
  })

  it('a gated knob under a non-matching FIXED controller is a no-op — not offered', () => {
    // momentum under Adam: suggested, merged, ignored by codegen (proven live).
    expect(sweepOfferable(momentum, { optimizer: 'Adam' }, [])).toBe(false)
    expect(sweepOfferable(stepSize, { scheduler: 'none' }, [])).toBe(false)
  })

  it('offered when the effective config already matches (the plain form rule)', () => {
    expect(sweepOfferable(momentum, { optimizer: 'SGD' }, [])).toBe(true)
    expect(sweepOfferable(stepSize, { scheduler: 'StepLR' }, [])).toBe(true)
  })

  it('SWEEPING the controller with a satisfying choice unlocks the knob', () => {
    // optimizer ∈ {Adam, SGD} → momentum is live in the SGD trials: a real
    // conditional sweep, not a no-op.
    const swept = [{ name: 'optimizer', choices: ['Adam', 'SGD'] }]
    expect(sweepOfferable(momentum, { optimizer: 'Adam' }, swept)).toBe(true)
  })

  it('a swept controller with NO satisfying choice keeps the knob out', () => {
    const swept = [{ name: 'optimizer', choices: ['Adam', 'AdamW'] }]
    expect(sweepOfferable(momentum, { optimizer: 'Adam' }, swept)).toBe(false)
  })
})

describe('sweepableChoices (modes a trial cannot supply)', () => {
  it('drops the Custom loss — it needs a companion class pick', () => {
    // Sweeping loss over [.., Custom] would abort the study on the first
    // Custom trial ("pick a registered module"), so it's never offered.
    expect(sweepableChoices('loss', ['CrossEntropyLoss', 'MSELoss', 'Custom']))
      .toEqual(['CrossEntropyLoss', 'MSELoss'])
  })

  it('leaves every other enum untouched', () => {
    expect(sweepableChoices('optimizer', ['Adam', 'SGD'])).toEqual(['Adam', 'SGD'])
    expect(sweepableChoices('scheduler', undefined)).toEqual([])
  })
})
