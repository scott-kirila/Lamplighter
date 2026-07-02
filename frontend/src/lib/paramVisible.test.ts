import { describe, expect, it } from 'vitest'
import { paramVisible } from './paramVisible'
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
