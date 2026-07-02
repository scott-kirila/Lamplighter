import { describe, expect, it } from 'vitest'
import { nodeColor } from './nodeColor'

describe('nodeColor', () => {
  it('maps Output specially, even though it shares the io category', () => {
    expect(nodeColor('io', 'Output')).toBe('var(--node-output)')
  })

  it('maps Input (io) to the io token', () => {
    expect(nodeColor('io', 'Input')).toBe('var(--node-io)')
  })

  it('maps each category to its token', () => {
    expect(nodeColor('layers', 'Linear')).toBe('var(--node-layers)')
    expect(nodeColor('activations', 'ReLU')).toBe('var(--node-activations)')
    expect(nodeColor('ops', 'Concat')).toBe('var(--node-ops)')
  })

  it('falls back to default for unknown or missing category', () => {
    expect(nodeColor(undefined, undefined)).toBe('var(--node-default)')
    expect(nodeColor('mystery', 'Whatever')).toBe('var(--node-default)')
  })
})
