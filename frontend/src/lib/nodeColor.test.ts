import { describe, expect, it } from 'vitest'
import { nodeColor, nodeTextColor } from './nodeColor'

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

describe('nodeTextColor', () => {
  it('pairs each fill token with its own foreground token', () => {
    expect(nodeTextColor('var(--node-ops)')).toBe('var(--node-ops-fg)')
    expect(nodeTextColor('var(--node-activations)')).toBe('var(--node-activations-fg)')
    expect(nodeTextColor('var(--node-output)')).toBe('var(--node-output-fg)')
  })

  // The pairing is what stops the two from drifting, so every colour the
  // category mapper can produce must have a foreground.
  it('covers every fill nodeColor can return', () => {
    const fills = [
      nodeColor('io', 'Input'), nodeColor('io', 'Output'), nodeColor('layers', 'Linear'),
      nodeColor('activations', 'ReLU'), nodeColor('ops', 'Concat'), nodeColor(undefined, undefined),
    ]
    for (const fill of fills) {
      expect(nodeTextColor(fill)).toBe(fill.replace(/\)$/, '-fg)'))
    }
  })

  it('falls back rather than emitting a token that does not exist', () => {
    expect(nodeTextColor('#ff0000')).toBe('var(--node-fg)')
    expect(nodeTextColor('var(--accent)')).toBe('var(--node-fg)')
    expect(nodeTextColor('')).toBe('var(--node-fg)')
  })
})
