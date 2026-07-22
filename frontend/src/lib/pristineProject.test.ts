import { describe, it, expect } from 'vitest'
import { isPristineProject } from './pristineProject'

const node = (nodeType: string) => ({ data: { nodeType } })
const seeded = {
  models: [{ id: 'model' }],
  dataNodes: [],
  nodes: [node('Input'), node('Output')],
  edges: [],
}

describe('isPristineProject', () => {
  it('recognises the freshly seeded scaffold', () => {
    expect(isPristineProject(seeded)).toBe(true)
  })

  it('accepts a partially seeded canvas (registry missing a node type)', () => {
    expect(isPristineProject({ ...seeded, nodes: [node('Input')] })).toBe(true)
    expect(isPristineProject({ ...seeded, nodes: [] })).toBe(true)
  })

  // Each of these means the user has made a decision, and the offer to replace
  // the project stops being free.
  it('backs off the moment anything has been built', () => {
    expect(isPristineProject({ ...seeded, nodes: [node('Input'), node('Linear'), node('Output')] })).toBe(false)
    expect(isPristineProject({ ...seeded, edges: [{}] })).toBe(false)
    expect(isPristineProject({ ...seeded, dataNodes: [{ id: 'data' }] })).toBe(false)
    expect(isPristineProject({ ...seeded, models: [{ id: 'a' }, { id: 'b' }] })).toBe(false)
  })

  // A wired Input→Output is a real (if trivial) model, not a scaffold.
  it('treats a wire between the seeds as work', () => {
    expect(isPristineProject({ ...seeded, edges: [{ id: 'e' }] })).toBe(false)
  })
})
